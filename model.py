import math
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import L1Loss, MSELoss
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.init import xavier_normal_ as _xavier_normal_
from transformers import AutoModel
from scipy.spatial.distance import pdist, squareform
import scipy.sparse.linalg

# compatibility alias for legacy code blocks
def xavier_normal(tensor):
    _xavier_normal_(tensor)


try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = torch.device("cpu")


class MIB_BertModel(nn.Module):
    """Backbone text encoder wrapper with projection to common dim."""
    def __init__(self, model_name: str, d_l: int = 50):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        try:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False
        except Exception:
            pass
        hidden = getattr(self.model.config, "hidden_size", 768)
        # project token hidden to d_l
        self.proj = nn.Conv1d(in_channels=hidden, out_channels=d_l, kernel_size=1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden = out.last_hidden_state  # [B, T, H]
        x = last_hidden.transpose(1, 2)     # [B, H, T]
        x = self.proj(x)                    # [B, d_l, T]
        x = x.transpose(1, 2)               # [B, T, d_l]
        return x  # sequence features in d_l


class DIB(nn.Module):
    """Full model: text backbone + A/V projection + temporal encoders + fusion"""
    def __init__(self, model_name: str, multimodal_config=None, text_dim=768, visual_dim=35, acoustic_dim=74, num_latents: int = 5, d_l: int = 50, mi_weight: float = 1.0):
        super().__init__()
        self.config = multimodal_config
        self.d_l = d_l
        self.mi_weight = float(mi_weight)

        # text backbone
        self.bert = MIB_BertModel(model_name, d_l=d_l)

        # project audio/visual to d_l via 1x1 conv on feature dim
        self.proj_a = nn.Conv1d(in_channels=acoustic_dim, out_channels=d_l, kernel_size=1)
        self.proj_v = nn.Conv1d(in_channels=visual_dim, out_channels=d_l, kernel_size=1)

        # simple temporal encoders (TransformerEncoder)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_l, nhead=5, batch_first=True)
        self.transa = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.transv = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # fusion
        self.fusion = My_concatFusion(input_dim=self.d_l, output_dim=1)
        # self.fusion = My_DIB_Fusion(num_latents=num_latents, d_l=d_l)

        self.loss_fct = MSELoss()

    # optional: plm freeze/unfreeze for main.py hooks
    def freeze_plm(self):
        for p in self.bert.parameters():
            p.requires_grad = False
        self.bert.eval()

    def unfreeze_plm(self):
        for p in self.bert.parameters():
            p.requires_grad = True
        self.bert.train()

    def _init_custom_weights(self):
        # lightweight xavier init for new layers
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        label_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        # Text sequence features in d_l
        output_l = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # [B,T,d_l]

        # A/V: input comes [B,T,D], conv1d expects [B,D,T]
        acoustic_t = acoustic.transpose(1, 2)               # [B,D_a,T]
        visual_t = visual.transpose(1, 2)                   # [B,D_v,T]
        proj_a = self.proj_a(acoustic_t).transpose(1, 2)    # [B,T,d_l]
        proj_v = self.proj_v(visual_t).transpose(1, 2)      # [B,T,d_l]
        out_a = self.transa(proj_a)                         # [B,T,d_l]
        out_v = self.transv(proj_v)                         # [B,T,d_l]

        # attention_mask-aware average pooling over time
        if attention_mask is not None:
            mask = attention_mask.float().unsqueeze(-1)  # [B,T,1]
            valid = mask.sum(dim=1).clamp_min(1.0)
            l_avg = (output_l * mask).sum(dim=1) / valid
            a_avg = (out_a * mask).sum(dim=1) / valid
            v_avg = (out_v * mask).sum(dim=1) / valid
        else:
            l_avg = output_l.mean(dim=1)
            a_avg = out_a.mean(dim=1)
            v_avg = out_v.mean(dim=1)

        logits, mi_loss = self.fusion(l_avg, a_avg, v_avg)
        logits = logits.squeeze(-1)

        loss = self.loss_fct(logits.view(-1).float(), label_ids.view(-1).float())
        
        
        if  mi_loss!=0:
            if torch.isfinite(mi_loss).all():
                loss = loss + self.mi_weight * mi_loss

        return logits, loss


class My_DIB_Fusion(nn.Module):
    """Cross-modal attention + VAE-style latent + MI loss (robust)"""
    def __init__(self, num_latents: int, d_l: int):
        super().__init__()
        self.d_l = d_l
        self.num_latents = num_latents

        # Per-modality encoders
        def enc_block(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 1024), nn.ReLU(inplace=True),
            )
        self.encoder_l = enc_block(d_l)
        self.encoder_a = enc_block(d_l)
        self.encoder_v = enc_block(d_l)
        self.encoder = enc_block(d_l * 4)

        # Latent heads
        self.fc_mu_l = nn.Linear(1024, d_l)
        self.fc_std_l = nn.Linear(1024, d_l)
        self.fc_mu_a = nn.Linear(1024, d_l)
        self.fc_std_a = nn.Linear(1024, d_l)
        self.fc_mu_v = nn.Linear(1024, d_l)
        self.fc_std_v = nn.Linear(1024, d_l)
        self.fc_mu = nn.Linear(1024, d_l)
        self.fc_std = nn.Linear(1024, d_l)

        # Decoder
        def dec_block():
            m = nn.Sequential(
                nn.Linear(d_l, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1)
            )
            # 将最后一层偏置初始化为 0，避免强常数项影响早期学习
            if isinstance(m[-1], nn.Linear) and m[-1].bias is not None:
                nn.init.zeros_(m[-1].bias)
            return m
        self.decoder = dec_block()

        # cross-modal attention
        self.cross_modal_attn = nn.MultiheadAttention(embed_dim=d_l, num_heads=5, batch_first=True)

        # pre-encode norm and residual skip from attn_fusion
        self.pre_encode_ln = nn.LayerNorm(d_l * 4)
        # 放大初值，保证早期就有可变输出和有效梯度通路
        self.skip_gate = nn.Parameter(torch.tensor(1.0))
        self.skip = nn.Linear(d_l, 1)

        # MI buffers to stabilize small-batch MI
        self.mi_buf_size = 32
        self.register_buffer('mi_buf_z', torch.empty(0, d_l))
        self.register_buffer('mi_buf_f', torch.empty(0, d_l * 4))

    # --- helpers ---
    def _std(self, x):
        return F.softplus(x - 5.0, beta=1.0)

    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self._std(self.fc_std(x))

    def encode_l(self, x):
        x = self.encoder_l(x)
        return self.fc_mu_l(x), self._std(self.fc_std_l(x))

    def encode_a(self, x):
        x = self.encoder_a(x)
        return self.fc_mu_a(x), self._std(self.fc_std_a(x))

    def encode_v(self, x):
        x = self.encoder_v(x)
        return self.fc_mu_v(x), self._std(self.fc_std_v(x))

    def reparameterise(self, mu, std):
        eps = torch.randn_like(std)
        return mu + std * eps

    # pairwise/gram for MI
    def pairwise_distances(self, x):
        bn = x.shape[0]
        x = x.view(bn, -1)
        inst_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
        return -2 * torch.mm(x, x.t()) + inst_norm + inst_norm.t()

    def calculate_gram_mat(self, x, sigma):
        # 强制在 FP32 下计算核矩阵，提升稳定性
        with torch.amp.autocast('cuda', enabled=False):
            x32 = x.float()
            dist = self.pairwise_distances(x32)
            sigma_safe = max(float(sigma), 1e-6)
            K = torch.exp(-dist / sigma_safe)
            # 对称化，避免数值误差导致非对称
            K = 0.5 * (K + K.transpose(0, 1))
            if self.training:
                # 更强抖动，且与规模相关，缓解病态情况
                jitter = 1e-5 * K.shape[0]
                I = torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
                K = K + jitter * I
        return K

    def calculate_lowrank(self, A, alpha, k, v, batch_size=48):
        n = A.shape[0]
        if n < 2:
            return torch.tensor(1e-6, device=A.device)
        if k >= n:
            k = max(1, n - 1)
        if n != batch_size:
            v = np.random.randn(n)
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        try:
            if n <= 64:
                eigs = torch.linalg.eigvalsh(A)
                eigs = torch.clamp(eigs, min=0)
            else:
                A_np = A.detach().cpu().numpy().astype(np.float64, copy=False)
                use_k = min(k, n - 1)
                _, U = scipy.sparse.linalg.eigsh(A_np, k=use_k, v0=v[:n], ncv=min(12, max(2, n)), tol=1e-1)
                U = torch.from_numpy(U).to(A.device)
                eigs = torch.clamp(torch.linalg.eigvalsh(torch.mm(U.t(), torch.mm(A, U))), min=0)
            tr = torch.sum(eigs ** alpha) + (n - k) * torch.clamp((1 - torch.sum(eigs)) / max(1, (n - k)), min=0) ** alpha
            return (1 / (1 - alpha)) * torch.log2(torch.clamp(tr, min=1e-6))
        except Exception as _e:
            print(f"[DEBUG] eigsh failed: {_e}")
            return torch.tensor(1e-6, device=A.device)

    def calculate_MI_lowrank(self, x, y, s_x, s_y, alpha, k, v):
        ky = self.calculate_gram_mat(y, s_y)
        tr = torch.trace(ky)
        if tr <= 1e-8 or not torch.isfinite(tr):
            tr = torch.tensor(1.0, device=ky.device)
        ky = ky / tr
        return self.calculate_lowrank(ky, alpha, k, v)

    def forward(self, x_l, x_a, x_v):
        # enc per modality
        mu_l, std_l = self.encode_l(x_l)
        z_l = self.reparameterise(mu_l, std_l)
        mu_a, std_a = self.encode_a(x_a)
        z_a = self.reparameterise(mu_a, std_a)
        mu_v, std_v = self.encode_v(x_v)
        z_v = self.reparameterise(mu_v, std_v)

        # cross-modal attn
        z_stack = torch.stack([z_l, z_a, z_v], dim=1)
        attn_output, _ = self.cross_modal_attn(z_stack, z_stack, z_stack)
        attn_fusion = attn_output.mean(dim=1)

        # fuse + latent
        f = torch.cat([attn_fusion, z_l, z_a, z_v], dim=-1)
        f = self.pre_encode_ln(f)
        mu, std = self.encode(f)
        z = self.reparameterise(mu, std)
        if self.training and z.shape[0] < 4:
            z = z + 1e-3 * torch.randn_like(z)
        out = self.decoder(z) + self.skip_gate * self.skip(attn_fusion)

        # MI over buffer
        def robust_sigma(Z_numpy, last_sigma=[1.0]):
            try:
                k_z = squareform(pdist(Z_numpy, 'euclidean'))
                if k_z.size > 1:
                    iu = np.triu_indices_from(k_z, k=1)
                    vals = k_z[iu]
                    vals = vals[vals > 0]
                    sigma = np.median(vals) if vals.size > 0 else np.mean(k_z)
                else:
                    sigma = float(last_sigma[0])
                if sigma < 1e-6:
                    sigma = last_sigma[0]
                else:
                    last_sigma[0] = sigma
            except Exception:
                sigma = last_sigma[0]
            return sigma

        # Build temporary buffers with current batch to compute MI loss with gradient
        tmp_f = torch.cat([self.mi_buf_f.detach(), f], dim=0)[-self.mi_buf_size:]
        tmp_z = torch.cat([self.mi_buf_z.detach(), z], dim=0)[-self.mi_buf_size:]
        n = tmp_z.shape[0]
        alpha = 1.9
        if n >= 8:
            # sigma estimated from detached numpy (no grad needed for sigma)
            try:
                Z_numpy = tmp_z.detach().cpu().numpy()
                sigma_z = robust_sigma(Z_numpy)
            except Exception:
                sigma_z = 1.0
            # 在 FP32 且关闭 AMP 下计算 MI，避免半精度导致的本征分解不收敛
            with torch.amp.autocast('cuda', enabled=False):
                Ky = self.calculate_gram_mat(tmp_z.float(), sigma_z ** 2)
                # 归一化 trace 并再次对称化
                tr = torch.trace(Ky)
                if not torch.isfinite(tr) or float(tr.item()) <= 1e-8:
                    tr = torch.tensor(1.0, device=Ky.device, dtype=Ky.dtype)
                Ky = Ky / tr
                Ky = 0.5 * (Ky + Ky.transpose(0, 1))
                # 再次加小抖动，确保严格正定
                eps = 1e-6 * Ky.shape[0]
                Ky = Ky + eps * torch.eye(Ky.shape[0], device=Ky.device, dtype=Ky.dtype)
                # Renyi entropy surrogate via eigenvalues
                try:
                    eigs = torch.linalg.eigvalsh(Ky)
                except Exception:
                    # 回退：使用奇异值或均匀分布，保持训练不中断
                    try:
                        s = torch.linalg.svdvals(Ky)
                        eigs = torch.clamp(s, min=1e-12)
                    except Exception:
                        eigs = torch.full((Ky.shape[0],), 1.0 / max(Ky.shape[0], 1), device=Ky.device, dtype=Ky.dtype)
                eigs = torch.clamp(eigs, min=1e-12)
                tr_alpha = torch.sum(eigs ** alpha)
                mi_loss = (1.0 / (1.0 - alpha)) * torch.log2(torch.clamp(tr_alpha, min=1e-12))
        else:
            mi_loss = torch.tensor(0.0, device=out.device)

        # Update persistent buffers without gradient tracking (for next steps)
        with torch.no_grad():
            self.mi_buf_f = torch.cat([self.mi_buf_f, f.detach()], dim=0)[-self.mi_buf_size:]
            self.mi_buf_z = torch.cat([self.mi_buf_z, z.detach()], dim=0)[-self.mi_buf_size:]

        return out, mi_loss

    def test(self, x_l, x_a, x_v):
        mu_l, std_l = self.encode_l(x_l)
        z_l = self.reparameterise(mu_l, std_l)
        mu_a, std_a = self.encode_a(x_a)
        z_a = self.reparameterise(mu_a, std_a)
        mu_v, std_v = self.encode_v(x_v)
        z_v = self.reparameterise(mu_v, std_v)
        z_stack = torch.stack([z_l, z_a, z_v], dim=1)
        attn_output, _ = self.cross_modal_attn(z_stack, z_stack, z_stack)
        attn_fusion = attn_output.mean(dim=1)
        f = torch.cat([attn_fusion, z_l, z_a, z_v], dim=-1)
        f = self.pre_encode_ln(f)
        mu, std = self.encode(f)
        z = self.reparameterise(mu, std)
        return self.decoder(z) + self.skip_gate * self.skip(attn_fusion)


class My_concatFusion(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=50, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, output_dim)
        )
        
        # 改善权重初始化
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, l, a, v):
        # 检查输入是否包含NaN或无穷大
        if torch.isnan(l).any() or torch.isnan(a).any() or torch.isnan(v).any():
            print(f"[WARNING] NaN detected in fusion inputs!")
            l = torch.nan_to_num(l, nan=0.0, posinf=1.0, neginf=-1.0)
            a = torch.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0) 
            v = torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # # Debug: 输入的均值和方差 (暂时关闭调试信息)
        # print(f"[DEBUG] l.mean: {l.mean().item():.6f}, std: {l.std().item():.6f}")
        # print(f"[DEBUG] a.mean: {a.mean().item():.6f}, std: {a.std().item():.6f}")
        # print(f"[DEBUG] v.mean: {v.mean().item():.6f}, std: {v.std().item():.6f}")

        # # Debug: 权重平均绝对值
        # print(f"[DEBUG] fusion.weight.mean_abs: {self.net[0].weight.data.abs().mean().item():.6f}")

        fused = torch.cat([l, a, v], dim=-1)
        # print(f"[DEBUG] fused.mean: {fused.mean().item():.6f}, std: {fused.std().item():.6f}")
        
        output = self.net(fused)
        # print(f"[DEBUG] fusion_output.mean: {output.mean().item():.6f}, std: {output.std().item():.6f}")
        
        # 检查输出
        if torch.isnan(output).any():
            print(f"[WARNING] NaN in fusion output!")
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return output, 0

    def test(self, l, a, v):
        return self.forward(l, a, v)



