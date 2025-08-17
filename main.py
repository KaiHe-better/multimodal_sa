from __future__ import absolute_import, division, print_function
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="5")
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei", "chsims"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=32)  
parser.add_argument("--dev_batch_size", type=int, default=64)   
parser.add_argument("--test_batch_size", type=int, default=64)  
parser.add_argument("--n_epochs", type=int, default=500)          
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.3) # 优化：减少正则化压力
parser.add_argument("--model", type=str, choices=["bert-base-uncased", "bert-base-chinese", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"], default= "bert-base-uncased",help="Pretrained transformer model to use")
parser.add_argument("--learning_rate", type=float, default=5e-6) # 更稳：降低默认学习率
parser.add_argument("--plm_lr", type=float, default=5e-6, help="PLM 参数组学习率，默认 5e-6")
parser.add_argument("--fusion_lr", type=float, default=3e-4, help="融合/解码等非 PLM 参数组学习率，默认 3e-4")
parser.add_argument("--gradient_accumulation_step", type=int, default=8)  # 提高累积步数，平滑更新
parser.add_argument("--warmup_proportion", type=float, default=0.2)  # 优化：更温和的预热
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--save_results", action="store_true", help="Save results to output directory")
parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
parser.add_argument("--save_interval", type=int, default=5, help="Save training progress plot and results summary every N epochs")
parser.add_argument("--limit_steps_enabled", action="store_true", help="启用后训练/验证/测试各阶段最多只运行指定步数")
parser.add_argument("--limit_steps", type=int, default=30, help="当启用limit_steps_enabled时，每个阶段最多运行的步数")
parser.add_argument("--freeze_plm_epochs", type=int, default=10, help="前N个epoch冻结预训练语言模型(Qwen/BERT)仅训练自定义模块；N之后解冻")
parser.add_argument("--mi_weight", type=float, default=0.25, help="MI 正则项的权重缩放（乘到模型内部的 mi_loss 上），推荐区间 0.15–0.35")
parser.add_argument("--mi_sweep", type=str, default="", help="逗号分隔的一组 MI 权重，留空表示不做 sweep。例如: 0.0,0.5,1.0,2.0")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  


import random
import pickle
import numpy as np
import datetime
import json
import sys
import warnings
import traceback
from typing import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import os

# 自动选择设备：优先CUDA，否则CPU；若CUDA初始化异常也回退CPU
try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = torch.device("cpu")
print(f"[INFO] Using device: {DEVICE}")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*The `device` argument is deprecated.*")
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*is deprecated.*")


# 添加matplotlib导入用于可视化（seaborn 可选）
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    sns = None
    HAS_SEABORN = False

# 设置绘图样式
plt.style.use('default')
if HAS_SEABORN:
    try:
        sns.set_palette("husl")
    except Exception:
        pass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from model import DIB
from global_configs import get_config








config = get_config(args.dataset)
ACOUSTIC_DIM = config["ACOUSTIC_DIM"]
VISUAL_DIM = config["VISUAL_DIM"]
TEXT_DIM = config["TEXT_DIM"]

# 统一拆包模型输出，兼容返回 Tensor 或 (logits, loss)
def _split_model_outputs(outputs):
    if isinstance(outputs, (tuple, list)):
        logits = outputs[0]
        aux_loss = outputs[1] if len(outputs) > 1 else None
        return logits, aux_loss
    return outputs, None

# Create a class to capture terminal output
class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def return_unk():
    return 0


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob


def convert_to_features(examples, max_seq_length, tokenizer, visual_dim, acoustic_dim, dataset_name):
    features = []

    # Ensure all modalities are exactly max_seq_length in length
    def pad_or_truncate(x, length):
        if isinstance(x, list):
            if len(x) > length:
                return x[:length]
            else:
                return x + [0] * (length - len(x))
        elif isinstance(x, np.ndarray):
            if x.shape[0] > length:
                return x[:length]
            else:
                pad_len = length - x.shape[0]
                padding = np.zeros((pad_len, x.shape[1]))
                return np.concatenate([x, padding], axis=0)

    for (ex_index, example) in enumerate(examples):
        (words, visual, acoustic), label_id, segment = example
        
        # 更严格的数据过滤 - 检查原始数据
        if (visual is None or acoustic is None or 
            np.isnan(acoustic).any() or np.isnan(visual).any() or 
            np.isinf(acoustic).any() or np.isinf(visual).any()):
            print(f"[WARNING] Skipping sample {ex_index} due to invalid data")
            continue
            
        # 数据集特异性的数值范围检查
        if dataset_name == 'chsims':
            # CHSIMS数据集使用更宽松的阈值
            if (np.abs(acoustic).max() > 1000 or np.abs(visual).max() > 50000000):
                print(f"[WARNING] Skipping sample {ex_index} due to extreme values")
                continue
        else:
            # MOSI/MOSEI数据集使用原有阈值
            if (np.abs(acoustic).max() > 100 or np.abs(visual).max() > 100):
                print(f"[WARNING] Skipping sample {ex_index} due to extreme values")
                continue
            
        # 标准化数据范围 - 使用数据集特异性的范围
        if dataset_name == 'chsims':
            # CHSIMS数据集使用更大的clip范围
            acoustic = np.clip(acoustic, -800, 800)
            visual = np.clip(visual, -300, 300)
        else:
            # MOSI/MOSEI数据集使用原有保守范围
            acoustic = np.clip(acoustic, -5, 5)
            visual = np.clip(visual, -5, 5)
        
        # 额外的数值检查
        acoustic = np.nan_to_num(acoustic, nan=0.0, posinf=1.0, neginf=-1.0)
        visual = np.nan_to_num(visual, nan=0.0, posinf=1.0, neginf=-1.0)
        
        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        # 确保正确设置prepare_input函数
        prepare_input = prepare_bert_input

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
        tokens, visual, acoustic, tokenizer, visual_dim, acoustic_dim
        )

        input_ids = pad_or_truncate(input_ids, args.max_seq_length)
        input_mask = pad_or_truncate(input_mask, args.max_seq_length)
        segment_ids = pad_or_truncate(segment_ids, args.max_seq_length)
        acoustic = pad_or_truncate(acoustic, args.max_seq_length)
        visual = pad_or_truncate(visual, args.max_seq_length)

        # 最终的数值检查
        if (isinstance(acoustic, np.ndarray) and 
            (np.isnan(acoustic).any() or np.isinf(acoustic).any())):
            print(f"[WARNING] Skipping sample {ex_index} due to invalid acoustic after processing")
            continue
            
        if (isinstance(visual, np.ndarray) and 
            (np.isnan(visual).any() or np.isinf(visual).any())):
            print(f"[WARNING] Skipping sample {ex_index} due to invalid visual after processing")
            continue

        label_id = float(label_id.item()) if isinstance(label_id, np.ndarray) else float(label_id)
        
        # 检查标签是否有效
        if np.isnan(label_id) or np.isinf(label_id):
            print(f"[WARNING] Skipping sample {ex_index} due to invalid label")
            continue

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
        
    print(f"[INFO] Converted {len(features)} valid samples from {len(examples)} total samples")
    return features


# def prepare_bert_input(tokens, visual, acoustic, tokenizer):
def prepare_bert_input(tokens, visual, acoustic, tokenizer, visual_dim, acoustic_dim):
    # 兼容无CLS/SEP的自回归模型（如Qwen），优先使用bos/eos
    cls_token = getattr(tokenizer, 'cls_token', None) or getattr(tokenizer, 'bos_token', None)
    sep_token = getattr(tokenizer, 'sep_token', None) or getattr(tokenizer, 'eos_token', None)
    # 仅在存在时添加，避免产生 None
    tokens = (([cls_token] if cls_token else []) + tokens + ([sep_token] if sep_token else []))
    # 过滤掉潜在的 None 或空字符串，确保 convert_tokens_to_ids 安全
    tokens = [t for t in tokens if isinstance(t, str) and len(t) > 0]

    if acoustic.ndim != 2 or visual.ndim != 2:
        raise ValueError(f"Input acoustic or visual is not 2D. Got: "
                         f"acoustic.shape={acoustic.shape}, visual.shape={visual.shape}")

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    # acoustic_zero = np.zeros((1, ACOUSTIC_DIM))  # (1,74)
    # acoustic_zero = np.zeros((1, acoustic.shape[1]))
    acoustic_zero = np.zeros((1, acoustic_dim))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))  # (22+2=24,74)
    # visual_zero = np.zeros((1, VISUAL_DIM))  
    # visual_zero = np.zeros((1, visual.shape[1]))
    visual_zero = np.zeros((1, visual_dim))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    # 将token转换为ids；对不识别的token，HF会返回unk
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, acoustic.shape[1]))
    visual_padding = np.zeros((pad_length, visual.shape[1]))
    # acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))  # (26,74)
    acoustic = np.concatenate((acoustic, acoustic_padding))  # (50,74)


    # visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))


    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids


def get_default_model_for_dataset(dataset):
    if dataset in ["mosi", "mosei"]:
        return "bert-base-uncased"
    elif dataset == "chsims":
        return "bert-base-chinese"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def get_appropriate_dataset(data):

    tokenizer = get_tokenizer(args.model)

    features = convert_to_features(data, args.max_seq_length, tokenizer, VISUAL_DIM, ACOUSTIC_DIM, args.dataset)

    
    # 优化tensor创建，避免性能警告
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    # 使用numpy.array()然后转换为tensor以提高性能
    import numpy as np
    visual_array = np.array([f.visual for f in features])
    acoustic_array = np.array([f.acoustic for f in features])
    all_visual = torch.tensor(visual_array, dtype=torch.float)
    all_acoustic = torch.tensor(acoustic_array, dtype=torch.float)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    return dataset


def set_up_data_loader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        * args.n_epochs
    )
    print(len(train_dataset))

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
 
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prep_for_training(num_train_optimization_steps: int, mi_weight: float = 1.0):
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
    )

    model = DIB(
        args.model,
        multimodal_config=multimodal_config,
        text_dim=TEXT_DIM,
        visual_dim=VISUAL_DIM,
        acoustic_dim=ACOUSTIC_DIM,
        mi_weight=mi_weight,
    )
   
    total_para = 0
    for param in model.parameters():
        total_para += np.prod(param.size())
    print('total parameter for the model: ', total_para)

    model.to(DEVICE)
    # 按需冻结预训练语言模型参数
    if args.freeze_plm_epochs and args.freeze_plm_epochs > 0:
        try:
            model.freeze_plm()
        except Exception as e:
            print(f"[WARNING] Initial freeze_plm failed: {e}")
    
    # 在模型移动到设备后进行权重初始化
    print("执行自定义权重初始化...")
    model._init_custom_weights()

    # 创建优化器参数组：PLM 使用较低 LR，其余模块使用较高 LR
    # 使用命令行超参数控制两组学习率
    plm_lr = float(args.plm_lr)
    fusion_lr = float(args.fusion_lr)
    # 提取 PLM 参数（HuggingFace 模型在 model.bert.model 下）
    try:
        plm_params = list(model.bert.model.parameters())
    except Exception:
        plm_params = list(model.bert.parameters())
    plm_param_ids = {id(p) for p in plm_params}
    other_params = [p for p in model.parameters() if id(p) not in plm_param_ids]
    print(f"Optimizer groups -> PLM: {len(plm_params)} params @ lr={plm_lr:.2e}, Other: {len(other_params)} params @ lr={fusion_lr:.2e}")
    optimizer = AdamW([
        {"params": plm_params, "lr": plm_lr},
        {"params": other_params, "lr": fusion_lr},
    ], eps=1e-8, weight_decay=0.01)
    
    # 添加学习率调度器 - 进一步提高稳定性
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
    
    return model, optimizer, scheduler


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, epoch_num=0):
    model.train()
    # 如果设置了冻结周期且当前epoch达到解冻点，则解冻PLM（只触发一次）
    try:
        if getattr(args, 'freeze_plm_epochs', 0) > 0 and epoch_num == args.freeze_plm_epochs:
            # 解冻发生在进入该epoch训练前
            if hasattr(model, 'unfreeze_plm'):
                model.unfreeze_plm()
        # 在冻结阶段（epoch < freeze_plm_epochs）保持PLM为eval，避免被上面的model.train()改回train
        if getattr(args, 'freeze_plm_epochs', 0) > 0 and epoch_num < args.freeze_plm_epochs:
            if hasattr(model, 'freeze_plm'):
                # 只确保模式，不重复打印
                try:
                    for p in model.bert.model.parameters():
                        p.requires_grad = False
                    model.bert.model.eval()
                except Exception:
                    model.freeze_plm()
    except Exception as _e:
        print(f"[WARNING] unfreeze check failed: {_e}")
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    valid_steps = 0  # 记录有效的训练步数
    use_amp = torch.cuda.is_available()
    # 若使用 Qwen/Qwen3-1.7B，强制关闭 AMP 以保证全精度
    try:
        if isinstance(args.model, str) and 'qwen' in args.model.lower():
            use_amp = False
    except Exception:
        pass
    from torch import amp as torch_amp
    use_bf16 = False
    if use_amp:
        # 优先使用bfloat16以更稳定的数值表现（在A100/H100等设备上）
        try:
            use_bf16 = torch.cuda.is_bf16_supported()
        except Exception:
            use_bf16 = False
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch_amp.GradScaler('cuda', enabled=(use_amp and not use_bf16))
    
    # 自适应梯度裁剪阈值 - 根据训练阶段调整
    if epoch_num < 5:
        clip_threshold = 3.0  # 早期训练：严格裁剪
        log_threshold = 2.5
    elif epoch_num < 15:
        clip_threshold = 5.0  # 中期训练：适中裁剪
        log_threshold = 4.0
    else:
        clip_threshold = 8.0  # 后期训练：宽松裁剪
        log_threshold = 6.0
    
    # 梯度累积设置 - 帮助稳定训练
    accumulation_steps = max(2, args.gradient_accumulation_step)  # 至少2步累积
    accumulated_loss = 0.0
    
    # print(f"[INFO] Epoch {epoch_num}: Using adaptive gradient clipping threshold: {clip_threshold}")

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch

        # 检查输入数据是否包含NaN
        if torch.isnan(visual).any() or torch.isnan(acoustic).any() or torch.isnan(label_ids).any():
            print("[WARNING] NaN detected in input data, skipping batch")
            continue

        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        
        try:
            # 使用AMP自动混合精度仅包裹前向与特征构建，损失在FP32中计算以避免类型不匹配
            with torch_amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                outputs, loss = model(  
                    input_ids,
                    visual,
                    acoustic,
                    label_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )
            # 记录logits用于调试（模型forward第一返回项即为logits）
            logits = outputs

            raw_loss_val = float(loss.detach().cpu())
            
            # 检查损失是否为NaN/Inf
            if not torch.isfinite(loss).all():
                print(f"[WARNING] Invalid loss detected, skipping batch")
                continue

            # 梯度累积 - 平均损失
            loss = loss / accumulation_steps
            accumulated_loss += loss.item()

            # 反向传播
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 只在累积完成后进行梯度更新
            if (step + 1) % accumulation_steps == 0:
                # 检查梯度是否包含NaN
                has_nan_grad = False
                bad_params = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            bad_params.append(name)
                            
                if has_nan_grad:
                    # 打印诊断信息帮助定位
                    try:
                        log_mean = float(logits.mean().detach().cpu())
                        log_std = float(logits.std().detach().cpu())
                    except Exception:
                        log_mean, log_std = 0.0, 0.0
                    print(f"[WARNING] Skipping optimizer step due to invalid gradients | raw_loss={raw_loss_val:.6f}, logits_mean={log_mean:.6f}, logits_std={log_std:.6f}")
                    if bad_params:
                        print(f"[DEBUG] Invalid grad params count: {len(bad_params)}, e.g., {bad_params[:5]}")
                    optimizer.zero_grad()
                    accumulated_loss = 0.0
                    continue
                
                # 自适应梯度剪裁
                # 先unscale再裁剪
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_threshold)
                
                # 更严格的梯度控制（不再跳步，始终裁剪后更新，极端情况仅记录）
                # if grad_norm > log_threshold:  # 记录但继续训练
                #     print(f"[INFO] Large gradient norm: {grad_norm:.2f}, clipped to {clip_threshold}")
                
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()  # 清零梯度准备下次累积
                
                tr_loss += accumulated_loss
                accumulated_loss = 0.0
                valid_steps += 1
            
            # 限制训练步数（调试/快速模式）
            if args.limit_steps_enabled and (step + 1) >= args.limit_steps:
                break
            
        except Exception as e:
            # 打印更详细的异常信息，便于定位
            print(f"[ERROR] Exception in training step: {repr(e)}")
            traceback.print_exc()
            optimizer.zero_grad()
            accumulated_loss = 0.0
            continue

        nb_tr_steps += 1  

    # 处理最后不完整的累积批次
    if accumulated_loss > 0:
        # 检查梯度是否包含NaN
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan_grad = True
                    break
                    
        if not has_nan_grad:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_threshold)
            if grad_norm <= clip_threshold * 10:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                tr_loss += accumulated_loss
                valid_steps += 1
        
    optimizer.zero_grad()  

    # 确保有有效的训练步数
    if valid_steps == 0:
        print("[ERROR] No valid training steps completed!")
        return 0.0
        
    avg_loss = tr_loss / valid_steps
    print(f"[INFO] Epoch {epoch_num} completed: {valid_steps} valid steps, avg loss: {avg_loss:.4f}")
    return avg_loss 


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    valid_steps = 0  # 记录有效的验证步数
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            
            # 检查输入数据是否包含NaN
            if torch.isnan(visual).any() or torch.isnan(acoustic).any() or torch.isnan(label_ids).any():
                print("[WARNING] NaN detected in validation input data, skipping batch")
                continue
                
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            
            try:
                # 使用与训练相同的forward方法，而不是test方法
                outputs = model(
                    input_ids,
                    visual,
                    acoustic,
                    label_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )

                logits, aux_loss = _split_model_outputs(outputs)
                
                # 检查输出是否包含NaN或inf
                if (not isinstance(logits, torch.Tensor)) or (not torch.isfinite(logits).all()):
                    print("[WARNING] Invalid outputs detected in validation, skipping batch")
                    continue

                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))
                if aux_loss is not None and torch.isfinite(aux_loss).all():
                    loss = loss + aux_loss
                
                # 检查损失是否为NaN或inf
                if not torch.isfinite(loss).all():
                    print(f"[WARNING] Invalid loss detected in validation, skipping batch")
                    continue

                if args.gradient_accumulation_step > 1:
                    loss = loss / args.gradient_accumulation_step

                dev_loss += loss.item()
                valid_steps += 1
                
            except Exception as e:
                print(f"[ERROR] Exception in validation step: {repr(e)}")
                traceback.print_exc()
                continue

            nb_dev_steps += 1
            if args.limit_steps_enabled and (step + 1) >= args.limit_steps:
                break

    # 确保有有效的验证步数
    if valid_steps == 0:
        print("[ERROR] No valid validation steps completed!")
        return float('inf')  # 返回无穷大表示验证失败
        
    return dev_loss / valid_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []
    input_ids_list = []  # Store input IDs for text reconstruction

    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader)):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            input_ids_list.extend(input_ids.cpu().numpy())
            
            # 检查输入数据是否包含NaN
            if torch.isnan(visual).any() or torch.isnan(acoustic).any() or torch.isnan(label_ids).any():
                print("[WARNING] NaN detected in test input data, skipping batch")
                continue
            
            if args.limit_steps_enabled and (step + 1) >= args.limit_steps:
                break
            
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            
            try:
                # 使用与训练相同的forward方法，而不是test方法
                outputs = model(
                    input_ids,
                    visual,
                    acoustic,
                    label_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )

                logits, _ = _split_model_outputs(outputs)
                
                # 检查输出是否包含NaN或inf
                if (not isinstance(logits, torch.Tensor)) or (not torch.isfinite(logits).all()):
                    print("[WARNING] Invalid outputs detected in test, skipping batch")
                    continue

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.detach().cpu().numpy()

                # 转为显式 float32 并展平为一维后再转list，避免奇异对象数组引发递归错误
                try:
                    logits = np.ravel(logits.astype(np.float32, copy=False)).tolist()
                    label_ids = np.ravel(label_ids.astype(np.float32, copy=False)).tolist()
                except Exception as _e:
                    print(f"[WARNING] Failed to convert arrays to 1D float list: {_e}, skipping batch")
                    continue

                preds.extend(logits)
                labels.extend(label_ids)
                
            except Exception as e:
                print(f"[ERROR] Exception in test step: {repr(e)}")
                traceback.print_exc()
                continue

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels, input_ids_list

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def save_predictions(preds, labels, input_ids, output_dir):
    """Save predictions and original sentences to a file"""
    tokenizer = get_tokenizer(args.model)
    
    with open(os.path.join(output_dir, "predictions.txt"), "w", encoding='utf-8') as f:
        f.write("Original\tPrediction\tGround Truth\n")
        
        for i in range(len(preds)):
            # Try to decode the input_ids back to text
            try:
                # Get only the non-padding tokens (non-zero values)
                valid_ids = [id for id in input_ids[i] if id != 0]
                # Skip CLS and SEP tokens for a cleaner output
                text = tokenizer.decode(valid_ids[1:-1], skip_special_tokens=True)
            except Exception as e:
                text = f"[Decoding Error: {str(e)}]"
                
            f.write(f"{text}\t{preds[i]}\t{labels[i]}\n")


def plot_training_progress(train_losses, valid_losses, test_metrics, results_dir, epoch_num):
    """
    绘制训练过程的可视化图表
    
    Args:
        train_losses: 训练损失列表
        valid_losses: 验证损失列表 
        test_metrics: 测试指标字典，包含 'acc', 'mae', 'acc7', 'f1', 'corr'
        results_dir: 结果保存目录
        epoch_num: 当前epoch数量
    """
    try:
        if not results_dir:
            return
            
        epochs = list(range(1, len(train_losses) + 1))
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Training Progress - Epoch {epoch_num}', fontsize=16, fontweight='bold')
        
        # 第一张图：训练指标
        axes[0, 0].plot(epochs, train_losses, 'b-', marker='o', linewidth=2, markersize=4, label='Train Loss')
        axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 第二张图：验证指标
        axes[0, 1].plot(epochs, valid_losses, 'r-', marker='s', linewidth=2, markersize=4, label='Valid Loss')
        axes[0, 1].set_title('Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 第三张图：测试指标 - 准确率相关
        if 'acc' in test_metrics and 'acc7' in test_metrics:
            axes[1, 0].plot(epochs, test_metrics['acc'], 'g-', marker='^', linewidth=2, markersize=4, label='Test Acc')
            axes[1, 0].plot(epochs, test_metrics['acc7'], 'orange', marker='d', linewidth=2, markersize=4, label='Acc7')
            if 'f1' in test_metrics:
                axes[1, 0].plot(epochs, test_metrics['f1'], 'purple', marker='v', linewidth=2, markersize=4, label='F1 Score')
        axes[1, 0].set_title('Test Accuracy Metrics', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)
        
        # 第四张图：测试指标 - MAE和相关系数
        ax2 = axes[1, 1]
        if 'mae' in test_metrics:
            ax2.plot(epochs, test_metrics['mae'], 'red', marker='o', linewidth=2, markersize=4, label='MAE')
            ax2.set_ylabel('MAE', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        
        # 添加第二个y轴显示相关系数
        if 'corr' in test_metrics:
            ax3 = ax2.twinx()
            ax3.plot(epochs, test_metrics['corr'], 'blue', marker='s', linewidth=2, markersize=4, label='Correlation')
            ax3.set_ylabel('Correlation', color='blue')
            ax3.tick_params(axis='y', labelcolor='blue')
            ax3.set_ylim(0, 1)
        
        ax2.set_title('Test MAE & Correlation', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.grid(True, alpha=0.3)
        
        # 组合图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        if 'corr' in test_metrics:
            lines2, labels2 = ax3.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax2.legend()
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(results_dir, f"training_progress_epoch_{epoch_num}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Training progress plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"[WARNING] Failed to create training progress plot: {e}")
        plt.close()  # 确保关闭图形以释放内存


def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False, output_dir=None):
    preds, y_test, input_ids = test_epoch(model, test_dataloader)
    
    # Save predictions if output directory is provided
    if output_dir:
        save_predictions(preds, y_test, input_ids, output_dir)
    
    # non_zeros = np.array(
    #     [i for i, e in enumerate(y_test) if e != 0 or use_zero])

    # 统一为numpy数组
    preds = np.asarray(preds)
    y_test = np.asarray(y_test)

    # 为空时直接返回默认值，避免后续下标/除零错误
    if preds.size == 0 or y_test.size == 0:
        return 0.0, float('inf'), 0.0, 0.0, 0.0

    # 采用布尔掩码，避免空数组成为float类型索引
    if use_zero:
        mask = np.ones_like(y_test, dtype=bool)
    else:
        mask = (y_test != 0)

    if mask.sum() == 0:
        # 没有非零样本，降级为在全部样本上计算受限指标
        mask = np.ones_like(y_test, dtype=bool)

    test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    # 若样本数为0，multiclass_acc内部会出错；此处已确保非空
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7) if len(test_truth_a7) > 0 else 0.0

    preds_non_zeros = preds[mask]
    y_test_non_zeros = y_test[mask]

    if preds_non_zeros.size == 0:
        return 0.0, float('inf'), 0.0, 0.0, mult_a7

    mae = float(np.mean(np.absolute(preds_non_zeros - y_test_non_zeros)))
    # 防止相关系数出现nan
    try:
        corr = float(np.corrcoef(preds_non_zeros, y_test_non_zeros)[0][1])
        if np.isnan(corr):
            corr = 0.0
    except Exception:
        corr = 0.0

    binary_preds = preds_non_zeros >= 0
    binary_y_test = y_test_non_zeros >= 0

    try:
        f_score = float(f1_score(binary_y_test, binary_preds, average="weighted"))
    except Exception:
        f_score = 0.0
    try:
        acc = float(accuracy_score(binary_y_test, binary_preds))
    except Exception:
        acc = 0.0

    return acc, mae, corr, f_score, mult_a7


def calibrate_threshold_on_dev(model: nn.Module, dev_dataloader: DataLoader, search_min=-0.6, search_max=0.6, step=0.02):
    """在开发集上扫描阈值，最大化 F1。返回最佳阈值与对应指标。"""
    model.eval()
    preds, labels, _ = test_epoch(model, dev_dataloader)
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    # 使用布尔掩码，处理空集
    mask = (labels != 0)
    preds_nz = preds[mask]
    labels_nz = labels[mask]
    if preds_nz.size == 0 or labels_nz.size == 0:
        return {'thr': 0.0, 'f1': 0.0, 'acc': 0.0}
    best = {
        'thr': 0.0,
        'f1': -1.0,
        'acc': 0.0
    }
    thr = search_min
    while thr <= search_max + 1e-8:
        by = labels_nz >= 0
        bp = preds_nz >= thr
        f1 = f1_score(by, bp, average='weighted')
        acc = accuracy_score(by, bp)
        if f1 > best['f1']:
            best = {'thr': float(thr), 'f1': float(f1), 'acc': float(acc)}
        thr += step
    return best




def train(
    model,
    optimizer,
    scheduler,
    train_dataloader,
    validation_dataloader,
    test_data_loader,
    results_dir=None
):
    # 存储所有训练指标用于绘图
    train_losses = []
    valid_losses = []
    test_metrics = {
        'acc': [],
        'mae': [],
        'acc7': [],
        'f1': [],
        'corr': []
    }
    
    best_loss = 1000
    best_acc = 0 
    best_mae = 0
    best_corr = 0
    best_f_score = 0
    best_acc_7 = 0
    best_epoch = 0  # 追踪最佳指标出现的epoch

    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, epoch_i)
        valid_loss = eval_epoch(model, validation_dataloader)
        # 默认阈值（0）下的测试指标
        test_acc, test_mae, test_corr, test_f_score, test_acc7 = test_score_model(model, test_data_loader)

        # 在开发集上做阈值校准，并用最佳阈值重新计算测试集分类指标
        thr_info = calibrate_threshold_on_dev(model, validation_dataloader)
        cal_thr = thr_info.get('thr', 0.0)
        # 复用 test_epoch 结果，避免重复前向
        preds_raw, y_test, _ = test_epoch(model, test_data_loader)
        preds_raw = np.asarray(preds_raw)
        y_test = np.asarray(y_test)
        if preds_raw.size == 0 or y_test.size == 0:
            cal_f1, cal_acc = 0.0, 0.0
        else:
            mask = (y_test != 0)
            if mask.sum() == 0:
                mask = np.ones_like(y_test, dtype=bool)
            by = (y_test[mask] >= 0)
            bp = (preds_raw[mask] >= cal_thr)
            try:
                cal_f1 = float(f1_score(by, bp, average='weighted'))
            except Exception:
                cal_f1 = 0.0
            try:
                cal_acc = float(accuracy_score(by, bp))
            except Exception:
                cal_acc = 0.0

        # 记录指标
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        test_metrics['acc'].append(test_acc)
        test_metrics['mae'].append(test_mae)
        test_metrics['acc7'].append(test_acc7)
        test_metrics['f1'].append(test_f_score)
        test_metrics['corr'].append(test_corr)

        print(
            "epoch:{}, train_loss:{:.4f}, valid_loss:{:.4f}, test_acc:{:.4f}".format(
                epoch_i, train_loss, valid_loss, test_acc
            )
        )

        print(
            "current mae:{:.4f}, acc@0:{:.4f}, acc7:{:.4f}, f1@0:{:.4f}, corr:{:.4f}, cal_thr:{:.2f}, acc@cal:{:.4f}, f1@cal:{:.4f}".format(
                test_mae, test_acc, test_acc7, test_f_score, test_corr, cal_thr, cal_acc, cal_f1
            )
        )

        if valid_loss < best_loss: 
            best_loss = valid_loss
            best_acc = test_acc 
            best_mae = test_mae
            best_corr = test_corr
            best_f_score = test_f_score
            best_acc_7 = test_acc7
            
            # 记录最佳指标出现的epoch
            best_epoch = epoch_i
            print(f"*** New best performance at epoch {best_epoch}! ***")
            
            # Save best predictions if we have a results directory
            if results_dir:
                test_score_model(model, test_data_loader, output_dir=results_dir)
                
        print(
            "best mae:{:.4f}, acc:{:.4f}, acc7:{:.4f}, f1:{:.4f}, corr:{:.4f} (epoch {})".format(
                best_mae, best_acc, best_acc_7, best_f_score, best_corr, best_epoch
            )
        )
        
        # 更新学习率调度器
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if epoch_i % 10 == 0:
            print(f"[INFO] Current learning rate: {current_lr:.2e}")
        
        # 每N个epoch或最后一个epoch绘制训练进度图和保存结果摘要
        should_save = ((epoch_i + 1) % args.save_interval == 0 or epoch_i == int(args.n_epochs) - 1)
        
        if results_dir and should_save:
            # 绘制训练进度图
            plot_training_progress(train_losses, valid_losses, test_metrics, results_dir, epoch_i + 1)
            
            # 保存中间结果摘要
            intermediate_results = {
                "current_epoch": epoch_i + 1,
                "best_loss": best_loss,
                "best_acc": best_acc,
                "best_mae": best_mae,
                "best_corr": best_corr,
                "best_f1_score": best_f_score,
                "best_acc_7": best_acc_7,
                "best_epoch": best_epoch,
                "calibrated_threshold": cal_thr,
                "acc_at_calibrated": cal_acc,
                "f1_at_calibrated": cal_f1,
                "train_losses": train_losses,
                "valid_losses": valid_losses,
                "test_metrics": test_metrics
            }
            
            with open(os.path.join(results_dir, "results_summary.json"), "w") as f:
                json.dump(intermediate_results, f, indent=4)
            
            print(f"[INFO] Saved intermediate results and plot at epoch {epoch_i + 1}")
    
    # Save final results summary (最终版本，包含完整信息)
    if results_dir:
        final_results = {
            "training_completed": True,
            "total_epochs": len(train_losses),
            "best_loss": best_loss,
            "best_acc": best_acc,
            "best_mae": best_mae,
            "best_corr": best_corr,
            "best_f1_score": best_f_score,
            "best_acc_7": best_acc_7,
            "best_epoch": best_epoch,
            # 注意：以下三个字段反映最后一个 epoch 的校准结果
            "calibrated_threshold": cal_thr,
            "acc_at_calibrated": cal_acc,
            "f1_at_calibrated": cal_f1,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "test_metrics": test_metrics
        }
        
        with open(os.path.join(results_dir, "results_summary.json"), "w") as f:
            json.dump(final_results, f, indent=4)
            
        # 绘制最终的训练进度图
        plot_training_progress(train_losses, valid_losses, test_metrics, results_dir, len(train_losses))
        print(f"[INFO] Final results and plot saved")

def setup_results_dir():
    """Create directory for saving experiment results"""
    if not args.save_results:
        print("Note: Results are not being saved because --save_results flag was not used.")
        return None
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, args.output_dir)
    
    # Create main results directory if it doesn't exist
    print(f"Creating results directory at: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create experiment-specific directory with timestamp
    exp_dir = os.path.join(results_dir, f"{args.dataset}_{timestamp}")
    print(f"Creating experiment directory at: {exp_dir}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Verify directories were created
    if not os.path.exists(results_dir):
        print(f"WARNING: Failed to create results directory at {results_dir}")
    if not os.path.exists(exp_dir):
        print(f"WARNING: Failed to create experiment directory at {exp_dir}")
    
    # Set up log file for terminal output
    log_file = os.path.join(exp_dir, "terminal_output.txt")
    try:
        sys.stdout = TeeLogger(log_file)
        print(f"Terminal output is now being logged to {log_file}")
    except Exception as e:
        print(f"Error setting up log file: {e}")
    
    # Save hyperparameters
    save_hyperparameters(exp_dir)
    
    return exp_dir

def save_hyperparameters(output_dir):
    """Save hyperparameters to a JSON file"""
    hyperparams = vars(args)
    with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=4)
    
    # Print hyperparameters to terminal/log
    print("Experiment hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    print("\n" + "="*50 + "\n")

def main():
    set_random_seed(args.seed)

    if args.model is None:
        args.model = get_default_model_for_dataset(args.dataset)
    print(f"Using model: {args.model}")
    
    # Set up results directory for this experiment
    results_dir = setup_results_dir()
    
    # Force save results if not explicitly set via command line
    if not args.save_results:
        print("Automatically enabling result saving...")
        args.save_results = True
        results_dir = setup_results_dir()
    
    print(f"Starting experiment with {args.dataset} dataset")
    if results_dir:
        print(f"Results will be saved to {results_dir}")
    else:
        print("WARNING: Results directory was not created successfully!")

    # tokenizer = get_tokenizer(args.model)

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    # 解析 sweep 列表
    sweep_values = []
    if isinstance(args.mi_sweep, str) and len(args.mi_sweep.strip()) > 0:
        try:
            sweep_values = [float(x) for x in args.mi_sweep.split(',') if x.strip() != '']
        except Exception as _e:
            print(f"[WARNING] Failed to parse --mi_sweep: {args.mi_sweep}, err={_e}")
            sweep_values = []

    if not sweep_values:
        # 单次运行
        model, optimizer, scheduler = prep_for_training(num_train_optimization_steps, mi_weight=args.mi_weight)
        train(
            model,
            optimizer,
            scheduler,
            train_data_loader,
            dev_data_loader,
            test_data_loader,
            results_dir
        )
    else:
        # 多次 sweep，对每个权重建立子目录
        base_dir = results_dir or os.getcwd()
        for w in sweep_values:
            sub_dir = os.path.join(base_dir, f"mi_{w}")
            try:
                os.makedirs(sub_dir, exist_ok=True)
            except Exception:
                pass
            print(f"\n===== Running MI sweep: weight={w} =====")
            # 保存该子实验的超参
            try:
                hyper = vars(args).copy()
                hyper['mi_weight'] = w
                with open(os.path.join(sub_dir, "hyperparameters.json"), "w") as f:
                    import json as _json
                    _json.dump(hyper, f, indent=4)
            except Exception:
                pass

            model, optimizer, scheduler = prep_for_training(num_train_optimization_steps, mi_weight=w)
            train(
                model,
                optimizer,
                scheduler,
                train_data_loader,
                dev_data_loader,
                test_data_loader,
                sub_dir
            )
    
    if results_dir:
        print(f"\nExperiment complete. Results saved to {results_dir}")
        # Restore original stdout if needed
        if isinstance(sys.stdout, TeeLogger):
            sys.stdout.close()
            sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main()
