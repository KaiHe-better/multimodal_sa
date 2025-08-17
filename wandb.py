# Minimal stub to avoid importing heavy wandb package in environments where it's broken or unavailable.
# Provides no-op interfaces commonly touched by trackers.

__all__ = [
    "init", "finish", "log", "run", "sdk", "wandb_sdk",
]

class _Run:
    def __init__(self, *args, **kwargs):
        pass
    def log(self, *_args, **_kwargs):
        pass
    def finish(self):
        pass

run = _Run()

def init(*args, **kwargs):
    return run

def log(*_args, **_kwargs):
    pass

def finish():
    pass

# Minimal nested sdk alias to satisfy "from wandb import sdk as wandb_sdk" patterns if any
class sdk:  # type: ignore
    pass

wandb_sdk = sdk
