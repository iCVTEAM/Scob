import torch.distributed as dist
from torch.distributed import * # type:ignore

_ddp_enabled=False

def enable_ddp():
    global _ddp_enabled
    _ddp_enabled=True

def is_enabled():
    return _ddp_enabled

def is_main_process():
    return not _ddp_enabled or dist.get_rank()==0