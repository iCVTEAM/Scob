import os,shutil
from typing import Optional,List
from datetime import datetime

from . import iddp as dist
from . import logging
_logger=logging.get_logger(__name__)
import pathlib


DIR_SHARE_STATE = "state"
DIR_CHECKPOINTS = "checkpoints"
DIR_DATASET = "data"
DIR_TENSORBOARD = "boards"

class GlobalFileHelper:
    def __init__(
            self, disk_root: str,
    ):
        self.disk_root = disk_root

        if dist.is_main_process():
            if not os.path.exists(self.disk_root):
                os.mkdir(self.disk_root)
            if not os.path.exists(os.path.join(self.disk_root, DIR_CHECKPOINTS)):
                os.mkdir(os.path.join(self.disk_root, DIR_CHECKPOINTS))
            if not os.path.exists(os.path.join(self.disk_root, DIR_TENSORBOARD)):
                os.mkdir(os.path.join(self.disk_root, DIR_TENSORBOARD))
            if not os.path.exists(os.path.join(self.disk_root, DIR_DATASET)):
                os.mkdir(os.path.join(self.disk_root, DIR_DATASET))
            if not os.path.exists(os.path.join(self.disk_root, DIR_SHARE_STATE)):
                os.mkdir(os.path.join(self.disk_root, DIR_SHARE_STATE))

        if dist.is_enabled():
            _logger.debug("found initialized ddp session")
            dist.barrier()
            _logger.debug("waited for filehelper distributed initialization")
        
        self._instance_files:List[InstanceFileHelper]=[]
    
    def get_dataset_slot(self, dataset_id: str) -> str:
        return os.path.join(self.disk_root, DIR_DATASET, dataset_id)
    
    def get_state_slot(self,*name:str)->str:
        path=os.path.join(self.disk_root,DIR_SHARE_STATE,*name)
        parent=pathlib.Path(path).parent
        if not parent.exists():
            os.makedirs(parent)
        return path
    
    def get_exp_instance(self,comment:str):
        t=InstanceFileHelper(self.disk_root,comment,self)
        self._instance_files.append(t)
        return t
    
    def clear_all_instance(self):
        for x in self._instance_files:
            x.clear_instance()

class InstanceFileHelper:
    def __init__(
            self, disk_root: str, comment: str, parent:GlobalFileHelper
    ):
        self.disk_root = disk_root
        self.comment = comment
        self._parent=parent

        if dist.is_main_process():
            if not os.path.exists(os.path.join(self.disk_root, DIR_CHECKPOINTS)):
                os.mkdir(os.path.join(self.disk_root, DIR_CHECKPOINTS))
            if not os.path.exists(os.path.join(self.disk_root, DIR_TENSORBOARD)):
                os.mkdir(os.path.join(self.disk_root, DIR_TENSORBOARD))
            if not os.path.exists(os.path.join(self.disk_root, DIR_DATASET)):
                os.mkdir(os.path.join(self.disk_root, DIR_DATASET))
            if not os.path.exists(os.path.join(self.disk_root, DIR_SHARE_STATE)):
                os.mkdir(os.path.join(self.disk_root, DIR_SHARE_STATE))

        if dist.is_enabled():
            _logger.debug("found initialized ddp session")
            dist.barrier()
            _logger.debug("waited for filehelper distributed initialization")

        self.ckpt_dir = os.path.join(self.disk_root, DIR_CHECKPOINTS, comment)
        # self.board_dir = os.path.join(
        #     self.disk_root,
        #     DIR_TENSORBOARD,
        #     f"{self.comment}-{datetime.now().strftime('%Y.%m.%d.%H.%M.%S')}",
        # )
    
    def clear_instance(self):
        paths=[
            self.ckpt_dir,
            self.default_board_dir,
        ]
        for path in paths:
            if os.path.exists(path):
                shutil.rmtree(path)

    def prepare_checkpoint_dir(self):
        if not os.path.exists(self.ckpt_dir):
            if dist.is_main_process():
                os.mkdir(self.ckpt_dir)
        if dist.is_enabled():
            dist.barrier()

    def find_latest_checkpoint(self) -> Optional[str]:
        """
        find the latest checkpoint file at checkpoint dir.
        """

        if not os.path.exists(self.ckpt_dir):
            return None

        checkpoint_files: List[str] = []
        for (dirpath, dirnames, filenames) in os.walk(self.ckpt_dir):
            checkpoint_files.extend(filenames)

        latest_checkpoint_path = None
        latest_checkpoint_epoch = -1

        for file in checkpoint_files:
            if file.find("best") != -1:
                continue
            if file.find("init") != -1:
                continue
            if file.find("checkpoint") == -1:
                continue
            epoch = int(file.split(".")[0].split("-")[1])
            if epoch > latest_checkpoint_epoch:
                latest_checkpoint_epoch = epoch
                latest_checkpoint_path = file

        if latest_checkpoint_path:
            return os.path.join(self.ckpt_dir, latest_checkpoint_path)

    def get_initparams_slot(self) -> str:
        if not dist.is_main_process():
            _logger.warning("[DDP] try to get checkpoint slot on follower")
        _logger.info("created initialization slot")
        filename = f"init.pth"
        return os.path.join(self.ckpt_dir, filename)

    def get_checkpoint_slot(self, cur_epoch: int) -> str:
        if not dist.is_main_process():
            _logger.warning("[DDP] try to get checkpoint slot on follower")
        filename = f"checkpoint-{cur_epoch}.pth"
        return os.path.join(self.ckpt_dir, filename)
    

    def get_best_checkpoint_slot(self) -> str:
        if not dist.is_main_process():
            _logger.warning("[DDP] try to get checkpoint slot on follower")
        return os.path.join(self.ckpt_dir, "best.pth")
    
    @property
    def default_board_dir(self) -> str:
        return os.path.join(
            self.disk_root,DIR_TENSORBOARD,self.comment,
        )
    
    def get_dataset_slot(self, dataset_id: str) -> str:
        return self._parent.get_dataset_slot(dataset_id)
    
    def get_state_slot(self,*name:str)->str:
        return self._parent.get_state_slot(*name)