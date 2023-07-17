import sys
import inspect
from typing import List,Dict,Any,Optional,Callable,Sequence
from typing_extensions import Self

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm

from .probes import AverageMeter,SmoothMeanMeter
from . import iddp as dist
from .utils import any_to
from . import logging
_logger=logging.get_logger(__name__)

class FunctionalSection:
    class JumpSectionException(Exception):
        pass

    def __init__(self,break_phase:bool,report_cb:Optional[Callable[[Dict[str,Any]],None]]=None) -> None:
        self._break_phase=break_phase
        self._state:Dict[str,Any]={}
        self._report_cb=report_cb

    def set(self,key:str,val:Any=True):
        self._state[key]=val

    def __enter__(self):
        if self._break_phase:
            sys.settrace(lambda *args,**keys: None)
            frame=sys._getframe(1)
            frame.f_trace=self.trace
        return self
    
    def trace(self,frame,event,arg):
        raise self.JumpSectionException()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type,self.JumpSectionException):
            return True
        
        if self._report_cb:
            self._report_cb(self._state)


class PhaseHelper:
    def __init__(
            self,
            phase_name: str,
            dataset: Any,
            dataloader: DataLoader,
            dev:Any,
            dry_run: bool = False,
            custom_probes: List[str] = [],
    ) -> None:
        self._phase_name = phase_name
        self._dry_run = dry_run
        self._dataset = dataset
        self._dataloader = dataloader
        self._dev=dev

        self._custom_probe_name = custom_probes

        self._score = AverageMeter("")
        self._loss_probe = AverageMeter("")
        self._realtime_loss_probe=SmoothMeanMeter(len(self._dataloader))
        self._custom_probes:Dict[str,AverageMeter] = dict(
            [(x, AverageMeter(x)) for x in self._custom_probe_name]
        )

        self._loss_updated = False
        self._score_updated = False

    def get_data_sample(self):
        for data in self._dataloader:
            return data

    def range_data(self):
        batch_len = len(self._dataloader)
        one_percent_len=max(1,(batch_len+25-1)//25)
        if dist.is_main_process():
            if self._dry_run:
                _logger.info("data preview can be found in log")
            with tqdm(total=batch_len,dynamic_ncols=True,ascii=' >=') as progressbar:
                for batchi, data in enumerate(self._dataloader):
                    if self._dry_run:
                        torch.set_printoptions(profile='full')
                        _logger.debug(data)
                        torch.set_printoptions(profile='default')
                    yield batchi, data
                    progressbar.update()
                    postfix={'loss':str(self._realtime_loss_probe)}
                    progressbar.set_postfix(postfix)
                    if batchi%one_percent_len==0:
                        _logger.debug(f"progress {batchi}/{batch_len}: {postfix}")
                    if self._dry_run and batchi>=2:
                        break
        else:
            for batchi, data in enumerate(self._dataloader):
                yield batchi, data
                if self._dry_run and batchi>=2:
                    break
    
    def _check_update(self):
        if not self._score_updated:
            _logger.error(f"no score updated during phase {self._phase_name}")
        if not self._loss_updated:
            _logger.error(f"no loss updated during phase {self._phase_name}")

        for name in self._custom_probe_name:
            _logger.error(f"{name} not updated during phase {self._phase_name}")

    def update_probe(self, name: str, value: float, n: int = 1):
        if name in self._custom_probe_name:
            self._custom_probe_name.remove(name)
        self._custom_probes[name].update(value, n)

    @staticmethod
    def validate_loss(loss: Tensor, panic: bool = True) -> bool:
        hasnan = loss.isnan().any().item()
        hasinf = loss.isinf().any().item()
        hasneg = (loss < 0).any().item()
        if panic:
            assert not hasnan, f"loss function returns invalid value `nan`: {loss}"
            assert not hasinf, f"loss function returns invalid value `inf`: {loss}"
            assert not hasneg, f"loss function returns negative value: {loss}"
        return not hasnan and not hasinf and not hasneg

    @staticmethod
    def validate_tensor(t: Tensor, panic: bool = True, msg: str = "") -> bool:
        hasnan = t.isnan().any().item()
        hasinf = t.isinf().any().item()

        if panic:
            assert not hasnan, f"tensor has invalid value `nan`: {t} ({msg})"
            assert not hasinf, f"tensor has invalid value `inf`: {t} ({msg})"

        return not hasnan and not hasinf

    def update_loss(self, loss: Tensor, n: int = 1):
        self._loss_updated = True
        self.validate_loss(loss)
        self._loss_probe.update(loss.item(), n)
        self._realtime_loss_probe.add(loss.item())

    def end_phase(self, score: float):
        self._score_updated = True
        self._score.update(score)

    @property
    def loss_probe(self):
        return self._loss_probe

    @property
    def score(self):
        return self._score.average()

    @property
    def custom_probes(self):
        return self._custom_probes