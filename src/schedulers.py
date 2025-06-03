import math
from abc import ABC, abstractmethod
from typing import List, Optional

from torch.optim import Optimizer, lr_scheduler


class SchedulerBase(lr_scheduler.LRScheduler, ABC):
    def __init__(self, optimizer: Optimizer, total_iters: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.optimizer = optimizer
        self.total_iters = total_iters
        self.eta_min = eta_min
        self._step_count = 0
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    @abstractmethod
    def get_lr(self) -> List[float]:
        pass


class SchedulerChain(SchedulerBase):

    def __init__(self, *schedulers: SchedulerBase):
        self.schedulers = list(schedulers)
        self.optimizer = self._sanity_check()

        iters = [s.total_iters for s in self.schedulers]
        self.next_iters = [sum(iters[:i + 1]) for i in range(len(iters) - 1)]
        self.total_iters = sum(iters)
        self.i_scheduler = 0

        super().__init__(optimizer=self.optimizer, total_iters=self.total_iters)

    def _sanity_check(self) -> Optimizer:
        """
        Ensure all schedulers use the same optimizer.
        """
        optimizers = [id(s.optimizer) for s in self.schedulers]
        assert len(set(optimizers)) == 1, (
            f"Expected one optimizer, but got {len(set(optimizers))}."
        )
        return self.schedulers[0].optimizer

    def step(self, epoch: Optional[int] = None) -> None:
        """
        Step the current scheduler.
        """
        if self._step_count in self.next_iters:
            self.i_scheduler += 1
        self.schedulers[self.i_scheduler].step(epoch)
        self._step_count += 1

    def get_lr(self) -> List[float]:
        return self.schedulers[self.i_scheduler].get_lr()


class WarmUpScheduler(SchedulerBase):
    def get_lr(self) -> List[float]:
        return [
            self.eta_min + (base_lr - self.eta_min) * self._step_count / self.total_iters
            for base_lr in self.base_lrs
        ]


class CosineAnnealingScheduler(SchedulerBase):
    def get_lr(self) -> List[float]:
        lrs = []
        for base_lr in self.base_lrs:
            rad = math.pi * self._step_count / self.total_iters
            factor = 0.5 * (math.cos(rad) + 1)
            lrs.append(self.eta_min + (base_lr - self.eta_min) * factor)
        return lrs
