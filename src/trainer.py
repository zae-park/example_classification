import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, Dict, Union, Optional, Sequence

import tqdm
import numpy as np
import torch
from torch import optim
from torch.utils import data as td


class Trainer(ABC):
    """
    Abstract class for experiments with 2 abstract methods train_step and test_step.
    Note that both the train_step and test_step receive batch and return a dictionary.
    More detail about those are in each method's docstring.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained or tested.
    device : Union[torch.device, Sequence[torch.device]]
        The device(s) to run the model on (e.g., 'cuda' or ['cuda:0', 'cuda:1']).
    mode : str
        The mode of the trainer, either 'train' or 'test'.
    optimizer : torch.optim.Optimizer
        The optimizer for training the model.
    scheduler : Optional[Union[torch.optim.lr_scheduler._LRScheduler, core.SchedulerBase]]
        The learning rate scheduler for the optimizer.
    log_bar : bool, optional
        Whether to display a progress bar during training/testing, by default True.
    scheduler_step_on_batch : bool, optional
        Whether to update the scheduler on each batch, by default False.
    gradient_clip : float, optional
        Gradient clipping value. If 0, no gradient clipping is applied, by default 0.0.

    Notes
    -----
    - The `metric_on_epoch_end` method is provided for calculating and logging custom metrics
      at the end of each epoch. It should be overridden by subclasses if custom metrics are needed.
    - The method `metric_on_epoch_end` is expected to return a dictionary where keys are metric
      names and values are the corresponding computed values. These metrics will be displayed in
      the progress bar if `log_bar` is enabled.
    """

    def __init__(
        self,
        model,
        device: Union[torch.device, Sequence[torch.device]],
        mode: str,
        optimizer: optim.Optimizer,
        scheduler: Optional,
        *,
        log_bar: bool = True,
        scheduler_step_on_batch: bool = False,
        gradient_clip: float = 0.0,
    ):
        self.primary_device_index = 0  # Default index is 0, will be adjusted in multi-GPU cases
        # Init with given args (positional params)
        self._set_device(device)
        self.model = self._to_device(model)
        self.mode = mode
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Init with given args (named params)
        self.log_bar = log_bar
        self.scheduler_step_on_batch = scheduler_step_on_batch
        self.gradient_clip = gradient_clip

        self.progress_checker = ProgressChecker()

        # Init vars
        self.log_train, self.log_test = defaultdict(list), defaultdict(list)
        self.train_metrics, self.test_metrics = {}, {}
        self.loss_history_train, self.loss_history_test = [], []
        self.loss_buffer, self.weight_buffer = torch.inf, defaultdict(list)
        self.loader, self.n_data, self.batch_size = None, None, None
        self.valid_loader, self.n_valid_data, self.valid_batch_size = None, None, None

    def _to_cpu(self, *args) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        if len(args) == 1:
            a = args[0]
            if isinstance(a, torch.Tensor):
                return a.detach().cpu() if "detach" in a.__dir__() else a.item() if "item" in a.__dir__() else a
            else:
                return a
        else:
            return tuple([self._to_cpu(a) for a in args])

    def _set_device(self, device: Union[torch.device, Sequence[torch.device]]):
        if isinstance(device, (list, tuple)):
            # multi GPU
            self.device = device
        else:
            # single GPU
            self.device = [device]
            if "cuda" in device.type:
                torch.cuda.set_device(device)

    def _to_device(
        self, *args, **kwargs
    ) -> Union[Tuple[Union[torch.Tensor, torch.nn.Module]], Union[torch.Tensor, torch.nn.Module]]:
        device = self.device[self.primary_device_index]  # Use the primary device index to select the correct device

        if args:
            if len(args) == 1:
                return args[0].to(device) if "to" in args[0].__dir__() else args[0]
            else:
                return tuple([a.to(device) if "to" in a.__dir__() else a for a in args])
        elif kwargs:
            return {k: v.to(device) if "to" in v.__dir__() else v for k, v in kwargs.items()}

    def _data_count(self, initial=False) -> None:
        if initial:
            self.n_data = self.loader.dataset.__len__() if self.loader is not None else 0
            self.n_valid_data = self.valid_loader.dataset.__len__() if self.valid_loader is not None else 0
        else:
            if self.mode == "test" and self.valid_loader is not None:
                self.n_valid_data -= self.valid_batch_size
            else:
                self.n_data -= self.batch_size

    def _check_batch_size(self):
        self.batch_size = self.loader.batch_size if self.loader is not None else 0
        self.valid_batch_size = self.valid_loader.batch_size if self.valid_loader is not None else 0

    def _scheduler_step_check(self, epoch: int) -> None:
        if "total_iters" in self.scheduler.__dict__ and self.scheduler_step_on_batch:
            need_steps = epoch * len(self.loader)
            assert self.scheduler.total_iters >= need_steps, (
                f'The "total_iters" {self.scheduler.total_iters} for the given scheduler is insufficient.'
                f"It must be at least more than the total iterations {need_steps} required during training."
            )

    def run(self, n_epoch: int, loader: td.DataLoader, valid_loader: Optional[td.DataLoader] = None, **kwargs) -> None:
        self.loader, self.valid_loader = loader, valid_loader
        self._check_batch_size()
        self._scheduler_step_check(n_epoch)
        pre_epoch = self.progress_checker.get_epoch() - 1
        progress = range(pre_epoch, total_epoch := (pre_epoch + n_epoch))
        progress = tqdm.tqdm(
            progress, position=0, dynamic_ncols=True, file=sys.stderr, leave=False, disable=not self.log_bar
        )

        for e in progress:
            printer = progress.set_description if self.log_bar else print
            printer(f"Epoch {e + 1}/{total_epoch}")

            self._data_count(initial=True)  # Initial data counts
            self.run_epoch(loader, **kwargs)  # Execute training epoch & validation epoch (if provided)

            # Run validation epoch if provided
            if valid_loader:
                self.toggle("test")
                self.run_epoch(valid_loader, **kwargs)
                self.toggle("train")

            # Log metrics at the end of the epoch
            train_summary = ", ".join([f"train_{k}: {v:.4f}" for k, v in self.train_metrics.items()])
            test_summary = ", ".join([f"test_{k}: {v:.4f}" for k, v in self.test_metrics.items()])
            epoch_summary = f"Epoch {e + 1}/{total_epoch} | {train_summary} | {test_summary}"
            progress.write(epoch_summary, file=sys.stderr)

            # Update training state (loss, scheduler, epoch)
            if self.mode == "train":
                cur_loss = np.mean(self.log_test["loss"] if valid_loader else self.log_train["loss"]).item()
                self.check_better(cur_epoch=e, cur_loss=cur_loss)
                if not self.scheduler_step_on_batch:
                    self.scheduler.step(**kwargs)
                self.progress_checker.update_epoch()

    def run_epoch(self, loader: td.DataLoader, **kwargs) -> None:
        self.log_reset()
        batch_progress = tqdm.tqdm(total=len(loader), position=1, leave=False, dynamic_ncols=True, file=sys.stderr)

        for i, batch in enumerate(loader):
            current_batch = batch

            if current_batch is not None:
                self.run_batch(current_batch, **kwargs)
                self._data_count(True if self.batch_size is None else False)

            # Ensure GPU operations are complete before updating progress
            # torch.cuda.synchronize()
            desc = self.print_log(cur_batch=i + 1, num_batch=len(loader))
            batch_progress.update(1)
            if self.log_bar:
                batch_progress.set_postfix(desc)
            else:
                print(desc)

        batch_progress.close()

        # Update metrics at the end of the epoch
        target_metrics = self.train_metrics if self.mode == "train" else self.test_metrics
        target_metrics.update(target_metrics)

    def run_batch(self, batch: Union[tuple, dict], **kwargs) -> None:
        batch = self._to_device(**batch) if isinstance(batch, dict) else self._to_device(*batch)
        if self.mode == "train":
            self.model.train()
            self.optimizer.zero_grad()
            step_dict = self.train_step(batch)
            step_dict["loss"].backward()
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            if self.scheduler_step_on_batch:
                self.scheduler.step(**kwargs)

        elif self.mode == "test":
            self.model.eval()
            with torch.no_grad():
                step_dict = self.test_step(batch)
        else:
            raise ValueError(f"Unexpected mode {self.mode}.")

        # torch.cuda.synchronize()  # Ensure GPU operations are complete
        self.logging(step_dict)
        self.progress_checker.update_step()

    @abstractmethod
    def train_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("train_step must be implemented by subclasses")

    @abstractmethod
    def test_step(self, batch: Union[tuple, dict]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("test_step must be implemented by subclasses")

    def logging(self, step_dict: Dict[str, torch.Tensor]) -> None:
        """
        Log the results of a step.

        Parameters
        ----------
        step_dict : Dict[str, torch.Tensor]
            A dictionary containing the results of a training or testing step.
        """
        for k, v in step_dict.items():
            v_ = self._to_cpu(v)
            if self.mode == "train":
                self.log_train[k].append(v_)
                self.loss_history_train.append(v_)
            elif self.mode == "test":
                self.log_test[k].append(v_)
                self.loss_history_test.append(v_)
            else:
                raise ValueError(f"Unexpected mode {self.mode}.")

    def toggle(self, mode: str = None) -> None:
        if mode:
            self.mode = mode
        else:
            self.mode = "test" if self.mode == "train" else "train"

    def check_better(self, cur_epoch: int, cur_loss: float) -> bool:
        if cur_loss >= self.loss_buffer:
            return False
        self.weight_buffer["epoch"].append(cur_epoch)
        self.weight_buffer["weight"].append(self.model.state_dict())
        self.loss_buffer = cur_loss
        return True

    def log_reset(self, epoch_end: bool = False) -> None:
        """
        Clear the log.
        """
        self.log_train.clear()
        self.log_test.clear()
        if epoch_end:
            self.train_metrics.clear()
            self.test_metrics.clear()

    def get_loss_history(self, mode: str = "train") -> list:
        if mode == "train":
            return self.loss_history_train
        elif mode == "test":
            return self.loss_history_test
        else:
            raise ValueError(f"Mode must be 'train' or 'test', got '{mode}'.")

    def print_log(self, cur_batch: int, num_batch: int) -> dict:
        log = self.log_train if self.mode == "train" else self.log_test
        LR = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0

        # Base log dictionary
        log_dict = {"Batch": f"{cur_batch}/{num_batch}", "LR": f"{LR:.3e}"}
        for k, v in log.items():
            if "output" not in k:  # Include only numerical entries
                log_dict[k] = f"{np.mean(v):.6f}"

        return log_dict

    def save_model(self, filename: str) -> None:
        torch.save(self.model.state_dict(), filename)

    def apply_weights(self, filename: str = None, strict: bool = True) -> None:
        try:
            if filename:
                weights = torch.load(filename, map_location=torch.device("cpu"))
            else:
                weights = self.weight_buffer["weight"][-1]
            self.model.load_state_dict(weights, strict=strict)
        except FileNotFoundError as e:
            print(f"Cannot find file {filename}", e)
        except IndexError as e:
            print(f"There is no weight in buffer of trainer.", e)

    def inference(self, loader: td.DataLoader) -> list:
        mode_buffer = self.mode
        self.toggle("test")  # Switch to test mode

        self.run_epoch(loader)
        results = self.log_test.get("output", [])
        self.toggle(mode_buffer)

        return results


class ProgressChecker:
    def __init__(self):
        self.__step = 1
        self.__epoch = 1

    def update_step(self):
        self.__step += 1

    def update_epoch(self):
        self.__epoch += 1

    def get_step(self):
        return self.__step

    def get_epoch(self):
        return self.__epoch

    def init_state(self):
        self.__step = 1
        self.__epoch = 1
