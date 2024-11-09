import torch
from typing import List

class CustomScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int=2000) -> None:
        self._step_num = 1
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self) -> List[int]:
        arg1 = torch.tensor(self._step_num) ** -0.5
        arg2 = self._step_num * (self.warmup_steps ** -1.5)
        lr = (self.d_model ** -0.5) * min(arg1, arg2)
        self._step_num += 1
        return [lr]
