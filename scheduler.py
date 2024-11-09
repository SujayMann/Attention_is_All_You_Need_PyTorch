import torch
from typing import List

class CustomScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Custom learning rate scheduler based on the paper 'Attention is All You Need'

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to apply scheduler to.
        d_model (int): Dimension of the model.
        warmup_steps (int): Number of steps to perform linear learning rate warmup (default 2000). """
    
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int=2000) -> None:
        self._step_num = 1
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Compute learning rate.
        
        Returns:
            List[float]: Learning rate as a list."""
        
        arg1 = torch.tensor(self._step_num) ** -0.5
        arg2 = self._step_num * (self.warmup_steps ** -1.5)
        lr = (self.d_model ** -0.5) * min(arg1, arg2)
        self._step_num += 1
        return [lr]
