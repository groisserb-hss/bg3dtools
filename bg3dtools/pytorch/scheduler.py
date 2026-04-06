"""
Learning rate schedulers for PyTorch optimization.

This module provides custom learning rate scheduling strategies including
warmup-decay, triangular cycles, and step decay schedules.

Classes
-------
ScheduledOptim
    Abstract base class for scheduled optimizers.
WarmDecay
    Warmup followed by polynomial decay schedule.
Triangle
    Triangular cyclic learning rate with decay.
Step
    Step decay schedule.
"""

from abc import ABC, abstractmethod


from torch.optim.lr_scheduler import CyclicLR, StepLR, ReduceLROnPlateau


def get_scheduler(optimizer, args, num_samples):
    if args.scheduler == 'triangular2':
        batches_per_epoch = num_samples // args.batch_size
        step_size = args.num_epochs * batches_per_epoch // 6
        scheduler = CyclicLR(optimizer, mode='triangular2', base_lr=0.00001, max_lr=args.learning_rate, step_size_up=step_size, cycle_momentum=False)
    elif args.scheduler == 'decay':
        gamma = 0.01 ** (1 / args.num_epochs)
        scheduler = StepLR(optimizer, gamma=gamma, step_size=1)
    elif args.scheduler == 'plateau':
        step_size = args.num_epochs // 5
        scheduler = ReduceLROnPlateau(optimizer, patience=step_size, factor=0.5, threshold=0.005, threshold_mode='rel', min_lr=args.learning_rate/100)
    else:
        raise ValueError('Invalid scheduler')

    return scheduler


class ScheduledOptim(ABC):
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.n_steps = 1
        self.curr_lr = -1

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    @abstractmethod
    def _get_lr_scale(self):
        return 1.0

    def get_last_lr(self):
        return self.curr_lr

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

        self.curr_lr = lr


class WarmDecay(ScheduledOptim):
    """
    modified version of scheduler from Attention is All You Need
    uses a steeper decay rate and scales to 1 at peak
    """
    def __init__(self, optimizer, lr_mul, n_warmup_steps, decay=-1.0):
        ScheduledOptim.__init__(self, optimizer, lr_mul)
        self.n_warmup_steps = n_warmup_steps
        self.scale = 1 / (n_warmup_steps**(decay+1))
        self.decay = decay

    def _get_lr_scale(self):
        step, warmups = self.n_steps, self.n_warmup_steps
        scale, decay = self.scale, self.decay
        return scale * warmups * min(step**decay, step * warmups ** (decay-1))


class Triangle(ScheduledOptim):
    def __init__(self, optimizer, lr_mul, interval, decay=0.5):
        ScheduledOptim.__init__(self, optimizer, lr_mul)
        self.interval = interval
        self.decay = decay

    def _get_lr_scale(self):
        step, interval, decay = self.n_steps, self.interval, self.decay

        subcycle = step // interval
        cycle = (step // (interval*2))
        scale = decay**cycle
        substep = step % interval

        if subcycle % 2 == 0:
            # rising edge
            return scale * substep / interval
        else:
            # falling edge
            return scale * (1 - substep / interval)

class Step(ScheduledOptim):
    def __init__(self, optimizer, lr_mul, interval, decay=0.9):
        ScheduledOptim.__init__(self, optimizer, lr_mul)
        self.interval = interval
        self.decay = decay

    def _get_lr_scale(self):
        step, interval, decay = self.n_steps, self.interval, self.decay

        cycle = step // interval
        return decay ** cycle
