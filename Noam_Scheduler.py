# https://github.com/tugstugi/pytorch-saltnet/blob/master/utils/lr_scheduler.py

from torch.optim.lr_scheduler import _LRScheduler

class Noam_Scheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super(Noam_Scheduler, self).__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)

        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** -1.5)
        return [base_lr * scale for base_lr in self.base_lrs]


class Modified_Noam_Scheduler(_LRScheduler):
    def __init__(self, optimizer, base):
        self.base = base
        super(Modified_Noam_Scheduler, self).__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)

        scale = self.base ** 0.5 * (last_epoch + self.base) ** (-0.5)
        return [base_lr * scale for base_lr in self.base_lrs]
