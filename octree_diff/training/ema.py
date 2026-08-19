import copy

import torch


class EMA:
    """Exponential moving average of model weights.

    Diffusion samples are noticeably cleaner when drawn from averaged weights
    than from the raw optimisation trajectory, so both the training scripts keep
    an EMA copy and save that as the primary checkpoint.

    The decay warms up as 1 - 1/(step + 1) for the first steps so the average is
    not dominated by the (essentially random) initial weights.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.step = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, 1.0 - 1.0 / (self.step + 1))
        for s, p in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(p.detach(), alpha=1.0 - d)
            else:
                s.copy_(p)

    def state_dict(self):
        return self.shadow.state_dict()


def lr_at(step, base_lr, total_steps, warmup_steps):
    """Linear warmup then cosine decay to 5% of the base rate."""
    import math

    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))
