"""
Cautious Muon Optimizer for WeNet ASR Training
Reference: https://github.com/KellerJordan/modded-nanogpt
"""

import torch
from torch.optim.optimizer import Optimizer


def newton_schulz_orthogonalize(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to orthogonalize gradient matrix G.
    Approximately computes U @ V.T where U, S, V = G.svd()
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    # Ensure wide matrix for efficiency
    transposed = False
    if X.shape[-2] > X.shape[-1]:
        X = X.mT
        transposed = True
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class CautiousMuon(Optimizer):
    """
    Cautious Muon Optimizer.

    Applies Muon (momentum + Newton-Schulz orthogonalization) to 2D weight
    matrices (Linear layers), and Adam to 1D parameters (bias, LayerNorm).
    Cautious weight decay: only applied when gradient and weight are aligned.

    Args:
        params: model parameters
        lr (float): learning rate for Muon (default: 0.023)
        momentum (float): momentum coefficient (default: 0.95)
        weight_decay (float): weight decay for Muon params (default: 0.1)
        beta1 (float): Adam beta1 for 1D params (default: 0.9)
        beta2 (float): Adam beta2 for 1D params (default: 0.999)
        eps (float): Adam epsilon (default: 1e-8)
        ns_steps (int): Newton-Schulz iterations (default: 5)
        adam_lr_scale (float): LR scale for Adam params (default: 0.1)
    """

    def __init__(self, params, lr=0.023, momentum=0.95, weight_decay=0.1,
                 beta1=0.9, beta2=0.999, eps=1e-8, ns_steps=5,
                 adam_lr_scale=0.1):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            beta1=beta1, beta2=beta2, eps=eps, ns_steps=ns_steps,
            adam_lr_scale=adam_lr_scale
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            ns_steps = group['ns_steps']
            adam_lr = lr * group['adam_lr_scale']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    if p.ndim >= 2:
                        # Muon: momentum buffer
                        state['momentum_buf'] = torch.zeros_like(p, dtype=torch.float32)
                        state['use_muon'] = True
                    else:
                        # Adam: first and second moment
                        state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                        state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)
                        state['use_muon'] = False

                state['step'] += 1

                if state['use_muon']:
                    # ===== Muon update for 2D weight matrices =====
                    buf = state['momentum_buf']
                    grad_f32 = grad.float()

                    # Nesterov momentum
                    buf.mul_(momentum).add_(grad_f32)
                    g = grad_f32.add(buf, alpha=momentum)

                    # Newton-Schulz orthogonalization
                    if g.ndim == 2:
                        g_ortho = newton_schulz_orthogonalize(g, steps=ns_steps)
                    else:
                        # For higher-dim params, reshape to 2D
                        orig_shape = g.shape
                        g_2d = g.view(g.shape[0], -1)
                        g_ortho = newton_schulz_orthogonalize(g_2d, steps=ns_steps)
                        g_ortho = g_ortho.view(orig_shape)

                    # Scale by sqrt(max(rows, cols)) for consistent update RMS
                    if g.ndim >= 2:
                        scale = max(g.shape[-2], g.shape[-1]) ** 0.5
                        g_ortho = g_ortho * (0.2 * scale)

                    # Cautious weight decay: only when grad and weight are aligned
                    if weight_decay != 0:
                        mask = (g_ortho * p.float()).gt(0).float()
                        p.mul_(1.0 - lr * weight_decay * mask)

                    # Apply update
                    p.add_(g_ortho, alpha=-lr)

                else:
                    # ===== Adam update for 1D params (bias, LayerNorm) =====
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    t = state['step']

                    grad_f32 = grad.float()
                    exp_avg.mul_(beta1).add_(grad_f32, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad_f32, grad_f32, value=1 - beta2)

                    bias_corr1 = 1 - beta1 ** t
                    bias_corr2 = 1 - beta2 ** t
                    step_size = adam_lr * (bias_corr2 ** 0.5) / bias_corr1

                    denom = exp_avg_sq.sqrt().add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

                    # Standard weight decay for Adam params
                    if weight_decay != 0:
                        p.mul_(1.0 - adam_lr * weight_decay * 0.01)

        return loss
