# Copyright (c) 2025 ROOT Optimizer Authors
# Licensed under the Apache License, Version 2.0 (the "License");
#
# This code is adapted from:
# https://github.com/huawei-noah/Efficient-Computing/tree/master/Optimization/ROOT

import logging
import math
import torch
from torch.optim.optimizer import Optimizer

# Track which shapes have already been warned about
_warned_shapes = set()


def simple_quantile(tensor: torch.Tensor, q: float) -> torch.Tensor:
    """
    A simple quantile implementation (unoptimized for GPU).
    """
    tensor = tensor.flatten()
    num_elements = tensor.numel()
    if num_elements == 0:
        return torch.tensor(float('nan'), dtype=tensor.dtype, device=tensor.device)
    index = torch.tensor(q * (num_elements - 1), device=tensor.device)

    lower_index = torch.floor(index).long()
    upper_index = torch.ceil(index).long()

    k_lower = lower_index + 1
    k_upper = upper_index + 1

    if k_lower == k_upper:
        return tensor.kthvalue(k_lower).values

    lower_value = tensor.kthvalue(k_lower).values
    upper_value = tensor.kthvalue(k_upper).values

    weight = index - lower_index
    return torch.lerp(lower_value, upper_value, weight)


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
def root5(G, steps):
    assert len(G.shape) == 2
    # Coefficients for ROOT used with 4 shapes used in this toy-train script
    COEFF_MAP_CANONICAL = {
        (2048, 2048): (3.4916, -4.8224, 2.1095),
        (2048, 16384): (3.1943, -3.8221, 1.5322),
        (2048, 3072): (3.4509, -4.5790, 1.9325),
        (2048, 8192): (2.9794, -3.6674, 1.6207)
    }
    rows, cols = G.size(0), G.size(1)

    shape_key = (min(rows, cols), max(rows, cols))
    try:
        a, b, c = COEFF_MAP_CANONICAL[shape_key]
    except KeyError:
        # Fallback to default Newton-Schulz coefficients for unknown shapes
        a, b, c = (3.4445, -4.7750, 2.0315)
        # Only warn once per unique shape
        if shape_key not in _warned_shapes:
            _warned_shapes.add(shape_key)
            logging.warning(f"Shape {shape_key} not found. Using default coefficients.")
    X = G
    is_tall = rows > cols
    if is_tall:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if is_tall:
        X = X.T

    return X


# This code snippet is a modified version adapted from the following GitHub repositories:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
# https://github.com/MoonshotAI/Moonlight/

class ROOT(Optimizer):
    """
    ROOT: Robust Orthogonalized Optimizer for Neural Network Training

    Paper: https://arxiv.org/abs/2511.20626
    Authors: Wei He, Kai Han, Hang Zhou, Hanting Chen, Zhicheng Liu, Xinghao Chen, Yunhe Wang

    Arguments:
        root_params: The parameters to be optimized by ROOT.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        root_steps: The number of ROOT iterations. Unlike static Newton-Schulz steps, ROOT uses
                    adaptive iterations tailored to matrix size and condition.
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `root_params` which are
                      {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        root_params=None,
        momentum=0.95,
        nesterov=True,
        root_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            root_steps=root_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(root_params) if root_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        # Sort parameters into those for which we will use ROOT, and those for which we will not
        if root_params is not None:
            for p in root_params:
                # Use ROOT for every parameter in root_params which is >= 2D and doesn't look like an embedding or head layer
                if p.ndim >= 2:
                    self.state[p]["use_root"] = True
                else:
                    self.state[p]["use_root"] = False
        if adamw_params is not None:
            for p in adamw_params:
                # Do not use ROOT for parameters in adamw_params
                self.state[p]["use_root"] = False

    def adjust_lr_for_root(self, lr, param_shape):
        A, B = param_shape[:2]
        # For ASR tasks, use a moderate adjustment
        # Original ROOT uses 0.2 * sqrt(max(A, B))
        # We use 0.1 to balance between stability and convergence speed
        adjusted_ratio = 0.1 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #          ROOT            #
            ############################

            params = [p for p in group["params"] if self.state[p].get("use_root", False)]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue

                # Save original shape for reshaping later
                original_shape = p.shape

                # For parameters with ndim > 2 (like conv layers), reshape to 2D
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                assert g is not None

                # calc update
                state = self.state[p]
                # Initialize momentum buffer with same shape as gradient
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                ## ROOT
                epsilon = simple_quantile(g.abs().float(), 0.9) + 1e-9
                o = torch.sign(g) * torch.nn.functional.relu(torch.abs(g) - epsilon)
                b = g - o
                u = root5(b, steps=group["root_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_root(lr, original_shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # Reshape u to match original parameter shape
                if original_shape != p.shape:
                    u = u.view(original_shape)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p].get("use_root", True)]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
