from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


VelocityFunction = Callable[[Tensor, Tensor], Tensor]


def integrate_fixed(
    function: VelocityFunction,
    initial: Tensor,
    *,
    steps: int,
    method: str,
) -> Tensor:
    if steps <= 0:
        raise ValueError("Integration steps must be positive")
    if method not in {"euler", "midpoint", "rk4"}:
        raise ValueError("Flow sampler must be euler, midpoint, or rk4")
    value = initial
    delta = value.new_tensor(1.0 / float(steps))
    time = value.new_zeros(())
    for _ in range(steps):
        if method == "euler":
            value = value + delta * function(value, time)
        elif method == "midpoint":
            first = function(value, time)
            value = value + delta * function(
                value + 0.5 * delta * first, time + 0.5 * delta
            )
        else:
            first = function(value, time)
            second = function(value + 0.5 * delta * first, time + 0.5 * delta)
            third = function(value + 0.5 * delta * second, time + 0.5 * delta)
            fourth = function(value + delta * third, time + delta)
            value = value + (delta / 6.0) * (
                first + 2.0 * second + 2.0 * third + fourth
            )
        time = time + delta
    return value

