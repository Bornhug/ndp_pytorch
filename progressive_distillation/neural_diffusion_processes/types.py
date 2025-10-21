from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, Union

import torch
import numpy as np

# PyTorch tensor or NumPy array
TensorOrArray = Union[torch.Tensor, np.ndarray]
Dtype = Any
Rng = torch.Generator  # torch.Generator instead of jax.random.KeyArray
Params = Any           # PyTorch doesn't have a direct optax-like Params type
Config = Any


@dataclass
class Batch:
    x_target: TensorOrArray
    y_target: TensorOrArray
    x_context: TensorOrArray | None = None
    y_context: TensorOrArray | None = None
    mask_target: TensorOrArray | None = None
    mask_context: TensorOrArray | None = None


Dataset = Generator[Batch, None, None]
