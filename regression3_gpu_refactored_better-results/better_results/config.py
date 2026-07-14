from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MODELS = ("ndp_cond", "ndp_uncond", "flownp")


@dataclass
class DataConfig:
    input_dim: int = 1
    batch_size: int = 32
    x_min: float = -2.0
    x_max: float = 2.0
    train_num_target_min: int = 32
    train_num_target_max: int = 64
    train_num_context_min: int = 0
    train_num_context_max: int = 32
    lengthscale: float = 0.25
    variance: float = 1.0
    observation_noise_std: float = 0.05
    jitter: float = 1e-6


@dataclass
class DiffusionConfig:
    schedule: str = "cosine"
    beta_start: float = 3e-4
    beta_end: float = 0.5
    timesteps: int = 500


@dataclass
class NetworkConfig:
    n_layers: int = 4
    hidden_dim: int = 64
    num_heads: int = 8


@dataclass
class NDPConfig:
    timestep_sampling: str = "stratified"
    repaint_inner_steps: int = 5
    repaint_context_noise: str = "fresh"
    ddim_eta: float = 0.0


@dataclass
class FlowNPConfig:
    dim_posenc: int = 24
    embedding_depth: int = 4
    num_layers: int = 6
    hidden_dim: int = 100
    num_heads: int = 4
    feedforward_dim: int = 200
    predictor_hidden_dim: int = 270
    dropout: float = 0.0
    time_sampling: str = "per_function"
    time_distribution: str = "stratified"
    output_dim: int = 1
    repo_noise_scale: float = 0.2


@dataclass
class OptimizerConfig:
    peak_lr: float = 1e-3
    warmup_steps: int = 512
    ema_rate: float = 0.995
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999


@dataclass
class TrainingConfig:
    seed: int = 42
    num_epochs: int = 250
    samples_per_epoch: int = 16384
    loss_type: str = "l1"
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    grad_clip_norm: float = 1.0
    log_every: int = 100


@dataclass
class Config:
    backend: str = "ndp_cond"
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ndp: NDPConfig = field(default_factory=NDPConfig)
    flownp: FlowNPConfig = field(default_factory=FlowNPConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def steps_per_epoch(self) -> int:
        return self.training.samples_per_epoch // self.data.batch_size

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.training.num_epochs

    def validate(self) -> None:
        if self.backend not in MODELS:
            raise ValueError(f"Unknown model: {self.backend}")
        if self.data.input_dim != 1:
            raise ValueError("This standalone project supports only one-dimensional inputs")
        if self.data.batch_size <= 0 or self.steps_per_epoch <= 0:
            raise ValueError("Training batch and epoch sizes must be positive")
        if self.data.train_num_context_min < 0:
            raise ValueError("Context counts cannot be negative")
        if self.data.train_num_target_min <= 0:
            raise ValueError("Target counts must be positive")
        if self.diffusion.timesteps <= 0:
            raise ValueError("Diffusion timesteps must be positive")
        if self.network.hidden_dim % self.network.num_heads:
            raise ValueError("NDP hidden_dim must be divisible by num_heads")
        if self.flownp.hidden_dim % self.flownp.num_heads:
            raise ValueError("FlowNP hidden_dim must be divisible by num_heads")
        if self.training.loss_type not in {"l1", "l2", "mse"}:
            raise ValueError("Training loss must be l1, l2, or mse")
        if not 0.0 <= self.optimizer.ema_rate <= 1.0:
            raise ValueError("EMA rate must lie in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def for_model(cls, model: str) -> "Config":
        config = cls(backend=model)
        config.validate()
        return config

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, backend: str | None = None) -> "Config":
        # Historical checkpoints contain many now-irrelevant sections. Only the
        # standalone training/model fields are selected here.
        config = cls(
            backend=backend or str(payload.get("backend", "ndp_cond")),
            data=DataConfig(**{
                key: value for key, value in payload.get("data", {}).items()
                if key in DataConfig.__dataclass_fields__
            }),
            diffusion=DiffusionConfig(**{
                key: value for key, value in payload.get("diffusion", {}).items()
                if key in DiffusionConfig.__dataclass_fields__
            }),
            network=NetworkConfig(**{
                key: value for key, value in payload.get("network", {}).items()
                if key in NetworkConfig.__dataclass_fields__
            }),
            ndp=NDPConfig(**{
                key: value for key, value in payload.get("ndp", {}).items()
                if key in NDPConfig.__dataclass_fields__
            }),
            flownp=FlowNPConfig(**{
                key: value for key, value in payload.get("flownp", {}).items()
                if key in FlowNPConfig.__dataclass_fields__
            }),
            optimizer=OptimizerConfig(**{
                key: value for key, value in payload.get("optimizer", {}).items()
                if key in OptimizerConfig.__dataclass_fields__
            }),
            training=TrainingConfig(**{
                key: value for key, value in payload.get("training", {}).items()
                if key in TrainingConfig.__dataclass_fields__
            }),
        )
        config.validate()
        return config

