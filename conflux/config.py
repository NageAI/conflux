"""
CONFLUX Configuration Module

Manages all hyperparameters and settings for the CONFLUX pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SourceModelConfig:
    """Configuration for a single source model."""
    name: str
    model_id: str
    quantization_bits: int = 2
    trust_remote_code: bool = True
    torch_dtype: str = "auto"


@dataclass
class ConfluxConfig:
    """
    Master configuration for the CONFLUX pipeline.

    Example:
        config = ConfluxConfig(
            primary_model_id="Qwen/Qwen3-8B",
            source_models=[
                SourceModelConfig("llama", "meta-llama/Llama-3.1-8B"),
                SourceModelConfig("mistral", "mistralai/Mistral-7B-v0.3"),
            ],
        )
    """

    # Primary model (W) - receives LoRA adapters
    primary_model_id: str = "Qwen/Qwen3-8B"
    primary_quantization_bits: int = 4

    # Source models (M1, M2, ..., MN) - knowledge donors
    source_models: list[SourceModelConfig] = field(default_factory=lambda: [
        SourceModelConfig("llama", "meta-llama/Llama-3.1-8B", quantization_bits=2),
    ])

    # LoRA rank bounds
    rank_min: int = 4
    rank_max: int = 64
    rank_default: int = 16

    # Quantization
    primary_quant_type: str = "nf4"
    source_quant_type: str = "nf2"
    compute_dtype: str = "bfloat16"

    # Profiling
    profiling_samples: int = 256          # v2: was 2048, mini-batch is sufficient
    cka_kernel: str = "linear"
    transfer_value_threshold: float = 0.1
    skip_early_layers: int = 8            # v2: skip first N layers (universal features)
    projection_mode: str = "procrustes"   # v2: "procrustes" or "learned"

    # Residual-SVD initialization
    svd_samples: int = 1024
    svd_scaling: float = 1.0

    # Training
    lambda_residual: float = 0.3
    lambda_decay: str = "adaptive"        # v2: "adaptive" (loss-aware), "cosine", "linear"
    lambda_adaptive_beta: float = 2.0     # v2: steepness of adaptive decay
    per_layer_alpha: bool = True           # v2: learnable per-layer scaling
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = -1
    num_epochs: int = 3

    # LoRA targets
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Offline mode (extract M hidden states to disk)
    offline_mode: bool = False
    cache_dir: str = "./conflux_cache"

    # Hardware
    device: str = "auto"
    max_memory: Optional[dict] = None

    @property
    def n_sources(self) -> int:
        return len(self.source_models)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def validate(self):
        """Validate configuration consistency."""
        assert self.rank_min > 0, "rank_min must be positive"
        assert self.rank_max >= self.rank_min, "rank_max must be >= rank_min"
        assert len(self.source_models) >= 1, "At least one source model required"
        assert 0.0 <= self.lambda_residual <= 1.0, "lambda_residual must be in [0, 1]"
        assert self.primary_quantization_bits in [2, 3, 4, 8, 16], "Invalid quantization bits"
        for sm in self.source_models:
            assert sm.quantization_bits in [2, 3, 4, 8, 16], f"Invalid bits for {sm.name}"
        return True

    def summary(self) -> str:
        """Print human-readable configuration summary."""
        lines = [
            f"CONFLUX Configuration",
            f"{'='*50}",
            f"Primary model : {self.primary_model_id} ({self.primary_quantization_bits}-bit)",
            f"Source models  : {self.n_sources}",
        ]
        for sm in self.source_models:
            lines.append(f"  - {sm.name}: {sm.model_id} ({sm.quantization_bits}-bit)")
        lines.extend([
            f"Rank range    : [{self.rank_min}, {self.rank_max}]",
            f"Lambda        : {self.lambda_residual} ({self.lambda_decay} decay)",
            f"Profiling     : {self.profiling_samples} samples, {self.cka_kernel} CKA",
            f"SVD init      : {self.svd_samples} samples",
            f"Offline mode  : {self.offline_mode}",
            f"{'='*50}",
        ])
        return "\n".join(lines)
