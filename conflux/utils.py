"""
CONFLUX Utilities

Shared utility functions across all modules.
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_model_info(model) -> dict:
    """Extract key information from a HuggingFace model."""
    config = model.config
    return {
        "model_type": getattr(config, "model_type", "unknown"),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_layers": getattr(config, "num_hidden_layers", None),
        "num_heads": getattr(config, "num_attention_heads", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "total_params": sum(p.numel() for p in model.parameters()),
    }


def estimate_vram(
    primary_bits: int,
    primary_params_b: float,
    source_bits: int,
    source_params_b: float,
    lora_rank: int = 32,
    num_lora_layers: int = 32,
    hidden_dim: int = 4096,
    batch_size: int = 4,
    seq_length: int = 2048,
) -> dict:
    """Estimate VRAM usage for CONFLUX training.

    Args:
        primary_bits: Quantization bits for W (typically 4)
        primary_params_b: Primary model parameters in billions
        source_bits: Quantization bits for M (typically 2)
        source_params_b: Source model parameters in billions
        lora_rank: Average LoRA rank
        num_lora_layers: Number of active LoRA layers
        hidden_dim: Hidden dimension
        batch_size: Training batch size
        seq_length: Max sequence length

    Returns:
        Dict with VRAM breakdown in GB.
    """
    primary_gb = primary_params_b * 1e9 * primary_bits / 8 / 1e9
    source_gb = source_params_b * 1e9 * source_bits / 8 / 1e9

    lora_params = 2 * lora_rank * hidden_dim * num_lora_layers * len(["q", "k", "v", "o", "gate", "up", "down"])
    lora_gb = lora_params * 2 / 1e9  # FP16

    optimizer_gb = lora_gb * 2  # AdamW states

    activation_gb = batch_size * seq_length * hidden_dim * num_lora_layers * 2 / 1e9

    projection_gb = hidden_dim * hidden_dim * 2 / 1e9  # one projection matrix FP16

    total = primary_gb + source_gb + lora_gb + optimizer_gb + activation_gb + projection_gb
    offline_total = primary_gb + lora_gb + optimizer_gb + activation_gb

    return {
        "primary_model_gb": round(primary_gb, 2),
        "source_model_gb": round(source_gb, 2),
        "lora_adapters_gb": round(lora_gb, 2),
        "optimizer_states_gb": round(optimizer_gb, 2),
        "activations_gb": round(activation_gb, 2),
        "projection_gb": round(projection_gb, 2),
        "total_online_gb": round(total, 2),
        "total_offline_gb": round(offline_total, 2),
    }


def print_vram_estimate(estimate: dict):
    """Pretty-print VRAM estimate."""
    print("\n╔════════════════════════════════════╗")
    print("║   CONFLUX VRAM Estimate            ║")
    print("╠════════════════════════════════════╣")
    print(f"║ Primary model:  {estimate['primary_model_gb']:>6.2f} GB          ║")
    print(f"║ Source model:   {estimate['source_model_gb']:>6.2f} GB          ║")
    print(f"║ LoRA adapters:  {estimate['lora_adapters_gb']:>6.2f} GB          ║")
    print(f"║ Optimizer:      {estimate['optimizer_states_gb']:>6.2f} GB          ║")
    print(f"║ Activations:    {estimate['activations_gb']:>6.2f} GB          ║")
    print(f"║ Projection:     {estimate['projection_gb']:>6.2f} GB          ║")
    print(f"╠════════════════════════════════════╣")
    print(f"║ Online total:   {estimate['total_online_gb']:>6.2f} GB          ║")
    print(f"║ Offline total:  {estimate['total_offline_gb']:>6.2f} GB          ║")
    print("╚════════════════════════════════════╝")
