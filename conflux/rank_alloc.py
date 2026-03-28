"""
CONFLUX Adaptive Rank Allocation (Module 1)

Assigns per-layer LoRA rank proportional to cross-architecture residual magnitude.
Layers with large Δ → high rank. Layers with small Δ → skip.
Ranks rounded to multiples of 4 for hardware efficiency.
"""

import torch
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RankAllocationResult:
    """Rank + bitwidth allocation for a single layer."""
    layer_idx: int
    rank: int
    bits: int
    residual_score: float
    transfer_value: float
    skip: bool = False


def _round_to_multiple_of_4(n: int) -> int:
    """Round to nearest multiple of 4 (min 4) for hardware efficiency."""
    return max(4, round(n / 4) * 4)


def allocate_ranks(
    residual_infos: list,
    rank_min: int = 4,
    rank_max: int = 64,
    total_param_budget: Optional[int] = None,
    skip_threshold: float = 0.05,
    bitwidth_thresholds: tuple[float, float] = (0.7, 0.3),
    hidden_dim: int = 4096,
) -> list[RankAllocationResult]:
    """Allocate per-layer LoRA rank based on residual magnitudes.

    Args:
        residual_infos: objects with .layer_w_idx, .residual_magnitude, .transfer_value
        rank_min: minimum rank for non-skipped layers
        rank_max: maximum rank
        total_param_budget: optional cap on total trainable params
        skip_threshold: layers below this normalized magnitude are skipped
        bitwidth_thresholds: (high, low) for bit-width assignment
        hidden_dim: model hidden size (for budget calculation)

    Returns:
        List of RankAllocationResult per layer.
    """
    if not residual_infos:
        return []

    allocations = []

    for info in residual_infos:
        mag = info.residual_magnitude
        tv = getattr(info, 'transfer_value', mag * 0.5)

        if mag < skip_threshold:
            allocations.append(RankAllocationResult(
                layer_idx=info.layer_w_idx, rank=0, bits=2,
                residual_score=mag, transfer_value=tv, skip=True,
            ))
            continue

        raw_rank = rank_min + (rank_max - rank_min) * mag
        rank = _round_to_multiple_of_4(int(raw_rank))
        rank = max(rank_min, min(rank_max, rank))

        high_t, low_t = bitwidth_thresholds
        bits = 8 if mag > high_t else (4 if mag > low_t else 2)

        allocations.append(RankAllocationResult(
            layer_idx=info.layer_w_idx, rank=rank, bits=bits,
            residual_score=mag, transfer_value=tv,
        ))

    if total_param_budget is not None:
        active = [a for a in allocations if not a.skip]
        current = sum(2 * a.rank * hidden_dim for a in active)
        if current > total_param_budget and active:
            scale = total_param_budget / current
            for a in active:
                a.rank = _round_to_multiple_of_4(max(rank_min, int(a.rank * scale)))

    active = [a for a in allocations if not a.skip]
    skipped = [a for a in allocations if a.skip]
    total_params = sum(2 * a.rank * hidden_dim for a in active)

    logger.info(
        f"Rank allocation: {len(active)} active, {len(skipped)} skipped. "
        f"Rank range: {min((a.rank for a in active), default=0)}"
        f"–{max((a.rank for a in active), default=0)}. "
        f"Total LoRA params: {total_params:,}"
    )

    return allocations


def generate_peft_config(
    allocations: list[RankAllocationResult],
    target_modules: list[str],
    lora_alpha_multiplier: float = 2.0,
    lora_dropout: float = 0.05,
) -> dict:
    """Generate PEFT/LoRA config dict from rank allocations.

    Args:
        allocations: from allocate_ranks()
        target_modules: e.g. ["q_proj", "v_proj", ...]
        lora_alpha_multiplier: alpha = rank * multiplier
        lora_dropout: dropout rate

    Returns:
        Dict for PEFT LoraConfig with rank_pattern and alpha_pattern.
    """
    active = [a for a in allocations if not a.skip]
    if not active:
        raise ValueError("No active layers in allocation")

    median_rank = sorted(a.rank for a in active)[len(active) // 2]

    rank_pattern = {}
    alpha_pattern = {}

    for a in active:
        for module in target_modules:
            key = f"model.layers.{a.layer_idx}.{module}"
            rank_pattern[key] = a.rank
            alpha_pattern[key] = int(a.rank * lora_alpha_multiplier)

    return {
        "r": median_rank,
        "lora_alpha": int(median_rank * lora_alpha_multiplier),
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def print_allocation_summary(allocations: list[RankAllocationResult], hidden_dim: int = 4096):
    """Print human-readable summary."""
    active = [a for a in allocations if not a.skip]
    skipped = [a for a in allocations if a.skip]

    print(f"\n{'═'*50}")
    print(f" CONFLUX Rank Allocation")
    print(f"{'═'*50}")
    print(f" Active:  {len(active)}")
    print(f" Skipped: {len(skipped)}")

    if active:
        ranks = [a.rank for a in active]
        total_params = sum(2 * a.rank * hidden_dim for a in active)
        uniform_params = len(active) * 2 * max(ranks) * hidden_dim

        print(f" Ranks:   {min(ranks)} – {max(ranks)}")
        print(f" Params:  {total_params:,} (vs {uniform_params:,} uniform)")
        print(f" Savings: {1 - total_params / uniform_params:.1%}")

    print(f"{'═'*50}")

    for a in sorted(allocations, key=lambda x: x.layer_idx):
        status = "SKIP" if a.skip else f"r={a.rank:>3} {a.bits}bit"
        print(f"  Layer {a.layer_idx:>3}: {status:>14}  residual={a.residual_score:.4f}")


class AdaptiveRankAllocator:
    """Class wrapper around the function API for backward compatibility.

    Example:
        alloc = AdaptiveRankAllocator(r_min=8, r_max=48)
        results = alloc.allocate(residual_infos, hidden_dim=4096)
    """

    def __init__(self, r_min=4, r_max=64, skip_threshold=0.05):
        self.r_min = r_min
        self.r_max = r_max
        self.skip_threshold = skip_threshold

    def allocate(self, residual_infos: list, hidden_dim: int = 4096,
                 total_param_budget: Optional[int] = None) -> list[RankAllocationResult]:
        return allocate_ranks(
            residual_infos, rank_min=self.r_min, rank_max=self.r_max,
            skip_threshold=self.skip_threshold, hidden_dim=hidden_dim,
            total_param_budget=total_param_budget,
        )
