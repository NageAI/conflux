"""
CONFLUX Informativeness Profiler — Module 4

Three-dimensional scoring for each layer pair:
  1. CKA similarity (structural alignment)
  2. Residual magnitude (knowledge gap size)
  3. Task relevance (is the gap useful for our task?)

transfer_value = (1 - cka) * residual * task_relevance
"""

import torch
from typing import Optional
from dataclasses import dataclass
import logging

from conflux.cka import CKAComputer
from conflux.residual import ResidualExtractor, ResidualInfo

logger = logging.getLogger(__name__)


@dataclass
class LayerProfile:
    """Profile for a single layer pair."""
    w_layer: int
    m_layer: int
    source_name: str
    cka_score: float
    residual_magnitude: float
    task_relevance: float
    transfer_value: float
    recommended_rank: int
    recommended_bits: int


class InformativenessProfiler:
    """
    Profiles all layer pairs to determine transfer value.

    Combines CKA (structural similarity), residual magnitude (gap size),
    and optional task relevance (gap usefulness) into a single score.

    High transfer_value → this layer pair is a rich source of knowledge.
    Low transfer_value → skip or minimal rank.
    """

    def __init__(
        self,
        cka_computer: Optional[CKAComputer] = None,
        task_relevance_method: str = "uniform",
    ):
        """
        Args:
            cka_computer: CKA computation instance (created if None)
            task_relevance_method: "uniform" (all equal), "ablation" (compute per-layer),
                                   or "gradient" (gradient-based importance)
        """
        self.cka = cka_computer or CKAComputer(kernel="linear")
        self.task_relevance_method = task_relevance_method

    def profile(
        self,
        hidden_states_w: list[torch.Tensor],
        hidden_states_m: list[torch.Tensor],
        layer_matching: list[tuple[int, int, float]],
        residual_infos: list[ResidualInfo],
        source_name: str = "source",
    ) -> list[LayerProfile]:
        """
        Compute full profile for all matched layer pairs.

        Returns list of LayerProfile sorted by transfer_value (descending).
        """
        # Normalize residual magnitudes
        mags = [info.magnitude for info in residual_infos]
        max_mag = max(mags) if mags else 1.0
        norm_mags = {info.w_layer: info.magnitude / max_mag for info in residual_infos}

        # Build CKA lookup
        cka_lookup = {(w, m): cka for w, m, cka in layer_matching}

        # Compute task relevance
        task_scores = self._compute_task_relevance(
            hidden_states_w, residual_infos
        )

        profiles = []
        for info in residual_infos:
            cka_score = cka_lookup.get((info.w_layer, info.m_layer), 0.5)
            norm_mag = norm_mags.get(info.w_layer, 0.0)
            task_rel = task_scores.get(info.w_layer, 1.0)

            # Core formula: transfer_value = (1 - cka) * residual * task_relevance
            transfer_value = (1.0 - cka_score) * norm_mag * task_rel

            profiles.append(LayerProfile(
                w_layer=info.w_layer,
                m_layer=info.m_layer,
                source_name=source_name,
                cka_score=cka_score,
                residual_magnitude=info.magnitude,
                task_relevance=task_rel,
                transfer_value=transfer_value,
                recommended_rank=0,
                recommended_bits=4,
            ))

        # Sort by transfer value
        profiles.sort(key=lambda p: -p.transfer_value)

        logger.info(
            f"Profiled {len(profiles)} layer pairs. "
            f"Top transfer value: {profiles[0].transfer_value:.4f} "
            f"(W[{profiles[0].w_layer}] ↔ M[{profiles[0].m_layer}])"
            if profiles else "No profiles computed"
        )

        return profiles

    def _compute_task_relevance(
        self,
        hidden_states_w: list[torch.Tensor],
        residual_infos: list,
    ) -> dict[int, float]:
        """Compute per-layer task relevance scores."""
        if self.task_relevance_method == "uniform":
            return {info.w_layer: 1.0 for info in residual_infos}

        elif self.task_relevance_method == "gradient":
            # Placeholder: in practice, requires a forward pass with task loss
            # and gradient magnitude per layer
            logger.warning("Gradient-based relevance requires task data. Using uniform.")
            return {info.w_layer: 1.0 for info in residual_infos}

        elif self.task_relevance_method == "ablation":
            # Placeholder: requires ablation study
            logger.warning("Ablation relevance not yet implemented. Using uniform.")
            return {info.w_layer: 1.0 for info in residual_infos}

        return {info.w_layer: 1.0 for info in residual_infos}

    def summary(self, profiles: list[LayerProfile]) -> str:
        """Generate human-readable profiling summary."""
        lines = [
            f"CONFLUX Informativeness Profile",
            f"{'='*60}",
            f"Layer pairs profiled: {len(profiles)}",
            f"",
            f"{'Layer':>8} {'CKA':>8} {'|Δ|':>10} {'Task':>8} {'Transfer':>10}",
            f"{'-'*52}",
        ]
        for p in profiles[:20]:
            lines.append(
                f"W[{p.w_layer:2d}]↔M[{p.m_layer:2d}] "
                f"{p.cka_score:8.4f} {p.residual_magnitude:10.4f} "
                f"{p.task_relevance:8.4f} {p.transfer_value:10.4f}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Function API (used by ConfluxTrainer)
# ═══════════════════════════════════════════════════════════

def profile_layer_pairs(
    hidden_states_w: list[torch.Tensor],
    hidden_states_m: list[torch.Tensor],
    primary_model_name: str = "",
    source_model_name: str = "",
    source_alias: str = "source",
    cka_kernel: str = "linear",
    matching_method: str = "cka_hungarian",
    min_transfer_value: float = 0.1,
) -> tuple[list, object]:
    """Run full profiling pipeline for a model pair.

    Combines CKA → matching → residual extraction → profiling.

    Args:
        hidden_states_w: Per-layer hidden states from W.
        hidden_states_m: Per-layer hidden states from M.
        primary_model_name: Name of primary model.
        source_model_name: Name of source model.
        source_alias: Short alias.
        cka_kernel: "linear" or "rbf".
        matching_method: "cka_hungarian" or "cka_greedy".
        min_transfer_value: Minimum transfer value for active layers.

    Returns:
        Tuple of (residual_infos, profiling_report_dict)
    """
    import time
    from conflux.cka import CKAComputer, compute_cka_matrix, match_layers_hungarian, match_layers_greedy
    from conflux.residual import ResidualExtractor, ResidualInfo

    start = time.time()
    n_w = len(hidden_states_w)
    n_m = len(hidden_states_m)

    logger.info(f"Profiling: {primary_model_name} ({n_w}L) × {source_model_name} ({n_m}L)")

    # Step 1: CKA matrix
    cka_matrix = compute_cka_matrix(hidden_states_w, hidden_states_m, kernel=cka_kernel)

    # Step 2: Layer matching
    if matching_method == "cka_hungarian":
        matches = match_layers_hungarian(cka_matrix)
    else:
        matches = match_layers_greedy(cka_matrix)

    # Step 3: Residual extraction
    d_w = hidden_states_w[0].shape[-1] if hidden_states_w[0].dim() == 2 else hidden_states_w[0].shape[-1]
    d_m = hidden_states_m[0].shape[-1] if hidden_states_m[0].dim() == 2 else hidden_states_m[0].shape[-1]

    extractor = ResidualExtractor(d_primary=d_w, d_source=d_m)
    residual_infos = extractor.extract_all_layers(
        hidden_states_w, hidden_states_m, matches, source_name=source_alias,
    )

    # Step 4: Profiling (transfer value computation)
    profiler = InformativenessProfiler(
        cka_computer=CKAComputer(kernel=cka_kernel),
    )
    profiles = profiler.profile(
        hidden_states_w, hidden_states_m, matches, residual_infos, source_alias,
    )

    # Set transfer_value on residual_infos
    profile_lookup = {p.w_layer: p.transfer_value for p in profiles}
    for info in residual_infos:
        tv = profile_lookup.get(info.w_layer, 0.0)
        object.__setattr__(info, 'transfer_value', tv)

    active = sum(1 for p in profiles if p.transfer_value >= min_transfer_value)
    elapsed = time.time() - start

    # Build report dict
    report = type('ProfilingReport', (), {
        'primary_model': primary_model_name,
        'source_model': source_model_name,
        'source_alias': source_alias,
        'num_layers_primary': n_w,
        'num_layers_source': n_m,
        'recommended_active_layers': active,
        'recommended_skip_layers': len(residual_infos) - active,
        'total_profiling_time_seconds': round(elapsed, 2),
        'transfer_value_stats': {
            'mean': round(sum(p.transfer_value for p in profiles) / max(len(profiles), 1), 4),
            'max': round(max((p.transfer_value for p in profiles), default=0), 4),
        },
        'save': lambda self, path: None,
    })()

    logger.info(f"Profiling done in {elapsed:.1f}s. {active} active / {len(residual_infos) - active} skip.")

    return residual_infos, report


def print_profiling_summary(report):
    """Print human-readable profiling summary."""
    print(f"\n{'='*56}")
    print(f" CONFLUX Profiling Report")
    print(f"{'='*56}")
    print(f" Primary:  {report.primary_model}")
    print(f" Source:   {report.source_model} ({report.source_alias})")
    print(f" Layers:   {report.num_layers_primary} (W) × {report.num_layers_source} (M)")
    print(f" Active:   {report.recommended_active_layers}")
    print(f" Skipped:  {report.recommended_skip_layers}")
    print(f" Time:     {report.total_profiling_time_seconds:.1f}s")
    print(f" Transfer: mean={report.transfer_value_stats['mean']:.4f}, max={report.transfer_value_stats['max']:.4f}")
    print(f"{'='*56}")
