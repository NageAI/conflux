"""
CONFLUX Composite Loss Module

Implements the dual-loss training objective:

    L_total = L_task + λ(t) · L_dual

Where:
    L_task = standard language modeling loss (cross-entropy)
    L_dual = residual guidance loss (MSE toward target representations)
    λ(t)  = annealing schedule (cosine, linear, or constant)

The residual guidance loss pushes LoRA adapters toward the cross-architecture
knowledge target during early training, then fades to let task loss dominate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)


class AnnealingSchedule:
    """Lambda annealing schedule for residual loss weight.

    v2 adds "adaptive" mode: λ decays based on task loss improvement,
    not just time. If task loss is dropping fast, guidance fades early.
    If task loss plateaus, guidance stays longer.

    Formula (adaptive): λ(t) = λ_0 · max(0, L_task(t) / L_task(0))^β
    """

    def __init__(
        self,
        initial_weight: float = 0.3,
        schedule: Literal["cosine", "linear", "constant", "adaptive"] = "adaptive",
        warmup_steps: int = 100,
        total_steps: int = 10000,
        adaptive_beta: float = 2.0,
    ):
        self.initial_weight = initial_weight
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.adaptive_beta = adaptive_beta
        self._initial_task_loss = None

    def get_weight(self, step: int, task_loss: Optional[float] = None) -> float:
        """Get λ(t) at current training step.

        Args:
            step: Current training step.
            task_loss: Current task loss (required for adaptive mode).
        """
        if step < self.warmup_steps:
            return self.initial_weight * (step / max(self.warmup_steps, 1))

        if self.schedule == "adaptive" and task_loss is not None:
            if self._initial_task_loss is None:
                self._initial_task_loss = task_loss
            ratio = task_loss / (self._initial_task_loss + 1e-8)
            return self.initial_weight * max(0.0, ratio) ** self.adaptive_beta

        progress = min(1.0, (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1))

        if self.schedule == "cosine":
            return self.initial_weight * 0.5 * (1 + math.cos(math.pi * progress))
        elif self.schedule == "linear":
            return self.initial_weight * (1 - progress)
        elif self.schedule == "constant":
            return self.initial_weight
        else:
            return self.initial_weight


class ResidualGuidanceLoss(nn.Module):
    """Dual-source residual guidance loss (L_dual).

    v2 adds per-layer learnable scaling: each layer gets its own α_i
    that is learned during training. Cost: one scalar per layer (~0 params).

    L_dual = Σ_k Σ_i  α_source_k · softplus(α_layer_i) · ‖h_adapted_i - h_target_ki‖²
    """

    def __init__(
        self,
        source_weights: Optional[dict[str, float]] = None,
        normalize: bool = True,
        num_layers: int = 32,
        per_layer_alpha: bool = True,
    ):
        super().__init__()
        self.source_weights = source_weights or {}
        self.normalize = normalize
        self.per_layer_alpha = per_layer_alpha

        if per_layer_alpha:
            # Learnable per-layer scaling (initialized to 0 → softplus(0) = 0.693)
            self.layer_alpha = nn.Parameter(torch.zeros(num_layers))
        else:
            self.layer_alpha = None

    def forward(
        self,
        adapted_hidden_states: dict[int, torch.Tensor],
        target_hidden_states: dict[str, dict[int, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute residual guidance loss."""
        total_loss = torch.tensor(0.0, device=next(iter(adapted_hidden_states.values())).device)
        count = 0

        for source_alias, targets in target_hidden_states.items():
            alpha_source = self.source_weights.get(source_alias, 1.0)

            for layer_idx, target in targets.items():
                if layer_idx not in adapted_hidden_states:
                    continue

                adapted = adapted_hidden_states[layer_idx]

                if self.normalize:
                    adapted = F.normalize(adapted, p=2, dim=-1)
                    target = F.normalize(target, p=2, dim=-1)

                layer_loss = F.mse_loss(adapted, target.detach())

                # Per-layer scaling
                if self.layer_alpha is not None and layer_idx < len(self.layer_alpha):
                    alpha_layer = F.softplus(self.layer_alpha[layer_idx])
                else:
                    alpha_layer = 1.0

                total_loss = total_loss + alpha_source * alpha_layer * layer_loss
                count += 1

        if count > 0:
            total_loss = total_loss / count

        return total_loss


class ConfluxLoss(nn.Module):
    """Complete CONFLUX composite loss.

    Combines task loss (cross-entropy) with residual guidance loss,
    controlled by an annealing schedule.

    L_total = L_task + λ(t) · L_dual
    """

    def __init__(
        self,
        initial_weight: float = 0.3,
        annealing_schedule: str = "adaptive",
        warmup_steps: int = 100,
        total_steps: int = 10000,
        adaptive_beta: float = 2.0,
        source_weights: Optional[dict[str, float]] = None,
        normalize_guidance: bool = True,
        num_layers: int = 32,
        per_layer_alpha: bool = True,
    ):
        super().__init__()
        self.guidance_loss = ResidualGuidanceLoss(
            source_weights=source_weights,
            normalize=normalize_guidance,
            num_layers=num_layers,
            per_layer_alpha=per_layer_alpha,
        )
        self.annealing = AnnealingSchedule(
            initial_weight=initial_weight,
            schedule=annealing_schedule,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            adaptive_beta=adaptive_beta,
        )
        self._step = 0
        self._last_task_loss = 0.0
        self._last_guidance_loss = 0.0
        self._last_lambda = 0.0

    def forward(
        self,
        task_loss: torch.Tensor,
        adapted_hidden_states: Optional[dict[int, torch.Tensor]] = None,
        target_hidden_states: Optional[dict[str, dict[int, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Compute composite loss.

        Args:
            task_loss: Standard language modeling loss.
            adapted_hidden_states: Current model hidden states (optional).
            target_hidden_states: Target representations from source models (optional).

        Returns:
            Composite loss = L_task + λ(t) · L_dual
        """
        self._last_task_loss = task_loss.item()

        if adapted_hidden_states is None or target_hidden_states is None:
            self._last_guidance_loss = 0.0
            self._last_lambda = 0.0
            self._step += 1
            return task_loss

        lam = self.annealing.get_weight(self._step, task_loss=task_loss.item())
        self._last_lambda = lam

        if lam < 1e-6:
            self._last_guidance_loss = 0.0
            self._step += 1
            return task_loss

        guidance = self.guidance_loss(adapted_hidden_states, target_hidden_states)
        self._last_guidance_loss = guidance.item()

        total = task_loss + lam * guidance

        self._step += 1
        return total

    def get_metrics(self) -> dict[str, float]:
        """Get current loss components for logging."""
        return {
            "loss/task": self._last_task_loss,
            "loss/guidance": self._last_guidance_loss,
            "loss/lambda": self._last_lambda,
            "loss/total": self._last_task_loss + self._last_lambda * self._last_guidance_loss,
        }
