"""
CONFLUX Unit Tests

Tests for all core modules without requiring GPU or actual model downloads.
Uses synthetic data to validate mathematical correctness.
"""

import torch
import pytest
import tempfile
import json
from pathlib import Path

from conflux.config import ConfluxConfig, SourceModelConfig
from conflux.cka import linear_cka, rbf_cka, compute_cka_matrix, match_layers_hungarian, match_layers_greedy
from conflux.svd_init import residual_svd_init, batch_svd_init
from conflux.rank_alloc import allocate_ranks, generate_peft_config, _nearest_power_of_2_or_multiple_of_4
from conflux.residual import ResidualInfo, ProjectionLayer, ProjectionBank, compute_residuals
from conflux.loss import AnnealingSchedule, ResidualGuidanceLoss, ConfluxLoss
from conflux.cache import OfflineCache
from conflux.profiler import profile_layer_pairs
from conflux.utils import estimate_vram


# ─── CKA Tests ───────────────────────────────────────────────

class TestCKA:
    def test_linear_cka_identical(self):
        """Identical representations should have CKA = 1.0."""
        X = torch.randn(100, 64)
        score = linear_cka(X, X)
        assert abs(score - 1.0) < 1e-4, f"Expected ~1.0, got {score}"

    def test_linear_cka_orthogonal(self):
        """Orthogonal representations should have CKA ≈ 0."""
        X = torch.randn(100, 64)
        Y = torch.randn(100, 64)
        Q, _ = torch.linalg.qr(Y)
        score = linear_cka(X, Q[:, :64])
        assert score < 0.3, f"Expected low CKA for near-orthogonal, got {score}"

    def test_linear_cka_scaled(self):
        """CKA should be invariant to isotropic scaling."""
        X = torch.randn(100, 64)
        Y = X * 5.0
        score = linear_cka(X, Y)
        assert abs(score - 1.0) < 1e-4, f"Expected ~1.0 for scaled copy, got {score}"

    def test_linear_cka_different_dims(self):
        """CKA should work with different feature dimensions."""
        X = torch.randn(100, 64)
        Y = torch.randn(100, 128)
        score = linear_cka(X, Y)
        assert 0 <= score <= 1.0, f"CKA out of range: {score}"

    def test_cka_matrix_shape(self):
        """CKA matrix should have correct dimensions."""
        hs_w = [torch.randn(50, 32) for _ in range(4)]
        hs_m = [torch.randn(50, 48) for _ in range(6)]
        matrix = compute_cka_matrix(hs_w, hs_m)
        assert matrix.shape == (4, 6)

    def test_cka_matrix_range(self):
        """All CKA values should be in [0, 1]."""
        hs_w = [torch.randn(50, 32) for _ in range(4)]
        hs_m = [torch.randn(50, 48) for _ in range(4)]
        matrix = compute_cka_matrix(hs_w, hs_m)
        assert matrix.min() >= -0.01, f"CKA below 0: {matrix.min()}"
        assert matrix.max() <= 1.01, f"CKA above 1: {matrix.max()}"

    def test_hungarian_matching(self):
        """Hungarian matching should produce valid one-to-one mapping."""
        matrix = torch.tensor([
            [0.9, 0.1, 0.2],
            [0.1, 0.8, 0.3],
            [0.2, 0.3, 0.7],
        ])
        matches = match_layers_hungarian(matrix)
        w_indices = [m[0] for m in matches]
        m_indices = [m[1] for m in matches]
        assert len(set(w_indices)) == len(w_indices), "Duplicate W layers"
        assert len(set(m_indices)) == len(m_indices), "Duplicate M layers"

    def test_greedy_matching(self):
        """Greedy matching should produce valid mapping."""
        matrix = torch.rand(5, 8)
        matches = match_layers_greedy(matrix)
        w_indices = [m[0] for m in matches]
        assert len(set(w_indices)) == len(w_indices)


# ─── SVD Init Tests ──────────────────────────────────────────

class TestSVDInit:
    def test_svd_init_shapes(self):
        """SVD init should produce correctly shaped A and B."""
        R = torch.randn(100, 64)
        A, B = residual_svd_init(R, rank=8)
        assert A.shape == (64, 8), f"A shape: {A.shape}"
        assert B.shape == (8, 64), f"B shape: {B.shape}"

    def test_svd_init_reconstruction(self):
        """A @ B should approximate the rank-r projection of R."""
        R = torch.randn(64, 64)
        rank = 16
        A, B = residual_svd_init(R, rank=rank)
        recon = A @ B
        assert recon.shape == (64, 64)

        U, S, Vh = torch.linalg.svd(R.float(), full_matrices=False)
        best_rank_r = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
        recon_error = torch.norm(R - recon).item()
        optimal_error = torch.norm(R - best_rank_r).item()
        assert recon_error < optimal_error * 1.5, "SVD init too far from optimal"

    def test_svd_init_scaling(self):
        """Scaling factor should scale the initialization."""
        R = torch.randn(100, 64)
        A1, B1 = residual_svd_init(R, rank=8, scaling_factor=1.0)
        A2, B2 = residual_svd_init(R, rank=8, scaling_factor=0.5)
        norm1 = torch.norm(A1 @ B1).item()
        norm2 = torch.norm(A2 @ B2).item()
        assert norm2 < norm1, "Smaller scaling should produce smaller norm"

    def test_batch_svd_init(self):
        """Batch SVD should process multiple layers."""
        infos = []
        for i in range(4):
            info = ResidualInfo(
                layer_w_idx=i, layer_m_idx=i,
                cka_score=0.5, residual_magnitude=0.5,
                residual_matrix=torch.randn(50, 64),
            )
            info.assigned_rank = 8
            infos.append(info)

        results = batch_svd_init(infos)
        assert len(results) == 4
        for r in results:
            assert r.A_init.shape[1] == 8
            assert r.explained_variance_ratio > 0


# ─── Rank Allocation Tests ───────────────────────────────────

class TestRankAllocation:
    def _make_residual_infos(self, magnitudes):
        return [
            ResidualInfo(
                layer_w_idx=i, layer_m_idx=i,
                cka_score=0.5, residual_magnitude=m,
                transfer_value=m * 0.5,
            )
            for i, m in enumerate(magnitudes)
        ]

    def test_high_residual_gets_high_rank(self):
        """Layers with high residual should get higher rank."""
        infos = self._make_residual_infos([0.1, 0.5, 0.9])
        allocs = allocate_ranks(infos, rank_min=4, rank_max=64)
        ranks = [a.rank for a in allocs if not a.skip]
        assert ranks[-1] > ranks[0], f"Expected increasing ranks, got {ranks}"

    def test_low_residual_skipped(self):
        """Layers below threshold should be skipped."""
        infos = self._make_residual_infos([0.01, 0.5, 0.9])
        allocs = allocate_ranks(infos, skip_threshold=0.05)
        assert allocs[0].skip is True
        assert allocs[1].skip is False

    def test_rank_bounds(self):
        """All ranks should be within min/max bounds."""
        infos = self._make_residual_infos([0.1, 0.3, 0.5, 0.7, 0.9])
        allocs = allocate_ranks(infos, rank_min=8, rank_max=32)
        for a in allocs:
            if not a.skip:
                assert a.rank >= 8, f"Rank {a.rank} below min 8"
                assert a.rank <= 32, f"Rank {a.rank} above max 32"

    def test_nearest_multiple_of_4(self):
        assert _nearest_power_of_2_or_multiple_of_4(5) == 4
        assert _nearest_power_of_2_or_multiple_of_4(6) == 8
        assert _nearest_power_of_2_or_multiple_of_4(14) == 16
        assert _nearest_power_of_2_or_multiple_of_4(1) == 4

    def test_peft_config_generation(self):
        """Generated PEFT config should be valid."""
        infos = self._make_residual_infos([0.3, 0.6, 0.9])
        allocs = allocate_ranks(infos)
        config = generate_peft_config(allocs, target_modules=["q_proj", "v_proj"])
        assert "r" in config
        assert "rank_pattern" in config
        assert config["task_type"] == "CAUSAL_LM"


# ─── Projection Tests ────────────────────────────────────────

class TestProjection:
    def test_projection_shape(self):
        """Projection should map between dimensions."""
        proj = ProjectionLayer(128, 64)
        x = torch.randn(10, 128)
        y = proj(x)
        assert y.shape == (10, 64)

    def test_projection_bank(self):
        """Bank should handle multiple sources."""
        bank = ProjectionBank(64, {"llama": 128, "mistral": 64, "gemma": 96})
        assert "llama" in bank.projections
        assert "mistral" not in bank.projections  # same dim → identity
        assert "gemma" in bank.projections

        x = torch.randn(10, 128)
        y = bank(x, "llama")
        assert y.shape == (10, 64)


# ─── Loss Tests ──────────────────────────────────────────────

class TestLoss:
    def test_annealing_cosine(self):
        """Cosine annealing should start high and end near zero."""
        sched = AnnealingSchedule(initial_weight=0.3, schedule="cosine", total_steps=100)
        w_start = sched.get_weight(0)
        w_end = sched.get_weight(100)
        assert w_start < 0.01  # before warmup
        assert w_end < 0.01

    def test_annealing_after_warmup(self):
        sched = AnnealingSchedule(initial_weight=0.3, schedule="cosine", warmup_steps=10, total_steps=1000)
        w = sched.get_weight(10)
        assert abs(w - 0.3) < 0.01

    def test_composite_loss_no_guidance(self):
        """Without guidance, should return task loss unchanged."""
        loss_fn = ConfluxLoss()
        task = torch.tensor(2.5, requires_grad=True)
        total = loss_fn(task)
        assert abs(total.item() - 2.5) < 1e-6

    def test_composite_loss_with_guidance(self):
        """With guidance, loss should be higher than task alone."""
        loss_fn = ConfluxLoss(initial_weight=0.3, warmup_steps=0)
        loss_fn._step = 1  # past warmup

        task = torch.tensor(2.0, requires_grad=True)
        adapted = {0: torch.randn(4, 32)}
        targets = {"source": {0: torch.randn(4, 32)}}

        total = loss_fn(task, adapted, targets)
        assert total.item() > 2.0, "Guidance should increase total loss"


# ─── Cache Tests ─────────────────────────────────────────────

class TestCache:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OfflineCache(tmpdir, "test_source")
            hs = [torch.randn(100, 64) for _ in range(4)]
            cache.save_hidden_states(hs, layer_indices=[0, 1, 2, 3])

            loaded = cache.load_layer(2)
            assert loaded.shape == (100, 64)
            assert torch.allclose(hs[2], loaded, atol=1e-5)

    def test_batch_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OfflineCache(tmpdir, "test")
            hs = [torch.randn(100, 64)]
            cache.save_hidden_states(hs, [0])

            batch = cache.load_batch(0, 10, 20)
            assert batch.shape == (10, 64)

    def test_cache_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OfflineCache(tmpdir, "test")
            assert not cache.exists()
            cache.save_hidden_states([torch.randn(10, 32)], [0])
            assert cache.exists()


# ─── Profiler Tests ──────────────────────────────────────────

class TestProfiler:
    def test_full_profiling_pipeline(self):
        """End-to-end profiling with synthetic data."""
        hs_w = [torch.randn(50, 64) for _ in range(4)]
        hs_m = [torch.randn(50, 64) for _ in range(6)]

        residuals, report = profile_layer_pairs(
            hs_w, hs_m,
            primary_model_name="test_primary",
            source_model_name="test_source",
            source_alias="test",
        )

        assert len(residuals) == 4  # matches W layer count
        assert report.num_layers_primary == 4
        assert report.num_layers_source == 6
        assert report.total_profiling_time_seconds > 0

    def test_report_save_load(self):
        """Profiling report should serialize/deserialize."""
        hs_w = [torch.randn(20, 32) for _ in range(3)]
        hs_m = [torch.randn(20, 32) for _ in range(3)]

        _, report = profile_layer_pairs(hs_w, hs_m, "p", "s", "test")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            report.save(f.name)
            from conflux.profiler import ProfilingReport
            loaded = ProfilingReport.load(f.name)
            assert loaded.primary_model == "p"
            assert loaded.num_layers_primary == 3


# ─── Config Tests ────────────────────────────────────────────

class TestConfig:
    def test_default_config(self):
        config = ConfluxConfig(source_models=[SourceModelConfig("test")])
        assert config.validate()

    def test_invalid_rank_range(self):
        config = ConfluxConfig(
            source_models=[SourceModelConfig("test")],
            rank_min=64, rank_max=8,
        )
        with pytest.raises(ValueError):
            config.validate()

    def test_invalid_loss_weight(self):
        config = ConfluxConfig(
            source_models=[SourceModelConfig("test")],
            residual_loss_weight=1.5,
        )
        with pytest.raises(ValueError):
            config.validate()

    def test_no_sources(self):
        config = ConfluxConfig()
        with pytest.raises(ValueError):
            config.validate()


# ─── VRAM Estimate Tests ─────────────────────────────────────

class TestVRAM:
    def test_estimate_positive(self):
        est = estimate_vram(4, 8.0, 2, 8.0)
        assert est["total_online_gb"] > 0
        assert est["total_offline_gb"] > 0
        assert est["total_offline_gb"] < est["total_online_gb"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
