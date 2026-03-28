# Changelog

All notable changes to CONFLUX are documented here.

## [0.3.0] - 2026-03-28

### Added
- First experimental results: Qwen3-8B + Llama-3.1-8B A/B comparison
- Validation loss 0.671 vs baseline 0.717 (6.4% improvement)
- 2x convergence acceleration documented
- Updated README with results, badges, and quick start guide
- CHANGELOG.md

### Changed
- Package name: `conflux` → `conflux-ft` (avoid PyPI conflicts)
- Build backend: fixed `setuptools.build_meta`

### Fixed
- Critical finding documented: balanced SVD init (A+B) causes loss explosion; asymmetric init (A-only, scale=0.01) is correct approach

## [0.2.0] - 2026-03-26

### Added
- Procrustes alignment (closed-form orthogonal, zero trainable params)
- Mini-batch CKA profiling (256 samples, down from 2048)
- `skip_early_layers` parameter (default 8)
- Per-layer learnable alpha in ResidualGuidanceLoss
- Adaptive lambda annealing: lambda(t) = lambda_0 * (L_task(t)/L_task(0))^beta
- Forgetting benchmark (MMLU subset + general QA)
- Class-based API: CKAComputer, ResidualSVDInitializer, AdaptiveRankAllocator
- Function-based API wrappers for pipeline integration

### Changed
- SVD initialization rewritten: clean covariance SVD replacing overlapping attempts
- CKA profiling 85% faster with mini-batch approach

## [0.1.1] - 2026-03-25

### Fixed
- All 15 trainer import compatibility issues resolved
- API compatibility wrappers for cka, svd_init, rank_alloc, residual, profiler

## [0.1.0] - 2026-03-24

### Added
- Initial implementation of all 5 modules
- ConfluxTrainer orchestrator
- Fehm and Cortex training examples
- CI/CD workflows (lint, test, type-check, build)
- Apache 2.0 license
