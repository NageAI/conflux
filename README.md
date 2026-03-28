# CONFLUX

**Cross-architecture Optimized N-source Fine-tuning via Low-rank Unified eXtraction**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

CONFLUX is a parameter-efficient fine-tuning framework that uses **cross-architecture knowledge transfer** to improve LoRA initialization. Instead of starting from zeros, CONFLUX extracts the knowledge residual between two frozen models of different architectures and uses SVD decomposition to initialize LoRA adapters in the direction of the knowledge gap.

Part of the [Nage](https://nage.ai) AI ecosystem.

## Results

First experimental validation: **Qwen3-8B** fine-tuned on 48,754 conversations, with **Llama-3.1-8B** as the knowledge source.

|  | Train Loss @400 | Val Loss @400 | Train Loss @800 | Val Loss @800 |
|---|:---:|:---:|:---:|:---:|
| Standard QLoRA (baseline) | 0.748 | 0.749 | 0.687 | 0.717 |
| **CONFLUX** | **0.713** | **0.716** | **0.686** | **0.671** |
| Delta | -0.035 | -0.033 | -0.001 | **-0.046** |

**Key findings:**
- **6.4% lower validation loss** at the same compute budget (800 steps)
- **2x convergence acceleration** — reaches baseline's final val loss by step 400
- **15 minutes overhead** for profiling + SVD init on A100 80GB
- CKA similarity (Qwen3-8B vs Llama-3.1-8B): **0.817 mean**
- SVD initialization captures **99.5%** of residual variance (EVR = 0.9946)

> **Critical finding:** Balanced SVD init (both A and B nonzero) causes loss explosion (9.83). Use **asymmetric init** — only `lora_A` with small scaling (0.01), keep `lora_B` at zeros.

## How it works

Different neural architectures trained on similar data converge to similar internal representations (CKA 0.7-0.9). CONFLUX exploits this:

```
Residual delta = align(h_source) - h_target     # what source knows that target doesn't
C = delta^T @ delta / n                          # covariance of knowledge gap
C = U @ diag(lambda) @ U^T                       # eigendecomposition
lora_A_init = sqrt(lambda_r) * U_r^T * alpha     # top-r directions, small scale
lora_B_init = 0                                   # preserve zero-output at init
```

The optimizer starts with directional knowledge of where to go, instead of random exploration.

## Five modules

| Module | Purpose |
|--------|---------|
| **Residual-Aware Rank Allocation** | Per-layer adaptive rank (4-48) based on residual magnitude |
| **Residual-SVD Initialization** | Initialize lora_A from SVD of cross-architecture residual |
| **Dual-Quantized Pipeline** | Target at NF4, source at NF4, sequential extraction |
| **Informativeness Profiling** | CKA layer matching + Procrustes alignment |
| **Multi-Source Switching** | N-source adapter blending for routing |

## Installation

```bash
# Core (torch + numpy only)
pip install conflux-ft

# Full (with transformers, peft, trl)
pip install conflux-ft[full]

# From source
git clone https://github.com/nage-ai/conflux.git
cd conflux
pip install -e ".[dev,full]"
```

## Quick start

```python
import torch
from unsloth import FastLanguageModel
from conflux.svd_init import residual_svd_init

# 1. Extract hidden states from both models
target_model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-8B-unsloth-bnb-4bit", max_seq_length=2048, load_in_4bit=True
)

def get_hidden(model, tok, texts):
    """Extract mean hidden states per layer."""
    model.eval()
    hs = {}
    with torch.no_grad():
        for text in texts:
            inp = tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
            out = model(**inp, output_hidden_states=True)
            for li, h in enumerate(out.hidden_states):
                if li not in hs: hs[li] = []
                hs[li].append(h.mean(1).squeeze(0).cpu().float())
    return {k: torch.stack(v) for k, v in hs.items()}

hs_target = get_hidden(target_model, tok, calibration_texts)
# ... same for source model

# 2. CKA layer matching (skip first 8 layers)
matches = {}  # target_layer -> (source_layer, cka_score)
for tl in range(8, n_target_layers):
    best_cka, best_sl = -1, 0
    X = hs_target[tl]; X_c = X - X.mean(0)
    for sl in range(8, n_source_layers):
        Y = hs_source[sl]; Y_c = Y - Y.mean(0)
        cka = torch.norm(X_c.T @ Y_c, "fro") ** 2 / (
            torch.norm(X_c.T @ X_c, "fro") * torch.norm(Y_c.T @ Y_c, "fro") + 1e-10
        )
        if cka.item() > best_cka:
            best_cka, best_sl = cka.item(), sl
    matches[tl] = (best_sl, best_cka)

# 3. SVD init from residuals
for tl, (sl, cka) in matches.items():
    residual = (hs_source[sl] - hs_target[tl]).float()
    A, B, svals, evr = residual_svd_init(residual, rank=48, scaling=0.5)
    svd_results[tl] = {"A": A, "evr": evr}

# 4. Apply to LoRA — A-only with small scale!
for name, param in model.named_parameters():
    if "lora_A" not in name or not param.requires_grad:
        continue
    layer_idx = int(name.split(".")[name.split(".").index("layers") + 1])
    if layer_idx in svd_results:
        A_init = svd_results[layer_idx]["A"][:param.shape[0], :param.shape[1]] * 0.01
        if A_init.shape == param.shape:
            param.data.copy_(A_init.to(param.device, param.dtype))

# 5. Train with standard SFT — same hyperparams, better results
```

## Full pipeline

```python
from conflux import ConfluxConfig, SourceModelConfig, ConfluxTrainer

config = ConfluxConfig(
    primary_model_id="Qwen/Qwen3-8B",
    primary_quantization_bits=4,
    source_models=[
        SourceModelConfig("llama", "meta-llama/Llama-3.1-8B", quantization_bits=4),
    ],
    rank_min=4,
    rank_max=48,
    skip_early_layers=8,
    profiling_samples=256,
    projection_mode="procrustes",
)

trainer = ConfluxTrainer(config)
trainer.run(calibration_texts=your_texts)
peft_model = trainer.create_peft_model(alias="llama")
```

## Project structure

```
conflux/
├── conflux/
│   ├── __init__.py       # Public API
│   ├── config.py         # ConfluxConfig
│   ├── cka.py            # CKA similarity
│   ├── residual.py       # Residual extraction + Procrustes
│   ├── svd_init.py       # SVD initialization
│   ├── rank_alloc.py     # Adaptive rank allocation
│   ├── profiler.py       # Informativeness profiling
│   ├── loss.py           # Composite loss
│   ├── trainer.py        # ConfluxTrainer orchestrator
│   ├── eval.py           # Forgetting benchmark
│   ├── cache.py          # Offline caching
│   └── utils.py          # VRAM estimation
├── examples/
│   ├── fehm_training.py
│   └── cortex_multi_source.py
├── tests/
│   └── test_conflux.py
├── pyproject.toml
├── Makefile
├── LICENSE
├── CONTRIBUTING.md
└── CHANGELOG.md
```

## Nage AI model portfolio

| Model | Size | Source | Status |
|-------|------|--------|--------|
| **Fehm** | 8B | Llama-3.1-8B | Validated (val loss -6.4%) |
| **Forge** | 14B | CodeLlama-13B | Planned |
| **Cortex** | 14B | Llama + Mistral | Planned (dual-source) |
| **Naksh** | 8B | Qwen3-Coder-Next | Planned |
| **Nuve** | 8B | Fehm (distill) | Planned |

## Citation

```bibtex
@article{asim2026conflux,
  title={CONFLUX: Cross-architecture Optimized N-source Fine-tuning
         via Low-rank Unified Extraction},
  author={Asim, Omer},
  year={2026},
  note={Nage AI. https://github.com/nage-ai/conflux}
}
```

## License

Apache 2.0
