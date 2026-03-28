"""
CONFLUX Example: Training Fehm-8B with Llama-3 knowledge transfer

This example demonstrates the complete CONFLUX pipeline for the
Nage ecosystem's Fehm-8B model, using Llama-3-8B as the source
model for cross-architecture knowledge transfer.

Requirements:
    pip install conflux[full]
    # GPU: A100 40GB or RTX 4090 24GB (with offline mode)
"""

from conflux import ConfluxConfig, SourceModelConfig, ConfluxTrainer
from conflux.utils import estimate_vram, print_vram_estimate


def main():
    # ── Step 1: Estimate VRAM ──────────────────────────────

    vram = estimate_vram(
        primary_bits=4,
        primary_params_b=8.0,       # Qwen3-8B
        source_bits=2,
        source_params_b=8.0,        # Llama-3-8B
        lora_rank=32,
        num_lora_layers=32,
        hidden_dim=4096,
        batch_size=4,
    )
    print_vram_estimate(vram)

    # ── Step 2: Configure ──────────────────────────────────

    config = ConfluxConfig(
        # Primary model (W) — the model being fine-tuned
        primary_model="Qwen/Qwen3-8B",
        primary_quantization_bits=4,

        # Source model (M) — the knowledge donor
        source_models=[
            SourceModelConfig(
                model_name_or_path="meta-llama/Llama-3-8B-Instruct",
                quantization_bits=2,
                alias="llama3",
            ),
        ],

        # Rank allocation (Module 1)
        rank_min=8,
        rank_max=48,
        rank_allocation_strategy="residual",

        # SVD initialization (Module 2)
        svd_init_enabled=True,
        svd_num_calibration_samples=512,

        # Profiling (Module 4)
        profiling_num_samples=2048,
        cka_kernel="linear",
        layer_matching_method="cka_hungarian",
        min_transfer_value=0.1,

        # Training
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        learning_rate=2e-5,
        residual_loss_weight=0.3,
        residual_loss_annealing="cosine",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        max_seq_length=2048,

        # Offline mode (for 24GB GPUs)
        offline_extraction=False,

        # Output
        output_dir="./conflux_output_fehm",
        save_profiling_report=True,
    )

    # ── Step 3: Run CONFLUX Pipeline ──────────────────────

    trainer = ConfluxTrainer(config)

    # Calibration texts — use a subset of your training data
    calibration_texts = [
        "Explain the concept of tokenization in natural language processing.",
        "What are the key differences between supervised and unsupervised learning?",
        "Write a Python function that implements binary search.",
        "Describe the architecture of a transformer model.",
        "What is the role of attention mechanisms in modern neural networks?",
    ] * 100  # 500 samples

    trainer.run(calibration_texts=calibration_texts)

    # ── Step 4: Create PEFT Model ─────────────────────────

    # Get the initialized model, ready for SFT
    peft_model = trainer.create_peft_model(alias="llama3")

    # ── Step 5: Train with TRL/SFTTrainer ─────────────────

    # from trl import SFTTrainer, SFTConfig
    # from datasets import load_dataset
    #
    # dataset = load_dataset("your_fehm_dataset")
    #
    # sft_config = SFTConfig(
    #     output_dir="./fehm_sft",
    #     num_train_epochs=3,
    #     per_device_train_batch_size=4,
    #     gradient_accumulation_steps=8,
    #     learning_rate=2e-5,
    #     warmup_ratio=0.05,
    #     max_seq_length=2048,
    #     logging_steps=10,
    # )
    #
    # sft_trainer = SFTTrainer(
    #     model=peft_model,
    #     train_dataset=dataset,
    #     args=sft_config,
    # )
    # sft_trainer.train()

    print("\n✓ CONFLUX pipeline complete. Model ready for SFT training.")


if __name__ == "__main__":
    main()
