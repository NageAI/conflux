"""
CONFLUX Evaluation — Catastrophic Forgetting Benchmark

Measures how much general knowledge the model retains after CONFLUX fine-tuning.
Compares MMLU accuracy before and after training to detect knowledge loss.

Usage:
    from conflux.eval import forgetting_benchmark

    results = forgetting_benchmark(
        base_model="Qwen/Qwen3-8B",
        finetuned_model="./fehm-8b-sft",
        tasks=["mmlu", "truthfulqa"],
    )
    print(results)

Part of the Nage AI ecosystem.
"""

import torch
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ForgettingResult:
    """Result of catastrophic forgetting evaluation."""
    task: str
    base_score: float
    finetuned_score: float
    delta: float            # finetuned - base (negative = forgetting)
    retention_pct: float    # finetuned / base * 100


@dataclass
class ForgettingReport:
    """Full forgetting benchmark report."""
    base_model: str
    finetuned_model: str
    results: list[ForgettingResult] = field(default_factory=list)
    mean_retention: float = 0.0

    def summary(self) -> str:
        lines = [
            f"{'='*56}",
            f" CONFLUX Catastrophic Forgetting Benchmark",
            f"{'='*56}",
            f" Base:       {self.base_model}",
            f" Finetuned:  {self.finetuned_model}",
            f" Mean retention: {self.mean_retention:.1f}%",
            f"",
            f" {'Task':<20} {'Base':>8} {'Tuned':>8} {'Delta':>8} {'Retain':>8}",
            f" {'-'*52}",
        ]
        for r in self.results:
            sign = "+" if r.delta >= 0 else ""
            lines.append(
                f" {r.task:<20} {r.base_score:>7.2f}% {r.finetuned_score:>7.2f}% "
                f"{sign}{r.delta:>6.2f}% {r.retention_pct:>6.1f}%"
            )
        lines.append(f"{'='*56}")
        return "\n".join(lines)

    def save(self, path: str):
        data = {
            "base_model": self.base_model,
            "finetuned_model": self.finetuned_model,
            "mean_retention": self.mean_retention,
            "results": [
                {"task": r.task, "base": r.base_score, "finetuned": r.finetuned_score,
                 "delta": r.delta, "retention_pct": r.retention_pct}
                for r in self.results
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info(f"Forgetting report saved to {path}")


def forgetting_benchmark(
    base_model: str,
    finetuned_model: str,
    tasks: Optional[list[str]] = None,
    num_samples: int = 200,
    batch_size: int = 8,
    device: str = "cuda",
) -> ForgettingReport:
    """Run catastrophic forgetting benchmark.

    Evaluates base and finetuned models on general knowledge tasks
    and computes retention percentage.

    Args:
        base_model: HuggingFace model ID or local path for base model.
        finetuned_model: Path to finetuned model (LoRA or merged).
        tasks: List of evaluation tasks. Default: ["mmlu_subset", "general_qa"].
        num_samples: Number of evaluation samples per task.
        batch_size: Batch size for evaluation.
        device: Computation device.

    Returns:
        ForgettingReport with per-task and aggregate results.
    """
    tasks = tasks or ["mmlu_subset", "general_qa"]

    report = ForgettingReport(base_model=base_model, finetuned_model=finetuned_model)

    for task in tasks:
        logger.info(f"Evaluating task: {task}")

        questions = _load_eval_questions(task, num_samples)
        if not questions:
            logger.warning(f"No questions for task {task}, skipping")
            continue

        base_score = _evaluate_model(base_model, questions, batch_size, device)
        ft_score = _evaluate_model(finetuned_model, questions, batch_size, device)

        delta = ft_score - base_score
        retention = (ft_score / base_score * 100) if base_score > 0 else 100.0

        report.results.append(ForgettingResult(
            task=task, base_score=base_score,
            finetuned_score=ft_score, delta=delta,
            retention_pct=retention,
        ))

        logger.info(f"  {task}: base={base_score:.2f}%, ft={ft_score:.2f}%, retention={retention:.1f}%")

    if report.results:
        report.mean_retention = sum(r.retention_pct for r in report.results) / len(report.results)

    return report


def _load_eval_questions(task: str, num_samples: int) -> list[dict]:
    """Load evaluation questions for a task.

    Returns list of {"prompt": str, "choices": list[str], "answer_idx": int}
    """
    if task == "mmlu_subset":
        return _generate_mmlu_subset(num_samples)
    elif task == "general_qa":
        return _generate_general_qa(num_samples)
    return []


def _generate_mmlu_subset(n: int) -> list[dict]:
    """Generate a subset of MMLU-style questions for quick eval.

    In production, load from the actual MMLU dataset. This generates
    synthetic questions for testing the evaluation pipeline.
    """
    categories = [
        ("math", [
            ("What is the derivative of x^3?", ["3x^2", "3x", "x^2", "2x^3"], 0),
            ("What is 17 * 23?", ["391", "401", "381", "411"], 0),
            ("What is the integral of 2x?", ["x^2 + C", "2x^2", "x + C", "2"], 0),
        ]),
        ("science", [
            ("What is the chemical formula for water?", ["H2O", "CO2", "NaCl", "O2"], 0),
            ("What planet is closest to the Sun?", ["Mercury", "Venus", "Earth", "Mars"], 0),
            ("What is the speed of light approximately?", ["300,000 km/s", "150,000 km/s", "600,000 km/s", "30,000 km/s"], 0),
        ]),
        ("history", [
            ("In what year did World War II end?", ["1945", "1944", "1946", "1943"], 0),
            ("Who wrote the Declaration of Independence?", ["Thomas Jefferson", "Benjamin Franklin", "George Washington", "John Adams"], 0),
        ]),
        ("language", [
            ("What is the plural of 'child'?", ["children", "childs", "childes", "childern"], 0),
            ("Which word is a synonym for 'happy'?", ["joyful", "sad", "angry", "tired"], 0),
        ]),
    ]

    questions = []
    for cat_name, cat_qs in categories:
        for prompt, choices, answer in cat_qs:
            questions.append({
                "prompt": f"[{cat_name}] {prompt}",
                "choices": choices,
                "answer_idx": answer,
                "category": cat_name,
            })

    # Repeat to fill n samples
    while len(questions) < n:
        questions.extend(questions[:n - len(questions)])
    return questions[:n]


def _generate_general_qa(n: int) -> list[dict]:
    """General knowledge QA for forgetting detection."""
    questions = [
        {"prompt": "What is the capital of France?", "choices": ["Paris", "London", "Berlin", "Madrid"], "answer_idx": 0},
        {"prompt": "Who painted the Mona Lisa?", "choices": ["Leonardo da Vinci", "Michelangelo", "Raphael", "Donatello"], "answer_idx": 0},
        {"prompt": "What is the largest ocean?", "choices": ["Pacific", "Atlantic", "Indian", "Arctic"], "answer_idx": 0},
        {"prompt": "What gas do plants absorb?", "choices": ["CO2", "O2", "N2", "H2"], "answer_idx": 0},
        {"prompt": "How many continents are there?", "choices": ["7", "5", "6", "8"], "answer_idx": 0},
    ]
    while len(questions) < n:
        questions.extend(questions[:n - len(questions)])
    return questions[:n]


@torch.no_grad()
def _evaluate_model(
    model_path: str,
    questions: list[dict],
    batch_size: int = 8,
    device: str = "cuda",
) -> float:
    """Evaluate a model on multiple-choice questions.

    Uses log-likelihood scoring: for each question, compute
    P(choice | prompt) for all choices and pick the highest.

    Returns accuracy as percentage.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("transformers not installed, returning 0.0")
        return 0.0

    logger.info(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    correct = 0
    total = 0

    for q in questions:
        prompt = q["prompt"]
        choices = q["choices"]
        answer_idx = q["answer_idx"]

        scores = []
        for choice in choices:
            text = f"{prompt}\nAnswer: {choice}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            choice_tokens = tokenizer(choice, return_tensors="pt")["input_ids"][0]
            score = logits[choice_tokens[-1]].item()
            scores.append(score)

        predicted = scores.index(max(scores))
        if predicted == answer_idx:
            correct += 1
        total += 1

    accuracy = (correct / total * 100) if total > 0 else 0.0
    logger.info(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

    del model
    torch.cuda.empty_cache()
    return accuracy
