"""GEPA optimization and HotpotQA ordering/split experiment harness.

From the ``gepa`` project directory, run::

    uv run python examples/hotpotqa/main.py

The five experiment modes are ``baseline``, ``reverse_order``, ``local_swap``,
``swap_split``, and ``partial_swap``. The training files supply the feedback
and Pareto sets; the labeled validation file supplies held-out evaluation.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from utils import (
    DEFAULT_FEEDBACK_EXAMPLES,
    DEFAULT_HELDOUT_EVAL_EXAMPLES,
    DEFAULT_PARETO_EXAMPLES,
    EXPERIMENT_NAMES,
    EXPERIMENT_ROOT,
    evaluate_on_dataset,
    hotpotqa_metric,
    load_experiment_dataset,
)

import gepa
from gepa.lm import LM

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "experiments"
SEED = 0
PAPER_TASK_MODEL = "openrouter/openai/gpt-oss-20b"
PAPER_REFLECTION_MODEL = "openrouter/openai/gpt-4.1-mini"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
MODEL_PRICES_PER_MILLION_TOKENS = {
    "openrouter/openai/gpt-oss-20b": {"input": 0.018, "output": 0.09},
    "openrouter/openai/gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}
SEED_CANDIDATE = {
    "system_prompt": (
        "Answer the question using only the supplied passages. Return only valid JSON with this shape: "
        '{"answer": "short answer", "sp": [["exact passage title", 0]]}. '
        "For each supporting fact, include the exact passage title and its zero-based sentence index. "
        "Include the minimal set of sentences that supports the answer. For yes-or-no questions, answer yes or no."
    )
}


class CompactLogger:
    """Keep GEPA progress messages while suppressing prompt and score dumps."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()

    def log(self, message: str) -> None:
        if len(message) > 300:
            return
        if any(
            marker in message
            for marker in (
                "Full prompt for reflection",
                "Proposed new text for",
                "Individual valset scores",
                "New valset pareto front scores",
                "Updated valset pareto front programs",
            )
        ):
            return
        elapsed = time.monotonic() - self.started_at
        print(f"[{elapsed:8.1f}s] {message}", flush=True)


def evaluate(example: dict[str, Any], response: str):
    """Score one model response with HotpotQA answer F1 and reflective feedback."""
    return hotpotqa_metric(example, response)


def make_run_dir(experiment: str, train_size: int, pareto_size: int, eval_size: int, resume: bool) -> Path:
    """Create a collision-free run path or select the newest run to resume."""
    seed_dir = (
        EXPERIMENT_ROOT
        / f"feedback-{train_size}_pareto-{pareto_size}_eval-{eval_size}"
        / experiment
        / f"seed-{SEED}"
    )
    if resume:
        timestamped_runs = [path for path in seed_dir.glob("run-*") if path.is_dir()]
        if timestamped_runs:
            return max(timestamped_runs, key=lambda path: path.stat().st_mtime_ns)
        if (seed_dir / "gepa_state.bin").exists():
            return seed_dir
        raise SystemExit(f"No previous run found to resume under {seed_dir}")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f%z")
    return seed_dir / f"run-{timestamp}-pid{os.getpid()}"


def main() -> None:
    program_started_at = time.monotonic()

    def exit_on_interrupt(signum: int, _frame: Any) -> None:
        # ThreadPoolExecutor normally waits for active HTTP calls during
        # interpreter shutdown, which can make Ctrl+C appear ineffective.
        # The user explicitly requested cancellation, so leave immediately;
        # already-written GEPA checkpoints remain available for --resume.
        print("\nInterrupted; stopping the HotpotQA run immediately.", flush=True)
        os._exit(128 + signum)

    signal.signal(signal.SIGINT, exit_on_interrupt)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=EXPERIMENT_NAMES,
        default="baseline",
        help="Ordering/split condition to run (default: baseline).",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=2_200,
        help="Task-model metric-call cap (default: 2200).",
    )
    parser.add_argument("--reflection-minibatch-size", type=int, default=6, help="Feedback examples per proposal.")
    parser.add_argument("--feedback-size", type=int, default=DEFAULT_FEEDBACK_EXAMPLES, help="Feedback examples.")
    parser.add_argument("--pareto-size", type=int, default=DEFAULT_PARETO_EXAMPLES, help="Pareto evaluation examples.")
    parser.add_argument(
        "--evaluation-size", type=int, default=DEFAULT_HELDOUT_EVAL_EXAMPLES, help="Held-out test examples."
    )
    parser.add_argument(
        "--max-reflection-cost",
        type=float,
        default=None,
        help="Optional GEPA reflection-LM cost ceiling in dollars (default: disabled).",
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=None,
        help="Optional GEPA optimization runtime ceiling in minutes (default: disabled).",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Also show GEPA's coarse metric-budget progress bar.",
    )
    parser.add_argument(
        "--no-task-progress-bar",
        action="store_true",
        help="Disable the per-example task-model progress bar.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run directory for this experiment.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load and sample data without calling any LMs.")
    args = parser.parse_args()

    if args.max_evals is not None and args.max_evals <= 0:
        parser.error("--max-evals must be positive when provided")
    if min(args.reflection_minibatch_size, args.feedback_size, args.pareto_size, args.evaluation_size) <= 0:
        parser.error("dataset sizes and --reflection-minibatch-size must be positive")
    if args.max_reflection_cost is not None and args.max_reflection_cost <= 0:
        parser.error("--max-reflection-cost must be positive when provided")
    if args.max_runtime_minutes is not None and args.max_runtime_minutes <= 0:
        parser.error("--max-runtime-minutes must be positive")

    task_model_name = os.environ.get("HOTPOTQA_TASK_MODEL", PAPER_TASK_MODEL)
    reflection_model_name = os.environ.get("HOTPOTQA_REFLECTION_MODEL", PAPER_REFLECTION_MODEL)
    print("Initializing task and reflection models...", flush=True)

    def make_task_lm(model_name: str) -> LM:
        provider_kwargs = {}
        provider_kwargs["api_base"] = os.environ.get("HOTPOTQA_TASK_API_BASE", OPENROUTER_API_BASE)
        provider_kwargs["extra_body"] = {"usage": {"include": True}}
        if task_api_key := os.environ.get("HOTPOTQA_TASK_API_KEY", os.environ.get("OPENROUTER_API_KEY")):
            provider_kwargs["api_key"] = task_api_key
        prices = MODEL_PRICES_PER_MILLION_TOKENS.get(model_name, {})
        return LM(
            model_name,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            timeout=60,
            num_retries=1,
            show_batch_progress=not args.no_task_progress_bar,
            batch_progress_desc="HotpotQA task evaluation",
            input_cost_per_million=prices.get("input"),
            output_cost_per_million=prices.get("output"),
            **provider_kwargs,
        )

    def make_paper_reflection_lm(model_name: str) -> LM:
        provider_kwargs = {}
        provider_kwargs["api_base"] = os.environ.get("HOTPOTQA_REFLECTION_API_BASE", OPENROUTER_API_BASE)
        provider_kwargs["extra_body"] = {"usage": {"include": True}}
        if reflection_api_key := os.environ.get(
            "HOTPOTQA_REFLECTION_API_KEY", os.environ.get("OPENROUTER_API_KEY")
        ):
            provider_kwargs["api_key"] = reflection_api_key
        prices = MODEL_PRICES_PER_MILLION_TOKENS.get(model_name, {})
        return LM(
            model_name,
            temperature=1.0,
            timeout=60,
            num_retries=1,
            input_cost_per_million=prices.get("input"),
            output_cost_per_million=prices.get("output"),
            **provider_kwargs,
        )

    task_lm = make_task_lm(task_model_name)
    reflection_lm = make_paper_reflection_lm(reflection_model_name)

    print("Loading HotpotQA sources and constructing the experiment split...", flush=True)
    trainset, valset, evalset, batch_sampler, dataset_metadata = load_experiment_dataset(
        args.experiment,
        args.reflection_minibatch_size,
        feedback_size=args.feedback_size,
        pareto_size=args.pareto_size,
        eval_size=args.evaluation_size,
        seed=SEED,
    )
    max_metric_calls = args.max_evals
    print(
        f"GEPA configuration: train={len(trainset)}, val={len(valset)}, "
        f"held_out_eval={len(evalset)}, max_metric_calls={max_metric_calls}, "
        f"experiment={args.experiment}, minibatch={args.reflection_minibatch_size}, "
        f"max_reflection_cost={args.max_reflection_cost}"
    )
    print(f"Task model: {task_model_name}; reflection model: {reflection_model_name}")
    print(f"Dataset manifest: {dataset_metadata['manifest_path']}")
    print(
        f"Loaded {len(trainset)} training, {len(valset)} validation, "
        f"and {len(evalset)} held-out evaluation examples."
    )
    if args.dry_run:
        print("Dry run: no model calls made.")
        return

    run_dir = make_run_dir(
        args.experiment,
        train_size=len(trainset),
        pareto_size=len(valset),
        eval_size=len(evalset),
        resume=args.resume,
    )
    if args.resume:
        print(f"Resuming newest run: {run_dir}", flush=True)
    print(f"Starting GEPA optimization; run directory: {run_dir}", flush=True)
    print(
        "The initial seed evaluation covers the full Pareto set; subsequent candidate evaluations "
        "may take time and use the task-model budget.",
        flush=True,
    )
    stop_callbacks = None
    if args.max_runtime_minutes is not None:
        remaining_runtime = max(
            1.0, args.max_runtime_minutes * 60.0 - (time.monotonic() - program_started_at)
        )
        stop_callbacks = gepa.TimeoutStopCondition(timeout_seconds=remaining_runtime)
        print(f"GEPA optimization timeout: {remaining_runtime / 60.0:.2f} minutes", flush=True)
    else:
        print("GEPA optimization timeout: disabled; metric-call budget controls stopping", flush=True)
    result = gepa.optimize(
        seed_candidate=SEED_CANDIDATE,
        trainset=trainset,
        valset=valset,
        task_lm=task_lm,
        evaluator=evaluate,
        reflection_lm=reflection_lm,
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        use_merge=False,
        batch_sampler=batch_sampler,
        max_metric_calls=max_metric_calls,
        max_reflection_cost=args.max_reflection_cost,
        stop_callbacks=stop_callbacks,
        run_dir=str(run_dir),
        logger=CompactLogger(),
        write_agent_state=True,
        seed=SEED,
        display_progress_bar=args.progress_bar,
    )

    best_prompt = result.best_candidate["system_prompt"]
    best_prompt_path = run_dir / "best_prompt.txt"
    best_prompt_path.write_text(best_prompt + "\n", encoding="utf-8")
    best_val_f1 = result.val_aggregate_scores[result.best_idx]
    optimization_task_cost = task_lm.total_cost
    optimization_reflection_cost = reflection_lm.total_cost
    print(f"Validation answer F1: {best_val_f1:.4f}")
    print(f"Metric calls: {result.total_metric_calls}")
    print(f"Best prompt saved to: {best_prompt_path}")

    evaluation_dir = run_dir / "held_out_evaluation"
    print(f"\nEvaluating the seed prompt on the held-out set ({len(evalset):,} examples)...", flush=True)
    cost_before_baseline_eval = task_lm.total_cost
    baseline_metrics = evaluate_on_dataset(
        SEED_CANDIDATE["system_prompt"], evalset, task_lm, evaluation_dir, "baseline"
    )
    baseline_eval_cost = task_lm.total_cost - cost_before_baseline_eval
    print(f"\nEvaluating the optimized prompt on the held-out set ({len(evalset):,} examples)...", flush=True)
    cost_before_optimized_eval = task_lm.total_cost
    optimized_metrics = evaluate_on_dataset(best_prompt, evalset, task_lm, evaluation_dir, "optimized")
    optimized_eval_cost = task_lm.total_cost - cost_before_optimized_eval

    for label, metrics in (("Baseline", baseline_metrics), ("Optimized", optimized_metrics)):
        print(
            f"{label} held-out: answer EM={metrics['em']:.4f}, answer F1={metrics['f1']:.4f}, "
            f"support EM={metrics['sp_em']:.4f}, support F1={metrics['sp_f1']:.4f}, "
            f"joint EM={metrics['joint_em']:.4f}, joint F1={metrics['joint_f1']:.4f}"
        )

    total_cost = task_lm.total_cost + reflection_lm.total_cost
    print("\nCost summary (USD):")
    print(f"  GEPA task-model evaluations:       ${optimization_task_cost:.6f}")
    print(f"  GEPA reflection-model calls:       ${optimization_reflection_cost:.6f}")
    print(f"  Held-out seed evaluation:          ${baseline_eval_cost:.6f}")
    print(f"  Held-out optimized evaluation:     ${optimized_eval_cost:.6f}")
    print(f"  Total:                             ${total_cost:.6f}")
    print(
        f"  Tokens: task={task_lm.total_tokens_in:,} in/{task_lm.total_tokens_out:,} out; "
        f"reflection={reflection_lm.total_tokens_in:,} in/{reflection_lm.total_tokens_out:,} out"
    )
    print(f"  Task cost sources: {task_lm.cost_breakdown['calls_by_source']}")
    print(f"  Reflection cost sources: {reflection_lm.cost_breakdown['calls_by_source']}")

    summary = {
        "experiment": args.experiment,
        "seed": SEED,
        "run_dir": str(run_dir),
        "manifest_path": dataset_metadata["manifest_path"],
        "task_model": task_model_name,
        "reflection_model": reflection_model_name,
        "train_examples": len(trainset),
        "pareto_examples": len(valset),
        "evaluation_examples": len(evalset),
        "reflection_minibatch_size": args.reflection_minibatch_size,
        "feedback_batch_count": dataset_metadata["feedback_batch_count"],
        "max_metric_calls": max_metric_calls,
        "metric_calls": result.total_metric_calls,
        "max_reflection_cost": args.max_reflection_cost,
        "max_runtime_minutes": args.max_runtime_minutes,
        "task_cost": task_lm.total_cost,
        "reflection_cost": reflection_lm.total_cost,
        "optimization_task_cost": optimization_task_cost,
        "optimization_reflection_cost": optimization_reflection_cost,
        "baseline_heldout_cost": baseline_eval_cost,
        "optimized_heldout_cost": optimized_eval_cost,
        "total_cost": total_cost,
        "task_tokens_in": task_lm.total_tokens_in,
        "task_tokens_out": task_lm.total_tokens_out,
        "reflection_tokens_in": reflection_lm.total_tokens_in,
        "reflection_tokens_out": reflection_lm.total_tokens_out,
        "task_cost_breakdown": task_lm.cost_breakdown,
        "reflection_cost_breakdown": reflection_lm.cost_breakdown,
        "best_validation_answer_f1": best_val_f1,
        "baseline_metrics": baseline_metrics,
        "optimized_metrics": optimized_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary saved to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
