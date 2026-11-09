"""HotpotQA scoring and held-out evaluation helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from hotpot_evaluate_v1 import eval as official_hotpot_eval
from hotpot_evaluate_v1 import f1_score as hotpot_f1_score

from gepa.adapters.default_adapter.default_adapter import DefaultAdapter, EvaluationResult
from gepa.lm import LM

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "experiments"
SEED = 0
DEFAULT_FEEDBACK_EXAMPLES = 600
DEFAULT_PARETO_EXAMPLES = 100
DEFAULT_HELDOUT_EVAL_EXAMPLES = 1_000
EXPERIMENT_NAMES = ("baseline", "reverse_order", "local_swap", "swap_split", "partial_swap")
EXPERIMENT_ROOT = EXPERIMENTS / "hotpotqa-gepa-experiments"


def manifest_path(feedback_size: int, pareto_size: int, eval_size: int, seed: int) -> Path:
    return EXPERIMENT_ROOT / (
        f"dataset_manifest_seed-{seed}_feedback-{feedback_size}_pareto-{pareto_size}_eval-{eval_size}.json"
    )


def reservoir_sample(items, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    """Take a reproducible uniform sample without retaining the full dataset."""
    sample: list[dict[str, Any]] = []
    count = 0
    for item in items:
        if count < limit:
            sample.append(item)
        else:
            replacement = rng.randrange(count + 1)
            if replacement < limit:
                sample[replacement] = item
        count += 1
    if count < limit:
        raise ValueError(f"Requested {limit} examples, but found only {count}.")
    return sample


def _read_jsonl_files(files: list[Path]):
    for file in files:
        print(f"Loading {file.name}...", flush=True)
        file_count = 0
        with file.open(encoding="utf-8") as stream:
            for line in stream:
                file_count += 1
                if file_count % 10_000 == 0:
                    print(f"  {file.name}: {file_count:,} rows read", flush=True)
                yield json.loads(line)
        print(f"Finished {file.name}: {file_count:,} rows", flush=True)


def _read_parquet_files(files: list[Path]):
    for file in files:
        parquet_file = pq.ParquetFile(file)
        columns = ["id", "question", "answer", "context", "supporting_facts"]
        for batch in parquet_file.iter_batches(columns=columns, batch_size=128):
            yield from batch.to_pylist()


class FixedBatchSampler:
    """Walk a preordered training list in fixed, non-overlapping minibatches."""

    def __init__(self, minibatch_size: int):
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        self.minibatch_size = minibatch_size
        self._iteration: int | None = None
        self._calls = 0

    def next_minibatch_ids(self, loader, state) -> list[int]:
        if state.i == self._iteration:
            self._calls += 1
        else:
            self._iteration = state.i
            self._calls = 0
        batch_count = len(loader) // self.minibatch_size
        if batch_count == 0 or len(loader) % self.minibatch_size:
            raise ValueError("FixedBatchSampler requires a training set divisible by minibatch_size")
        batch_index = (state.i + self._calls) % batch_count
        start = batch_index * self.minibatch_size
        return list(range(start, start + self.minibatch_size))


def make_gepa_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        context = row["context"]
        passages = [
            f"[{title}]\n" + "\n".join(f"[{sentence_id}] {sentence}" for sentence_id, sentence in enumerate(sentences))
            for title, sentences in zip(context["title"], context["sentences"], strict=True)
        ]
        context_text = "\n\n".join(passages)
        examples.append(
            {
                "input": f"Passages:\n{context_text}\n\nQuestion: {row['question']}",
                "additional_context": {},
                "answer": row["answer"],
                "id": row["id"],
                "supporting_facts": [
                    [title, sentence_id]
                    for title, sentence_id in zip(
                        row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"], strict=True
                    )
                ],
            }
        )
    return examples


def load_hotpot_dataset(
    train_limit: int | None = None,
    val_limit: int | None = None,
    eval_limit: int | None = None,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split training rows into GEPA train/Pareto sets and use validation for evaluation."""
    train_jsonl = sorted(EXPERIMENTS.glob("train-*.jsonl"))
    train_parquet = sorted(EXPERIMENTS.glob("train-*.parquet"))
    train_files = train_jsonl or train_parquet
    if not train_files:
        raise FileNotFoundError(f"No training JSONL or Parquet files found in {EXPERIMENTS}")
    all_train_rows = list(_read_jsonl_files(train_files) if train_jsonl else _read_parquet_files(train_files))
    if train_limit is None and val_limit is None:
        usable_count = min(DEFAULT_OPTIMIZATION_TRAIN_EXAMPLES, len(all_train_rows))
        selected_train_rows = reservoir_sample(all_train_rows, usable_count, random.Random(seed))
        train_limit = DEFAULT_FEEDBACK_EXAMPLES
        val_limit = DEFAULT_PARETO_EXAMPLES
    else:
        if train_limit is None or val_limit is None:
            raise ValueError("train_limit and val_limit must both be set or both be omitted")
        selected_train_rows = reservoir_sample(
            all_train_rows, train_limit + val_limit, random.Random(seed)
        )
    train_rows, pareto_rows = selected_train_rows[:train_limit], selected_train_rows[train_limit:]

    val_jsonl = EXPERIMENTS / "validation-00000-of-00001.jsonl"
    val_parquet = EXPERIMENTS / "validation-00000-of-00001.parquet"
    if val_jsonl.exists():
        validation_rows = _read_jsonl_files([val_jsonl])
    elif val_parquet.exists():
        validation_rows = _read_parquet_files([val_parquet])
    else:
        raise FileNotFoundError(f"No validation JSONL or Parquet file found in {EXPERIMENTS}")

    all_eval_rows = list(validation_rows)
    eval_rows = all_eval_rows if eval_limit is None else reservoir_sample(
        all_eval_rows, eval_limit, random.Random(seed + 1)
    )
    all_ids = [str(row["id"]) for row in train_rows + pareto_rows + eval_rows]
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("Train, validation, and evaluation sets must have disjoint IDs.")
    return make_gepa_examples(train_rows), make_gepa_examples(pareto_rows), make_gepa_examples(eval_rows)


def _load_source_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_jsonl = sorted(EXPERIMENTS.glob("train-*.jsonl"))
    train_parquet = sorted(EXPERIMENTS.glob("train-*.parquet"))
    train_files = train_jsonl or train_parquet
    if not train_files:
        raise FileNotFoundError(f"No training JSONL or Parquet files found in {EXPERIMENTS}")
    train_rows = list(_read_jsonl_files(train_files) if train_jsonl else _read_parquet_files(train_files))

    val_jsonl = EXPERIMENTS / "validation-00000-of-00001.jsonl"
    val_parquet = EXPERIMENTS / "validation-00000-of-00001.parquet"
    if val_jsonl.exists():
        eval_rows = list(_read_jsonl_files([val_jsonl]))
    elif val_parquet.exists():
        eval_rows = list(_read_parquet_files([val_parquet]))
    else:
        raise FileNotFoundError(f"No validation JSONL or Parquet file found in {EXPERIMENTS}")
    return train_rows, eval_rows


def _chunk_ids(ids: list[str], size: int) -> list[list[str]]:
    if len(ids) % size:
        raise ValueError(f"Expected {len(ids)} IDs to be divisible by minibatch size {size}")
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def _interleave_split(
    feedback_ids: list[str], pareto_ids: list[str], minibatch_size: int
) -> tuple[list[str], list[str]]:
    feedback_batches = _chunk_ids(feedback_ids, minibatch_size)
    paired_size = min(len(feedback_ids), len(pareto_ids))
    paired_size -= paired_size % minibatch_size
    pareto_batches = _chunk_ids(pareto_ids[:paired_size], minibatch_size)
    new_feedback: list[str] = []
    new_pareto: list[str] = []
    paired_batch_count = min(len(feedback_batches), len(pareto_batches))
    for index in range(paired_batch_count):
        feedback_batch = feedback_batches[index]
        pareto_batch = pareto_batches[index]
        if index % 2:
            new_feedback.extend(pareto_batch)
            new_pareto.extend(feedback_batch)
        else:
            new_feedback.extend(feedback_batch)
            new_pareto.extend(pareto_batch)
    for batch in feedback_batches[paired_batch_count:]:
        new_feedback.extend(batch)
    new_pareto.extend(pareto_ids[paired_size:])
    return new_feedback, new_pareto


def _swap_split_membership(feedback_ids: list[str], pareto_ids: list[str]) -> tuple[list[str], list[str]]:
    """Replace the smaller split completely while preserving both split sizes."""
    swap_count = min(len(feedback_ids), len(pareto_ids))
    if len(feedback_ids) <= len(pareto_ids):
        return pareto_ids[:swap_count], feedback_ids[:swap_count] + pareto_ids[swap_count:]
    return pareto_ids[:swap_count] + feedback_ids[swap_count:], feedback_ids[:swap_count]


def _build_experiment_ids(
    feedback_ids: list[str], pareto_ids: list[str], experiment: str, minibatch_size: int
) -> tuple[list[str], list[str]]:
    feedback_batches = _chunk_ids(feedback_ids, minibatch_size)
    if experiment == "baseline":
        return feedback_ids, pareto_ids
    if experiment == "reverse_order":
        return [item for batch in reversed(feedback_batches) for item in batch], pareto_ids
    if experiment == "local_swap":
        swapped: list[list[str]] = []
        for index in range(0, len(feedback_batches), 2):
            pair = feedback_batches[index : index + 2]
            swapped.extend(reversed(pair))
        return [item for batch in swapped for item in batch], pareto_ids
    if experiment == "swap_split":
        return _swap_split_membership(feedback_ids, pareto_ids)
    if experiment == "partial_swap":
        return _interleave_split(feedback_ids, pareto_ids, minibatch_size)
    raise ValueError(f"Unknown experiment {experiment!r}; expected one of {EXPERIMENT_NAMES}")


def load_experiment_dataset(
    experiment: str,
    minibatch_size: int,
    feedback_size: int = DEFAULT_FEEDBACK_EXAMPLES,
    pareto_size: int = DEFAULT_PARETO_EXAMPLES,
    eval_size: int = DEFAULT_HELDOUT_EVAL_EXAMPLES,
    seed: int = SEED,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    FixedBatchSampler,
    dict[str, Any],
]:
    """Load the shared manifest and construct one of the five ordered experiments."""
    if experiment not in EXPERIMENT_NAMES:
        raise ValueError(f"Unknown experiment {experiment!r}; expected one of {EXPERIMENT_NAMES}")
    if minibatch_size <= 0 or feedback_size <= 0 or pareto_size <= 0 or eval_size <= 0:
        raise ValueError("Dataset sizes and minibatch_size must be positive")
    if feedback_size % minibatch_size:
        raise ValueError("The feedback size must be divisible by minibatch_size")
    feedback_batch_count = feedback_size // minibatch_size
    if feedback_batch_count % 2:
        raise ValueError(
            f"The feedback split must contain an even number of minibatches; "
            f"{feedback_size} examples / {minibatch_size} gives {feedback_batch_count} batches"
        )
    if pareto_size % 2:
        raise ValueError("The Pareto set size must be even for the split experiments")

    manifest_file = manifest_path(feedback_size, pareto_size, eval_size, seed)
    train_rows, eval_rows = _load_source_rows()
    train_by_id = {str(row["id"]): row for row in train_rows}
    eval_by_id = {str(row["id"]): row for row in eval_rows}
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("seed") != seed:
            raise ValueError(f"Existing manifest {MANIFEST_PATH} was created with a different seed")
    else:
        selected = reservoir_sample(
            train_rows, feedback_size + pareto_size, random.Random(seed)
        )
        random.Random(seed + 1).shuffle(selected)
        selected_ids = [str(row["id"]) for row in selected]
        manifest = {
            "schema_version": 2,
            "seed": seed,
            "minibatch_size": minibatch_size,
            "selected_training_ids": selected_ids,
            "feedback_ids": selected_ids[:feedback_size],
            "pareto_ids": selected_ids[feedback_size:],
            "evaluation_ids": [
                str(row["id"])
                for row in reservoir_sample(
                    eval_rows, eval_size, random.Random(seed + 2)
                )
            ],
        }
        EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    feedback_ids = [str(item) for item in manifest["feedback_ids"]]
    pareto_ids = [str(item) for item in manifest["pareto_ids"]]
    evaluation_ids = [str(item) for item in manifest["evaluation_ids"]]
    if len(feedback_ids) != feedback_size or len(pareto_ids) != pareto_size:
        raise RuntimeError(
            f"The dataset manifest must contain {feedback_size} feedback and "
            f"{pareto_size} Pareto IDs"
        )
    optimization_ids = feedback_ids + pareto_ids
    if len(set(optimization_ids)) != len(optimization_ids):
        raise RuntimeError("The dataset manifest contains duplicate optimization IDs")
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise RuntimeError("The dataset manifest contains duplicate evaluation IDs")
    if len(evaluation_ids) != eval_size:
        raise RuntimeError(
            f"The dataset manifest must contain exactly {eval_size} held-out evaluation IDs"
        )
    if set(optimization_ids) & set(evaluation_ids):
        raise RuntimeError("Optimization and held-out evaluation IDs must be disjoint")
    missing = [item for item in feedback_ids + pareto_ids if item not in train_by_id]
    missing += [item for item in evaluation_ids if item not in eval_by_id]
    if missing:
        raise RuntimeError(f"Manifest references {len(missing)} IDs missing from local datasets")

    experiment_feedback_ids, experiment_pareto_ids = _build_experiment_ids(
        feedback_ids, pareto_ids, experiment, minibatch_size
    )
    trainset = make_gepa_examples([train_by_id[item] for item in experiment_feedback_ids])
    valset = make_gepa_examples([train_by_id[item] for item in experiment_pareto_ids])
    evalset = make_gepa_examples([eval_by_id[item] for item in evaluation_ids])
    metadata = {
        "manifest_path": str(manifest_file),
        "experiment": experiment,
        "feedback_ids": experiment_feedback_ids,
        "pareto_ids": experiment_pareto_ids,
        "evaluation_ids": evaluation_ids,
        "feedback_batch_count": len(trainset) // minibatch_size,
    }
    return trainset, valset, evalset, FixedBatchSampler(minibatch_size), metadata


def parse_prediction(response: str) -> dict[str, Any]:
    """Read the answer and supporting facts from the model's JSON response."""
    text = response.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()

    start = text.find("{")
    if start >= 0:
        try:
            payload, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(payload, dict):
                answer = payload.get("answer", "")
                raw_facts = payload.get("sp", [])
                facts = []
                if isinstance(raw_facts, list):
                    for fact in raw_facts:
                        if isinstance(fact, (list, tuple)) and len(fact) == 2:
                            try:
                                facts.append([str(fact[0]), int(fact[1])])
                            except (TypeError, ValueError):
                                continue
                return {"answer": str(answer), "sp": facts}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Keep a usable answer score if the model did not follow the JSON format.
    return {"answer": text, "sp": []}


def hotpotqa_metric(data: dict[str, Any], response: str) -> EvaluationResult:
    """Use the official HotpotQA answer F1 as GEPA's optimization metric."""
    prediction = parse_prediction(response)
    answer_f1, _, _ = hotpot_f1_score(prediction["answer"], data["answer"])

    predicted_facts = {tuple(fact) for fact in prediction["sp"]}
    gold_facts = {tuple(fact) for fact in data["supporting_facts"]}
    missing_facts = sorted(gold_facts - predicted_facts)
    extra_facts = sorted(predicted_facts - gold_facts)
    feedback = (
        f"Answer F1: {answer_f1:.3f}. Gold answer: {data['answer']!r}. "
        f"Predicted answer: {prediction['answer']!r}. "
        f"Missing supporting facts: {missing_facts[:5]}. "
        f"Extra supporting facts: {extra_facts[:5]}. "
        "Return JSON with an answer string and supporting facts as [title, zero_based_sentence_id] pairs."
    )
    return EvaluationResult(score=float(answer_f1), feedback=feedback)


def evaluate_on_dataset(
    prompt: str,
    dataset: list[dict[str, Any]],
    task_model: str | LM,
    output_dir: Path,
    prediction_name: str,
) -> dict[str, float]:
    """Run a prompt on held-out examples and score with hotpot_evaluate_v1.py."""
    adapter = DefaultAdapter(model=task_model, evaluator=hotpotqa_metric, max_litellm_workers=10)
    evaluated = adapter.evaluate(dataset, {"system_prompt": prompt})

    predictions: dict[str, Any] = {"answer": {}, "sp": {}}
    gold = []
    for example, output in zip(dataset, evaluated.outputs, strict=True):
        example_id = str(example["id"])
        prediction = parse_prediction(output["full_assistant_response"])
        predictions["answer"][example_id] = prediction["answer"]
        predictions["sp"][example_id] = prediction["sp"]
        gold.append(
            {
                "_id": example_id,
                "answer": example["answer"],
                "supporting_facts": example["supporting_facts"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{prediction_name}_predictions.json"
    gold_path = output_dir / "gold.json"
    prediction_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    gold_path.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = official_hotpot_eval(str(prediction_path), str(gold_path))
    return metrics
