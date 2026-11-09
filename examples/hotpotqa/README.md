# HotpotQA GEPA ordering experiments

`main.py` runs the five conditions from `Take home exercise - Guysu Journal.md`:

- `baseline`: fixed feedback/Pareto split and minibatch order.
- `reverse_order`: reverse the feedback minibatch order.
- `local_swap`: swap adjacent feedback minibatches.
- `swap_split`: replace the complete feedback split with an equal-sized slice of Pareto data.
- `partial_swap`: exchange every other aligned minibatch between feedback and Pareto.

Defaults are minibatch size 6, 600 feedback examples, 100 Pareto examples, 1,000 held-out
evaluation examples, and 2,200 metric calls. CLI flags `--reflection-minibatch-size`,
`--feedback-size`, `--pareto-size`, `--evaluation-size`, and `--max-evals` control them.
The first run creates a deterministic manifest whose filename includes all split sizes.
Feedback and Pareto examples are sampled from local training files; held-out examples come
from local validation. Feedback has 100 minibatches. GEPA evaluates all Pareto examples
together; the loader requires an even Pareto size and an even feedback minibatch count.

Because the split sizes are unequal, `swap_split` swaps all 100 Pareto examples with the first
100 feedback examples and leaves the remaining 500 feedback examples in place. `partial_swap`
exchanges alternating 6-example groups over the aligned portion; any Pareto remainder stays in
the Pareto set.

From the `gepa` directory:

```bash
export OPENROUTER_API_KEY=...
UV_CACHE_DIR=/tmp/odem-uv-cache uv run python examples/hotpotqa/main.py \
  --experiment baseline
```

Run the other conditions by changing `--experiment`. Use `--dry-run` to validate loading and
ordering without model calls. Results are isolated under
`experiments/hotpotqa-gepa-experiments/feedback-600_pareto-100_eval-1000/<experiment>/seed-0/run-<timestamp>/`.
Each fresh execution gets a local-time timestamp with microsecond and UTC-offset precision plus
its process ID, so parallel or repeated runs do not conflict. Pass `--resume` to continue the
newest timestamped run for that experiment. Older checkpoints stored directly under `seed-0/`
remain resumable.

The task model defaults to `openrouter/openai/gpt-oss-20b` and the reflection model to
`openrouter/openai/gpt-4.1-mini`. Override them with `HOTPOTQA_TASK_MODEL` and
`HOTPOTQA_REFLECTION_MODEL` when needed.

Optimization stops on the 2,200 metric-call budget by default. Runtime and reflection-cost
ceilings are disabled unless `--max-runtime-minutes` or `--max-reflection-cost` is supplied.
These optional ceilings can stop optimization before the metric budget is exhausted.
