# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Thin LM abstraction over LiteLLM that handles retries, truncation
warnings, and cross-model compatibility.

Usage::

    from gepa.lm import LM

    lm = LM("openai/gpt-4.1", temperature=0.7, max_tokens=4096)
    response: str = lm("Solve this problem...")

    # Also works with chat messages
    response = lm([{"role": "user", "content": "Hello"}])

The returned callable conforms to the ``LanguageModel`` protocol
(``(str | list[dict]) -> str``) used throughout GEPA.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, cast

logger = logging.getLogger(__name__)


class LMOutput(str):
    """A string completion that retains optional provider termination metadata.

    It remains a ``str`` for complete backwards compatibility with existing LM
    callables and user code. ``strip`` preserves the metadata because signature
    execution normalizes whitespace before parsing.
    """

    finish_reason: str | None

    def __new__(cls, value: str, *, finish_reason: str | None = None):
        instance = super().__new__(cls, value)
        instance.finish_reason = finish_reason
        return instance

    def strip(self, chars: str | None = None) -> LMOutput:
        return type(self)(super().strip(chars), finish_reason=self.finish_reason)


class LM:
    """A lightweight language model wrapper over LiteLLM.

    Handles:

    - **Retries** with exponential backoff via LiteLLM's ``num_retries``.
    - **Truncation detection** — logs a warning when ``finish_reason='length'``.
    - **drop_params=True** so unsupported params are silently ignored
      (with a warning logged for transparency).

    Conforms to the :class:`~gepa.proposer.reflective_mutation.base.LanguageModel`
    protocol, so it can be used anywhere GEPA expects a ``LanguageModel``.

    Args:
        model: LiteLLM model identifier, e.g. ``"openai/gpt-4.1"`` or ``"anthropic/claude-sonnet-4-6"``.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        num_retries: Number of retries on transient failures (default 3).
        show_batch_progress: Show per-request progress for batched completions.
        batch_progress_desc: Label used by the batch progress bar.
        input_cost_per_million: Fallback USD price per million input tokens.
        output_cost_per_million: Fallback USD price per million output tokens.
        **kwargs: Extra keyword arguments forwarded to ``litellm.completion``
            (e.g. ``top_p``, ``stop``, ``api_key``, ``api_base``).
    """

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_retries: int = 3,
        show_batch_progress: bool = False,
        batch_progress_desc: str = "LM batch",
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.num_retries = num_retries
        self.show_batch_progress = show_batch_progress
        self.batch_progress_desc = batch_progress_desc
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self._total_cost: float = 0.0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0
        self._cost_by_source: dict[str, float] = {"provider": 0.0, "litellm": 0.0, "estimated": 0.0}
        self._cost_calls_by_source: dict[str, int] = {"provider": 0, "litellm": 0, "estimated": 0, "unpriced": 0}
        self._cost_lock = threading.Lock()

        self.completion_kwargs: dict[str, Any] = {
            **({"temperature": temperature} if temperature is not None else {}),
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            **kwargs,
        }

    @property
    def total_cost(self) -> float:
        """Cumulative USD cost of all calls made through this LM instance."""
        return self._total_cost

    @property
    def total_tokens_in(self) -> int:
        """Cumulative input (prompt) tokens across all calls."""
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        """Cumulative output (completion) tokens across all calls."""
        return self._total_tokens_out

    @property
    def cost_breakdown(self) -> dict[str, Any]:
        """Cost totals and response counts grouped by how cost was obtained."""
        with self._cost_lock:
            return {
                "usd_by_source": dict(self._cost_by_source),
                "calls_by_source": dict(self._cost_calls_by_source),
                "contains_estimates": self._cost_calls_by_source["estimated"] > 0,
                "contains_unpriced_calls": self._cost_calls_by_source["unpriced"] > 0,
            }

    @staticmethod
    def _numeric_attr(obj: Any, name: str) -> float | None:
        value = getattr(obj, name, None) if obj is not None else None
        return float(value) if isinstance(value, int | float) else None

    def _extract_response_cost(self, completion: Any, litellm: Any, tokens_in: int, tokens_out: int) -> tuple[float, str]:
        """Prefer provider-reported cost, then LiteLLM, then configured token prices."""
        usage = getattr(completion, "usage", None)
        for field in ("cost", "provider_reported_cost_usd"):
            provider_cost = self._numeric_attr(usage, field)
            if provider_cost is not None and provider_cost > 0:
                return provider_cost, "provider"

        hidden = getattr(completion, "_hidden_params", None)
        if isinstance(hidden, dict):
            hidden_cost = hidden.get("response_cost")
            if isinstance(hidden_cost, int | float) and hidden_cost > 0:
                return float(hidden_cost), "provider"

        try:
            computed_cost = litellm.completion_cost(completion_response=completion)
            if isinstance(computed_cost, int | float) and computed_cost > 0:
                return float(computed_cost), "litellm"
        except Exception:
            pass

        if self.input_cost_per_million is not None and self.output_cost_per_million is not None:
            estimated_cost = (
                tokens_in * self.input_cost_per_million + tokens_out * self.output_cost_per_million
            ) / 1_000_000
            return estimated_cost, "estimated"
        return 0.0, "unpriced"

    def _record_completion_usage(self, completion: Any, litellm: Any) -> None:
        usage = getattr(completion, "usage", None)
        raw_tokens_in = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
        raw_tokens_out = getattr(usage, "completion_tokens", 0) if usage is not None else 0
        tokens_in = int(raw_tokens_in) if isinstance(raw_tokens_in, int) else 0
        tokens_out = int(raw_tokens_out) if isinstance(raw_tokens_out, int) else 0
        cost, source = self._extract_response_cost(completion, litellm, tokens_in, tokens_out)

        with self._cost_lock:
            self._total_cost += cost
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out
            self._cost_calls_by_source[source] += 1
            if source != "unpriced":
                self._cost_by_source[source] += cost

    def _check_truncation(self, choices: list[Any]) -> None:
        if any(getattr(c, "finish_reason", None) == "length" for c in choices):
            max_tok = self.completion_kwargs.get("max_tokens") or self.completion_kwargs.get("max_completion_tokens")
            logger.warning(
                f"LM response was truncated (finish_reason='length', max_tokens={max_tok}). "
                "Consider increasing max_tokens for better results."
            )

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        import litellm

        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        completion = litellm.completion(
            model=self.model,
            messages=messages,
            num_retries=self.num_retries,
            drop_params=True,
            **self.completion_kwargs,
        )

        # Non-streaming calls always return ModelResponse (not CustomStreamWrapper)
        self._check_truncation(completion.choices)  # type: ignore[union-attr]

        self._record_completion_usage(completion, litellm)

        choice = completion.choices[0]  # type: ignore[union-attr]
        return LMOutput(choice.message.content or "", finish_reason=getattr(choice, "finish_reason", None))

    def batch_complete(
        self, messages_list: list[list[dict[str, Any]]], max_workers: int = 10, **kwargs: Any
    ) -> list[str]:
        """Run multiple completions in parallel using ``litellm.batch_completion``.

        Args:
            messages_list: List of message lists, one per request.
            max_workers: Maximum concurrent requests.
            **kwargs: Extra keyword arguments forwarded to ``litellm.batch_completion``
                (e.g. ``timeout``, ``api_base``).  These override any matching keys
                set during ``__init__``.

        Returns:
            List of response strings, one per input.
        """
        import litellm

        merged = {**self.completion_kwargs, **kwargs}
        if self.show_batch_progress and messages_list:
            # LiteLLM's batch_completion waits for the entire executor before
            # returning, so callers cannot observe incremental progress. Submit
            # the same calls here and consume futures as they finish while
            # retaining the input order expected by GEPA adapters.
            from tqdm.auto import tqdm

            responses: list[Any] = [None] * len(messages_list)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(
                        litellm.completion,
                        model=self.model,
                        messages=messages,
                        num_retries=self.num_retries,
                        drop_params=True,
                        **merged,
                    ): index
                    for index, messages in enumerate(messages_list)
                }
                completed = as_completed(future_to_index)
                with tqdm(
                    completed,
                    total=len(future_to_index),
                    desc=self.batch_progress_desc,
                    unit="example",
                    dynamic_ncols=True,
                    leave=True,
                ) as progress:
                    for future in progress:
                        responses[future_to_index[future]] = future.result()
        else:
            responses = litellm.batch_completion(
                model=self.model,
                messages=messages_list,
                max_workers=max_workers,
                num_retries=self.num_retries,
                drop_params=True,
                **merged,
            )

        results: list[str] = []
        for resp in responses:
            self._check_truncation(resp.choices)
            choice = resp.choices[0]
            results.append(
                LMOutput(choice.message.content or "", finish_reason=getattr(choice, "finish_reason", None)).strip()
            )
            self._record_completion_usage(resp, litellm)

        return results

    def __repr__(self) -> str:
        params = [f"model={self.model!r}"]
        for k, v in self.completion_kwargs.items():
            params.append(f"{k}={v!r}")
        return f"LM({', '.join(params)})"


class TrackingLM:
    """Wraps an arbitrary callable to track estimated token usage.

    For callables that don't go through LiteLLM, we can't get real token
    counts or costs.  This wrapper estimates tokens from string lengths
    (~4 chars/token) and reports ``total_cost = 0.0``.

    This ensures that *every* reflection LM — whether an ``LM`` instance
    or a plain callable — exposes ``total_cost``, ``total_tokens_in``,
    and ``total_tokens_out``.
    """

    _CHARS_PER_TOKEN = 4

    def __init__(self, fn: Any):
        self._fn = fn
        self._total_cost: float = 0.0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens_in(self) -> int:
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return self._total_tokens_out

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // self._CHARS_PER_TOKEN)

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            self._total_tokens_in += self._estimate_tokens(prompt)
        else:
            self._total_tokens_in += self._estimate_tokens(str(prompt))

        result = self._fn(prompt)

        if isinstance(result, str):
            self._total_tokens_out += self._estimate_tokens(result)

        return result

    def __getattr__(self, name: str):
        # Conditionally expose batch_complete: hasattr(tracking_lm,
        # "batch_complete") must be True exactly when the wrapped callable
        # provides it, so batched reflection (StatelessReflectionLM) is not
        # silently downgraded to the per-task path by this wrapper.
        if name == "batch_complete":
            inner = getattr(self._fn, "batch_complete", None)
            if not callable(inner):
                raise AttributeError(name)

            def tracked_batch_complete(messages_list):
                for messages in messages_list:
                    self._total_tokens_in += self._estimate_tokens(str(messages))
                results = list(cast(Any, inner)(messages_list))
                for result in results:
                    if isinstance(result, str):
                        self._total_tokens_out += self._estimate_tokens(result)
                return results

            return tracked_batch_complete
        raise AttributeError(name)

    def __repr__(self) -> str:
        return f"TrackingLM({self._fn!r})"
