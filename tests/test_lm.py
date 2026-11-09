# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import uuid
from unittest.mock import MagicMock, patch

import httpx
import litellm
import openai
import pytest

from gepa.lm import LM, LMOutput


class TestLMInit:
    """Test LM constructor parameter handling."""

    def test_defaults(self):
        lm = LM("openai/gpt-4.1")
        assert "temperature" not in lm.completion_kwargs
        assert "max_tokens" not in lm.completion_kwargs

    def test_custom_params(self):
        lm = LM("openai/gpt-4.1", temperature=0.5, max_tokens=4096)
        assert lm.completion_kwargs["temperature"] == 0.5
        assert lm.completion_kwargs["max_tokens"] == 4096

    def test_extra_kwargs_forwarded(self):
        lm = LM("openai/gpt-4.1", top_p=0.9, stop=["\n"])
        assert lm.completion_kwargs["top_p"] == 0.9
        assert lm.completion_kwargs["stop"] == ["\n"]

    def test_reasoning_model_no_special_treatment(self):
        """Reasoning models should NOT get special parameter handling."""
        lm = LM("openai/gpt-5-mini", temperature=0.7, max_tokens=4096)
        assert lm.completion_kwargs["temperature"] == 0.7
        assert lm.completion_kwargs["max_tokens"] == 4096
        assert "max_completion_tokens" not in lm.completion_kwargs


class TestLMCall:
    """Test LM __call__ method."""

    @patch("litellm.completion")
    def test_string_prompt(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response text"
        mock_response.choices[0].finish_reason = "stop"
        mock_completion.return_value = mock_response

        lm = LM("openai/gpt-4.1", temperature=0.5)
        result = lm("hello")

        assert result == "response text"
        mock_completion.assert_called_once_with(
            model="openai/gpt-4.1",
            messages=[{"role": "user", "content": "hello"}],
            num_retries=3,
            drop_params=True,
            temperature=0.5,
        )

    @patch("litellm.completion")
    def test_messages_prompt(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "chat response"
        mock_response.choices[0].finish_reason = "stop"
        mock_completion.return_value = mock_response

        lm = LM("openai/gpt-4.1")
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        result = lm(messages)

        assert result == "chat response"
        mock_completion.assert_called_once_with(
            model="openai/gpt-4.1",
            messages=messages,
            num_retries=3,
            drop_params=True,
        )

    @patch("litellm.completion")
    def test_truncation_warning(self, mock_completion, caplog):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "truncated"
        mock_response.choices[0].finish_reason = "length"
        mock_completion.return_value = mock_response

        lm = LM("openai/gpt-4.1", max_tokens=100)
        result = lm("hello")

        assert result == "truncated"
        assert isinstance(result, LMOutput)
        assert result.finish_reason == "length"
        assert result.strip().finish_reason == "length"
        assert "truncated" in caplog.text.lower()


class TestLMBatchComplete:
    """Test LM batch_complete method."""

    @patch("litellm.batch_completion")
    def test_batch_complete(self, mock_batch):
        resp1 = MagicMock()
        resp1.choices = [MagicMock()]
        resp1.choices[0].message.content = " answer1 "
        resp1.choices[0].finish_reason = "stop"
        resp2 = MagicMock()
        resp2.choices = [MagicMock()]
        resp2.choices[0].message.content = " answer2 "
        resp2.choices[0].finish_reason = "stop"
        mock_batch.return_value = [resp1, resp2]

        lm = LM("openai/gpt-4.1")
        msgs = [
            [{"role": "user", "content": "q1"}],
            [{"role": "user", "content": "q2"}],
        ]
        results = lm.batch_complete(msgs, max_workers=5)

        assert results == ["answer1", "answer2"]
        assert [result.finish_reason for result in results] == ["stop", "stop"]
        mock_batch.assert_called_once_with(
            model="openai/gpt-4.1",
            messages=msgs,
            max_workers=5,
            num_retries=3,
            drop_params=True,
        )

    @patch("litellm.batch_completion")
    def test_batch_complete_forwards_extra_kwargs(self, mock_batch):
        """Extra kwargs passed to batch_complete should be forwarded to litellm."""
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ans"
        resp.choices[0].finish_reason = "stop"
        mock_batch.return_value = [resp]

        lm = LM("openai/gpt-4.1", temperature=0.5)
        lm.batch_complete(
            [[{"role": "user", "content": "q"}]],
            max_workers=3,
            timeout=30,
            api_base="https://custom.api",
        )

        call_kwargs = mock_batch.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["timeout"] == 30
        assert call_kwargs["api_base"] == "https://custom.api"

    @patch("litellm.batch_completion")
    def test_batch_complete_kwargs_override_init(self, mock_batch):
        """Kwargs passed at call time should override init kwargs."""
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ans"
        resp.choices[0].finish_reason = "stop"
        mock_batch.return_value = [resp]

        lm = LM("openai/gpt-4.1", temperature=0.5)
        lm.batch_complete(
            [[{"role": "user", "content": "q"}]],
            temperature=0.9,  # override init value
        )

        call_kwargs = mock_batch.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    @patch("tqdm.auto.tqdm")
    @patch("litellm.completion")
    def test_batch_complete_progress_preserves_input_order(self, mock_completion, mock_tqdm):
        """The observable progress path returns results in request order."""
        mock_tqdm.side_effect = lambda iterator, **_kwargs: MagicMock(
            __enter__=lambda _self: iterator,
            __exit__=lambda *_args: False,
        )

        def complete(*, messages, **_kwargs):
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = messages[0]["content"]
            response.choices[0].finish_reason = "stop"
            response.usage = None
            return response

        mock_completion.side_effect = complete
        lm = LM("openai/gpt-4.1", show_batch_progress=True, batch_progress_desc="test batch")
        msgs = [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ]

        assert lm.batch_complete(msgs, max_workers=2) == ["first", "second"]
        mock_tqdm.assert_called_once()
        assert mock_tqdm.call_args.kwargs["total"] == 2
        assert mock_tqdm.call_args.kwargs["desc"] == "test batch"


class TestLMCostTracking:
    @patch("litellm.completion_cost", return_value=999.0)
    @patch("litellm.completion")
    def test_provider_reported_cost_takes_priority(self, mock_completion, _mock_cost):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "answer"
        response.choices[0].finish_reason = "stop"
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 20
        response.usage.cost = 0.0123
        response.usage.provider_reported_cost_usd = None
        mock_completion.return_value = response

        lm = LM("openrouter/example")
        lm("question")

        assert lm.total_cost == 0.0123
        assert lm.cost_breakdown["calls_by_source"]["provider"] == 1

    @patch("litellm.completion_cost", return_value=0.0)
    @patch("litellm.completion")
    def test_configured_token_price_fallback(self, mock_completion, _mock_cost):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "answer"
        response.choices[0].finish_reason = "stop"
        response.usage.prompt_tokens = 1_000_000
        response.usage.completion_tokens = 500_000
        response.usage.cost = None
        response.usage.provider_reported_cost_usd = None
        response._hidden_params = {}
        mock_completion.return_value = response

        lm = LM("openrouter/example", input_cost_per_million=0.02, output_cost_per_million=0.10)
        lm("question")

        assert lm.total_cost == 0.07
        assert lm.cost_breakdown["calls_by_source"]["estimated"] == 1
        assert lm.cost_breakdown["contains_estimates"] is True


class TestLMRepr:
    def test_repr(self):
        lm = LM("openai/gpt-4.1", temperature=0.5)
        assert "gpt-4.1" in repr(lm)
        assert "temperature=0.5" in repr(lm)


class TestLMConformsToProtocol:
    """Verify LM satisfies the LanguageModel protocol."""

    def test_callable(self):
        lm = LM("openai/gpt-4.1")
        assert callable(lm)


class TestLMRetries:
    """Exercise the retries that ``num_retries`` asks of LiteLLM, without mocking LiteLLM."""

    def test_failed_call_is_retried_and_raises_the_provider_error(self, monkeypatch):
        requests = []

        def not_found(request):
            requests.append(request)
            return httpx.Response(404, json={"error": {"message": "model not found", "type": "invalid_request_error"}})

        monkeypatch.setattr(litellm, "client_session", httpx.Client(transport=httpx.MockTransport(not_found)))
        # A fresh api_base keeps LiteLLM from reusing a client cached by another test.
        lm = LM("openai/gpt-4.1", api_key="sk-test", api_base=f"http://{uuid.uuid4().hex}.invalid/v1")

        # LiteLLM maps the provider's error to one of its own, which subclass OpenAI's.
        with pytest.raises(openai.APIStatusError, match="model not found"):
            lm("hello")

        # The OpenAI SDK does not resend a 404, so the first attempt sends one request.
        # LiteLLM then retries the call itself, which needs tenacity, with three more.
        assert len(requests) == 4
