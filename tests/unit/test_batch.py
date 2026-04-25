#!/usr/bin/env python3
"""Tests for the shared batch API utilities."""

from unittest.mock import MagicMock, patch

import pytest

from classivore.batch import (
    aggregate_batch_usage,
    get_api_client,
    iter_succeeded_results,
    poll_until_complete,
    submit_batch,
)


def _make_result(custom_id, *, input_tokens=0, output_tokens=0,
                 cache_creation=0, cache_read=0, result_type="succeeded"):
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = cache_creation
    usage.cache_read_input_tokens = cache_read

    entry = MagicMock()
    entry.custom_id = custom_id
    entry.result.type = result_type
    entry.result.message.usage = usage
    return entry


class TestGetApiClient:
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"})
    @patch("classivore.batch.anthropic")
    def test_from_anthropic_key(self, mock_anthropic):
        client = get_api_client()
        mock_anthropic.Anthropic.assert_called_once_with(api_key="sk-test-key")

    @patch.dict("os.environ", {
        "CLASSIVORE_API_KEY": "sk-classivore",
        "ANTHROPIC_API_KEY": "sk-anthropic",
    })
    @patch("classivore.batch.anthropic")
    def test_classivore_key_takes_precedence(self, mock_anthropic):
        client = get_api_client()
        mock_anthropic.Anthropic.assert_called_once_with(api_key="sk-classivore")

    @patch("classivore.batch.load_dotenv")
    @patch.dict("os.environ", {}, clear=True)
    def test_missing_key_raises(self, mock_dotenv):
        from classivore.errors import ConfigError
        with pytest.raises(ConfigError, match="No API key found"):
            get_api_client()


class TestSubmitBatch:
    def test_submits_requests(self):
        client = MagicMock()
        client.messages.batches.create.return_value = MagicMock(id="batch-123")

        requests = [{"custom_id": "req-1", "params": {}}]
        batch_id = submit_batch(client, requests)

        assert batch_id == "batch-123"
        client.messages.batches.create.assert_called_once_with(requests=requests)

    def test_empty_returns_none(self):
        client = MagicMock()
        batch_id = submit_batch(client, [])

        assert batch_id is None
        client.messages.batches.create.assert_not_called()


class TestPollUntilComplete:
    @patch("classivore.batch.time.sleep")
    def test_polls_until_ended(self, mock_sleep):
        client = MagicMock()

        in_progress = MagicMock()
        in_progress.processing_status = "in_progress"
        in_progress.request_counts = MagicMock(
            succeeded=0, processing=5, errored=0,
        )

        ended = MagicMock()
        ended.processing_status = "ended"
        ended.request_counts = MagicMock(
            succeeded=5, processing=0, errored=0,
        )

        client.messages.batches.retrieve.side_effect = [in_progress, ended]

        result = poll_until_complete(client, "batch-123", poll_interval=1)

        assert result.processing_status == "ended"
        assert mock_sleep.call_count == 1

    @patch("classivore.batch.time.sleep")
    def test_returns_immediately_if_already_ended(self, mock_sleep):
        client = MagicMock()

        ended = MagicMock()
        ended.processing_status = "ended"

        client.messages.batches.retrieve.return_value = ended

        result = poll_until_complete(client, "batch-123")

        assert result.processing_status == "ended"
        mock_sleep.assert_not_called()


class TestIterSucceededResults:
    def test_yields_succeeded(self):
        client = MagicMock()

        msg = MagicMock()
        entry = MagicMock()
        entry.custom_id = "cat-1"
        entry.result.type = "succeeded"
        entry.result.message = msg

        client.messages.batches.results.return_value = [entry]

        results = list(iter_succeeded_results(client, "batch-123"))

        assert len(results) == 1
        assert results[0] == ("cat-1", msg)

    def test_skips_errors_and_logs(self, capsys):
        from classivore.logging_config import configure_logging
        configure_logging()

        client = MagicMock()

        succeeded = MagicMock()
        succeeded.custom_id = "cat-1"
        succeeded.result.type = "succeeded"
        succeeded.result.message = MagicMock()

        errored = MagicMock()
        errored.custom_id = "cat-2"
        errored.result.type = "errored"

        expired = MagicMock()
        expired.custom_id = "cat-3"
        expired.result.type = "expired"

        client.messages.batches.results.return_value = [succeeded, errored, expired]

        results = list(iter_succeeded_results(client, "batch-123"))

        assert len(results) == 1
        output = capsys.readouterr().err
        assert "cat-2" in output
        assert "cat-3" in output


class TestAggregateBatchUsage:
    def test_sums_usage_across_succeeded_results(self):
        client = MagicMock()
        client.messages.batches.results.return_value = [
            _make_result("r1", input_tokens=100, output_tokens=50,
                         cache_creation=2000, cache_read=0),
            _make_result("r2", input_tokens=100, output_tokens=60,
                         cache_creation=0, cache_read=2000),
            _make_result("r3", input_tokens=100, output_tokens=40,
                         cache_creation=0, cache_read=2000),
        ]

        stats = aggregate_batch_usage(client, "batch-123")

        assert stats["total_input_tokens"] == 300
        assert stats["total_output_tokens"] == 150
        assert stats["total_cache_creation_tokens"] == 2000
        assert stats["total_cache_read_tokens"] == 4000

    def test_skips_non_succeeded_results(self):
        client = MagicMock()
        client.messages.batches.results.return_value = [
            _make_result("r1", input_tokens=100, output_tokens=50, cache_read=1000),
            _make_result("r2", input_tokens=999, output_tokens=999, result_type="errored"),
            _make_result("r3", input_tokens=999, output_tokens=999, result_type="expired"),
        ]

        stats = aggregate_batch_usage(client, "batch-123")

        assert stats["total_input_tokens"] == 100
        assert stats["total_output_tokens"] == 50
        assert stats["total_cache_read_tokens"] == 1000

    def test_handles_none_cache_fields(self):
        """Responses without caching return None for cache token fields."""
        client = MagicMock()

        usage = MagicMock()
        usage.input_tokens = 500
        usage.output_tokens = 100
        usage.cache_creation_input_tokens = None
        usage.cache_read_input_tokens = None

        entry = MagicMock()
        entry.custom_id = "r1"
        entry.result.type = "succeeded"
        entry.result.message.usage = usage

        client.messages.batches.results.return_value = [entry]

        stats = aggregate_batch_usage(client, "batch-123")

        assert stats["total_input_tokens"] == 500
        assert stats["total_cache_creation_tokens"] == 0
        assert stats["total_cache_read_tokens"] == 0

    def test_cache_hit_rate(self):
        client = MagicMock()
        # 1 cache-creation call (2000 tokens written) + 4 cache-hit calls (2000 read each)
        results = [_make_result("r0", input_tokens=100, output_tokens=50,
                                cache_creation=2000, cache_read=0)]
        for i in range(1, 5):
            results.append(_make_result(f"r{i}", input_tokens=100, output_tokens=50,
                                        cache_creation=0, cache_read=2000))
        client.messages.batches.results.return_value = results

        stats = aggregate_batch_usage(client, "batch-123")

        # cache_hit_rate = cache_read / (cache_read + cache_creation + input)
        # = 8000 / (8000 + 2000 + 500) = 8000 / 10500
        assert stats["cache_hit_rate"] == pytest.approx(8000 / 10500, rel=1e-6)

    def test_cache_hit_rate_zero_when_no_tokens(self):
        client = MagicMock()
        client.messages.batches.results.return_value = []

        stats = aggregate_batch_usage(client, "batch-123")

        assert stats["cache_hit_rate"] == 0.0
        assert stats["total_input_tokens"] == 0
        assert stats["estimated_cost_usd"] == 0.0

    def test_cost_reflects_cache_discount(self):
        """Cache reads cost less than fresh input; cache writes cost more."""
        client = MagicMock()

        # Case A: all fresh input
        client.messages.batches.results.return_value = [
            _make_result("r1", input_tokens=10_000, output_tokens=0),
        ]
        no_cache_cost = aggregate_batch_usage(client, "b")["estimated_cost_usd"]

        # Case B: same total tokens, all cache-read
        client.messages.batches.results.return_value = [
            _make_result("r1", input_tokens=0, output_tokens=0, cache_read=10_000),
        ]
        cache_hit_cost = aggregate_batch_usage(client, "b")["estimated_cost_usd"]

        # Case C: same total tokens, all cache-write
        client.messages.batches.results.return_value = [
            _make_result("r1", input_tokens=0, output_tokens=0, cache_creation=10_000),
        ]
        cache_write_cost = aggregate_batch_usage(client, "b")["estimated_cost_usd"]

        assert cache_hit_cost < no_cache_cost
        assert cache_write_cost > no_cache_cost
        assert cache_hit_cost > 0
