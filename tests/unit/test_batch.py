#!/usr/bin/env python3
"""Tests for the shared batch API utilities."""

from unittest.mock import MagicMock, patch

import pytest

from classivore.batch import (
    get_api_client,
    iter_succeeded_results,
    poll_until_complete,
    submit_batch,
)


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
        with pytest.raises(RuntimeError, match="No API key found"):
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
        output = capsys.readouterr().out
        assert "cat-2" in output
        assert "errored" in output
        assert "cat-3" in output
        assert "1 succeeded, 2 failed" in output
