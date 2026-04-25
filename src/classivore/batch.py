#!/usr/bin/env python3
"""Shared Anthropic Message Batches API utilities.

Used by enricher, labeler, and any future module that submits batch jobs.
"""

import os
import time

import anthropic
from dotenv import load_dotenv

from classivore.errors import ConfigError
from classivore.logging_config import get_logger

logger = get_logger(__name__)


def get_api_client():
    """Create an Anthropic client using API key from environment.

    Checks CLASSIVORE_API_KEY first, then ANTHROPIC_API_KEY.

    Returns:
        anthropic.Anthropic client instance.

    Raises:
        ConfigError: If no API key is found.
    """
    load_dotenv()

    api_key = os.getenv("CLASSIVORE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ConfigError(
            "No API key found. Set CLASSIVORE_API_KEY or ANTHROPIC_API_KEY "
            "in your environment or .env file."
        )

    return anthropic.Anthropic(api_key=api_key)


def submit_batch(client, requests):
    """Submit a batch of requests to the Anthropic Message Batches API.

    Args:
        client: anthropic.Anthropic client instance.
        requests: List of batch request dicts (custom_id + params).

    Returns:
        Batch ID string, or None if requests is empty.
    """
    if not requests:
        return None

    batch = client.messages.batches.create(requests=requests)
    return batch.id


def poll_until_complete(client, batch_id, poll_interval=30, verbose=False):
    """Poll a batch until processing completes.

    Args:
        client: anthropic.Anthropic client instance.
        batch_id: The batch ID to poll.
        poll_interval: Seconds between polls (default 30).
        verbose: Print progress updates.

    Returns:
        The final MessageBatch object.
    """
    while True:
        batch = client.messages.batches.retrieve(batch_id)

        if verbose:
            counts = batch.request_counts
            logger.info(
                "batch_poll",
                batch_id=batch_id,
                succeeded=counts.succeeded,
                processing=counts.processing,
                errored=counts.errored,
            )

        if batch.processing_status == "ended":
            return batch

        time.sleep(poll_interval)


def iter_succeeded_results(client, batch_id):
    """Iterate over succeeded results from a completed batch.

    Yields (custom_id, message) tuples for succeeded results.
    Logs warnings for errored, canceled, and expired results.

    Args:
        client: anthropic.Anthropic client instance.
        batch_id: The batch ID to retrieve results for.

    Yields:
        Tuple of (custom_id, message) for each succeeded result.
    """
    succeeded = 0
    failed = 0

    for entry in client.messages.batches.results(batch_id):
        if entry.result.type == "succeeded":
            succeeded += 1
            yield (entry.custom_id, entry.result.message)
        else:
            failed += 1
            logger.warning("batch_result_failed", custom_id=entry.custom_id, type=entry.result.type)

    logger.info("batch_results_complete", batch_id=batch_id, succeeded=succeeded, failed=failed)


# Haiku 4.5 Batch API pricing (USD per million tokens).
# Source: anthropic.com/pricing — 50% batch discount applied.
# Kept in batch.py because classivore only uses Haiku for labeling batches.
_HAIKU_4_5_BATCH_PRICING = {
    "input": 0.50,
    "output": 2.50,
    "cache_write_1h": 1.00,
    "cache_read": 0.05,
}


def aggregate_batch_usage(client, batch_id, pricing=None):
    """Sum token usage across a batch's succeeded results.

    Args:
        client: anthropic.Anthropic client instance.
        batch_id: Batch ID to aggregate.
        pricing: Optional per-million-token USD dict overriding the Haiku 4.5
            batch defaults. Keys: input, output, cache_write_1h, cache_read.

    Returns:
        Dict with total_input_tokens (non-cached), total_output_tokens,
        total_cache_creation_tokens, total_cache_read_tokens, cache_hit_rate
        (0.0-1.0), estimated_cost_usd.
    """
    prices = pricing or _HAIKU_4_5_BATCH_PRICING

    input_tokens = 0
    output_tokens = 0
    cache_creation = 0
    cache_read = 0

    for entry in client.messages.batches.results(batch_id):
        if entry.result.type != "succeeded":
            continue
        usage = entry.result.message.usage
        input_tokens += getattr(usage, "input_tokens", 0) or 0
        output_tokens += getattr(usage, "output_tokens", 0) or 0
        cache_creation += getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read += getattr(usage, "cache_read_input_tokens", 0) or 0

    total_input = input_tokens + cache_creation + cache_read
    cache_hit_rate = cache_read / total_input if total_input > 0 else 0.0

    cost = (
        input_tokens * prices["input"]
        + output_tokens * prices["output"]
        + cache_creation * prices["cache_write_1h"]
        + cache_read * prices["cache_read"]
    ) / 1_000_000

    return {
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_cache_creation_tokens": cache_creation,
        "total_cache_read_tokens": cache_read,
        "cache_hit_rate": cache_hit_rate,
        "estimated_cost_usd": cost,
    }
