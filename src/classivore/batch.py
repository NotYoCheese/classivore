#!/usr/bin/env python3
"""Shared Anthropic Message Batches API utilities.

Used by enricher, labeler, and any future module that submits batch jobs.
"""

import os
import time

import anthropic
from dotenv import load_dotenv


def get_api_client():
    """Create an Anthropic client using API key from environment.

    Checks CLASSIVORE_API_KEY first, then ANTHROPIC_API_KEY.

    Returns:
        anthropic.Anthropic client instance.

    Raises:
        RuntimeError: If no API key is found.
    """
    load_dotenv()

    api_key = os.getenv("CLASSIVORE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
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
            print(
                f"  Batch {batch_id}: "
                f"{counts.succeeded} succeeded, "
                f"{counts.processing} processing, "
                f"{counts.errored} errored"
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
            print(f"  Warning: {entry.custom_id} — {entry.result.type}")

    print(f"  Batch results: {succeeded} succeeded, {failed} failed")
