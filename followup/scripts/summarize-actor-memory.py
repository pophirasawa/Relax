#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Summarize colocated system GPU memory during actor-training phases.

The TensorBoard event wall time marks the end of a reported training step.
Weight synchronization follows actor training, so each actor interval is
reconstructed as::

    end = metric_wall_time - update_weights_time
    start = end - actor_train_time

NVML samples inside those intervals include every process on the colocated
GPU. The result is therefore actor-phase *system* memory, not PyTorch actor
allocated memory.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import statistics
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ACTOR_TIME_TAG = "perf/actor_train_time"
UPDATE_TIME_TAG = "perf/update_weights_time"


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for an ordered sample."""

    if not values:
        raise ValueError("cannot calculate a percentile from an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def load_intervals(run_dir: str, start_step: int, end_step: int) -> list[dict[str, float | int]]:
    files = sorted(glob.glob(os.path.join(run_dir, "tensorboard_log", "**", "events.out.tfevents*"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no TensorBoard event file under {run_dir}")

    accumulator = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])
    if ACTOR_TIME_TAG not in available or UPDATE_TIME_TAG not in available:
        raise KeyError(f"{run_dir} is missing actor/update timing metrics")

    actor_events = {event.step: event for event in accumulator.Scalars(ACTOR_TIME_TAG)}
    update_times = {event.step: event.value for event in accumulator.Scalars(UPDATE_TIME_TAG)}
    intervals = []
    for step in range(start_step, end_step + 1):
        event = actor_events.get(step)
        if event is None:
            continue
        actor_end = event.wall_time - update_times.get(step, 0.0)
        actor_start = actor_end - event.value
        intervals.append(
            {
                "step": step,
                "start_time": actor_start,
                "end_time": actor_end,
                "duration_s": event.value,
            }
        )
    if not intervals:
        raise ValueError(f"{run_dir} has no actor interval in step {start_step}-{end_step}")
    return intervals


def parse_timestamp(value: str) -> float:
    return dt.datetime.fromisoformat(value).timestamp()


def interval_for(timestamp: float, intervals: list[dict[str, float | int]]) -> int | None:
    for interval in intervals:
        if float(interval["start_time"]) <= timestamp <= float(interval["end_time"]):
            return int(interval["step"])
    return None


def summarize_run(label: str, run_dir: str, start_step: int, end_step: int) -> dict[str, Any]:
    intervals = load_intervals(run_dir, start_step, end_step)
    metrics_path = os.path.join(run_dir, "gpu_metrics.csv")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(metrics_path)

    by_gpu: dict[str, list[float]] = {}
    by_step: dict[int, list[float]] = {}
    all_values: list[float] = []
    with open(metrics_path, newline="") as handle:
        for row in csv.DictReader(handle):
            step = interval_for(parse_timestamp(row["timestamp"]), intervals)
            if step is None:
                continue
            value = float(row["memory_used_mib"])
            all_values.append(value)
            by_gpu.setdefault(row["gpu_index"], []).append(value)
            by_step.setdefault(step, []).append(value)

    if not all_values:
        raise ValueError(f"{run_dir} has no NVML sample inside reconstructed actor intervals")

    per_gpu_means = {gpu: statistics.mean(values) for gpu, values in sorted(by_gpu.items())}
    return {
        "label": label,
        "run_dir": label.lower(),
        "window_steps": [start_step, end_step],
        "interval_method": "tb_wall_time - update_weights_time - actor_train_time",
        "intervals": intervals,
        "actor_phase_system_memory_mib": {
            "mean": statistics.mean(all_values),
            "p95": percentile(all_values, 0.95),
            "peak": max(all_values),
            "min": min(all_values),
            "samples": len(all_values),
            "per_gpu_mean": per_gpu_means,
            "per_gpu_mean_min": min(per_gpu_means.values()),
            "per_gpu_mean_max": max(per_gpu_means.values()),
            "per_step_mean": {str(step): statistics.mean(values) for step, values in sorted(by_step.items())},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help='"Label:/path/to/run"')
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--end", type=int, default=11)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    summaries = []
    for spec in args.run:
        label, separator, run_dir = spec.partition(":")
        if not separator:
            raise ValueError(f"invalid run specification: {spec}")
        summaries.append(summarize_run(label, run_dir, args.start, args.end))

    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False)

    print("| Run | Actor-phase mean MiB | P95 MiB | Peak MiB | Per-GPU mean range MiB | Samples |")
    print("|---|---:|---:|---:|---:|---:|")
    for summary in summaries:
        memory = summary["actor_phase_system_memory_mib"]
        print(
            f"| {summary['label']} | {memory['mean']:,.0f} | {memory['p95']:,.0f} | "
            f"{memory['peak']:,.0f} | {memory['per_gpu_mean_min']:,.0f}-{memory['per_gpu_mean_max']:,.0f} | "
            f"{memory['samples']} |"
        )


if __name__ == "__main__":
    main()
