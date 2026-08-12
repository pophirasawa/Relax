#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Plot actor-phase and end-to-end GPU memory for the Task 25 runs."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = {
    "Full-param": "#c0392b",
    "Single-LoRA": "#2980b9",
    "Mixture-LoRA": "#27ae60",
}


def gib(value_mib: float) -> float:
    return value_mib / 1024


def add_value_labels(ax, bars, suffix=""):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-json", required=True)
    parser.add_argument("--perf-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.actor_json, encoding="utf-8") as handle:
        actor_runs = json.load(handle)
    with open(args.perf_json, encoding="utf-8") as handle:
        perf_runs = json.load(handle)

    labels = [run["label"] for run in actor_runs]
    colors = [COLORS[label] for label in labels]
    perf_by_name = {
        "full-param": perf_runs[0],
        "single-lora": perf_runs[1],
        "mixture-lora": perf_runs[2],
    }
    perf_keys = {
        "Full-param": "full-param",
        "Single-LoRA": "single-lora",
        "Mixture-LoRA": "mixture-lora",
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    # Per-step means show whether a summary is driven by a single transient peak.
    ax = axes[0]
    for run, color in zip(actor_runs, colors):
        values = run["actor_phase_system_memory_mib"]["per_step_mean"]
        steps = sorted(int(step) for step in values)
        ax.plot(
            steps,
            [gib(values[str(step)]) for step in steps],
            marker="o",
            linewidth=1.7,
            markersize=4,
            color=color,
            label=run["label"],
        )
    ax.set_title("Actor-phase memory by step")
    ax.set_xlabel("training step")
    ax.set_ylabel("system memory per GPU (GiB)")
    ax.set_xticks(range(2, 12))
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8)

    # Mean, P95 and peak describe both the typical level and short-lived spikes.
    ax = axes[1]
    metrics = [("mean", "Mean"), ("p95", "P95"), ("peak", "Peak")]
    width = 0.24
    x = list(range(len(metrics)))
    for index, (run, color) in enumerate(zip(actor_runs, colors)):
        memory = run["actor_phase_system_memory_mib"]
        positions = [value + (index - 1) * width for value in x]
        bars = ax.bar(
            positions,
            [gib(memory[key]) for key, _ in metrics],
            width,
            color=color,
            label=run["label"],
        )
        add_value_labels(ax, bars)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_title("Actor-phase memory summary")
    ax.set_ylabel("system memory per GPU (GiB)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    # End-to-end peaks are dominated by rollout allocation and answer OOM risk.
    ax = axes[2]
    actor_peaks = [
        gib(run["actor_phase_system_memory_mib"]["peak"]) for run in actor_runs
    ]
    full_peaks = [
        gib(perf_by_name[perf_keys[label]]["gpu"]["peak_memory_mib_max"])
        for label in labels
    ]
    x = list(range(len(labels)))
    actor_bars = ax.bar(
        [value - 0.18 for value in x],
        actor_peaks,
        0.36,
        color=colors,
        alpha=0.65,
        label="Actor-phase peak",
    )
    full_bars = ax.bar(
        [value + 0.18 for value in x],
        full_peaks,
        0.36,
        color=colors,
        hatch="///",
        label="Step-window peak",
    )
    add_value_labels(ax, actor_bars)
    add_value_labels(ax, full_bars)
    ax.set_xticks(x, labels)
    ax.set_title("Actor peak vs end-to-end peak")
    ax.set_ylabel("system memory per GPU (GiB)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Task 25 GPU memory comparison - step 2-11 (5-second NVML samples)",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.01,
        "System memory includes all colocated processes. End-to-end peaks are rollout-dominated.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
