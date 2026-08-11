#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Task 25 收尾实验：三组吞吐对比出图，窗口口径与 summarize-perf-window.py 一致。

统一窗口 step 2-11。三组使用同一批 step 下标可以统一统计口径，但短窗口仍然
存在生成和采样波动。Mixture 基线虽然跑了 200 step，这里也只取 step 2-11，
和只跑 12 step 的对照组同口径。

图上一律用英文标签：容器里没有装任何 CJK 字体，中文会渲染成方框。说明文字放在
markdown 报告里。

用法：
    python3 plot-perf-comparison.py --out-dir <目录> \
        --run "Full-param:/path/to/full" \
        --run "Single-LoRA:/path/to/single" \
        --run "Mixture-LoRA:/path/to/mixture" \
        [--start 2] [--end 11]
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)

# 每张子图一个指标：(tag, 标题, 单位, 越大越好)
PANELS = [
    ("perf/step_time", "Step time", "s", False),
    ("perf/rollout_time", "Rollout time", "s", False),
    ("perf/actor_train_time", "Actor train time", "s", False),
    ("perf/step_resp_token_per_s", "Response throughput", "tok/s", True),
    ("perf/actor_train_tok_per_s", "Actor train throughput", "tok/s", True),
    ("rollout/response_len/mean", "Mean response length", "tokens", None),
]

COLORS = {0: "#c0392b", 1: "#2980b9", 2: "#27ae60", 3: "#8e44ad"}


def load(run_dir: str) -> dict[str, dict[int, float]]:
    tb_dir = os.path.join(run_dir, "tensorboard_log")
    files = sorted(glob.glob(os.path.join(tb_dir, "**", "events.out.tfevents*"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no tensorboard event file under {tb_dir}")
    acc = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags()["scalars"])
    out: dict[str, dict[int, float]] = {}
    for tag, *_ in PANELS:
        if tag in available:
            out[tag] = {e.step: e.value for e in acc.Scalars(tag)}
    return out


def window_mean(series: dict[int, float], lo: int, hi: int) -> float | None:
    vals = [series[i] for i in range(lo, hi + 1) if i in series]
    return statistics.mean(vals) if vals else None


def plot_curves(runs, lo, hi, out_path, xmax):
    """逐 step 曲线，把统计窗口画成阴影带。

    画到 xmax 为止而不是画满：Mixture 基线有 200 step，对照组只有 12 step，
    全画出来会让窗口那一段挤成一条线，看不出逐步波动。
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (tag, title, unit, _) in zip(axes.flat, PANELS):
        for idx, (label, series_by_tag) in enumerate(runs):
            series = series_by_tag.get(tag)
            if not series:
                continue
            steps = sorted(s for s in series if s <= xmax)
            ax.plot(
                steps,
                [series[s] for s in steps],
                marker="o",
                markersize=3,
                linewidth=1.4,
                color=COLORS[idx % len(COLORS)],
                label=label,
            )
        ax.axvspan(lo, hi, color="#f1c40f", alpha=0.18, zorder=0)
        ax.set_title(f"{title} ({unit})", fontsize=11)
        ax.set_xlabel("rollout step")
        ax.grid(alpha=0.3, linewidth=0.5)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle(
        f"Task 25 throughput comparison - per-step curves "
        f"(shaded = statistics window, step {lo}-{hi})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_bars(runs, lo, hi, out_path):
    """窗口均值柱状图，数值直接标在柱子上，省得对着刻度猜。"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (tag, title, unit, higher_better) in zip(axes.flat, PANELS):
        labels, values = [], []
        for label, series_by_tag in runs:
            series = series_by_tag.get(tag)
            mean = window_mean(series, lo, hi) if series else None
            if mean is None:
                continue
            labels.append(label)
            values.append(mean)
        bars = ax.bar(labels, values, color=[COLORS[i % len(COLORS)] for i in range(len(labels))])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:,.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        arrow = "" if higher_better is None else (" (higher better)" if higher_better else " (lower better)")
        ax.set_title(f"{title} ({unit}){arrow}", fontsize=11)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        if values:
            ax.set_ylim(0, max(values) * 1.18)
    fig.suptitle(f"Task 25 throughput comparison - window means (step {lo}-{hi})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_overhead_breakdown(runs, lo, hi, out_path):
    """画 step time 构成和 Mixture 相对单 LoRA 的变化。"""

    def mean(run, tag):
        series = run[1].get(tag)
        value = window_mean(series, lo, hi) if series else None
        if value is None:
            raise ValueError(f"{run[0]} is missing {tag} in step {lo}-{hi}")
        return value

    labels = [run[0] for run in runs]
    step_times = [mean(run, "perf/step_time") for run in runs]
    rollout_times = [mean(run, "perf/rollout_time") for run in runs]
    train_times = [mean(run, "perf/actor_train_time") for run in runs]
    other_times = [step - rollout - train for step, rollout, train in zip(step_times, rollout_times, train_times)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axes[0]
    x = range(len(labels))
    ax.bar(x, rollout_times, label="Rollout", color="#2980b9")
    ax.bar(x, train_times, bottom=rollout_times, label="Actor train", color="#c0392b")
    bottoms = [rollout + train for rollout, train in zip(rollout_times, train_times)]
    ax.bar(x, other_times, bottom=bottoms, label="Other", color="#7f8c8d")
    for index, total in enumerate(step_times):
        ax.text(index, total, f"{total:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("seconds per step")
    ax.set_title("Step-time composition")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    single = runs[1]
    mixture = runs[2]
    changes = [
        ("Step time", mean(mixture, "perf/step_time") / mean(single, "perf/step_time") - 1),
        ("Rollout time", mean(mixture, "perf/rollout_time") / mean(single, "perf/rollout_time") - 1),
        ("Actor train time", mean(mixture, "perf/actor_train_time") / mean(single, "perf/actor_train_time") - 1),
        (
            "Response tok/s",
            mean(mixture, "perf/step_resp_token_per_s") / mean(single, "perf/step_resp_token_per_s") - 1,
        ),
        (
            "Actor train tok/s",
            mean(mixture, "perf/actor_train_tok_per_s") / mean(single, "perf/actor_train_tok_per_s") - 1,
        ),
    ]
    ax = axes[1]
    names = [item[0] for item in changes]
    percentages = [item[1] * 100 for item in changes]
    colors = ["#c0392b" if value > 0 and "tok/s" not in name else "#2980b9" for name, value in zip(names, percentages)]
    bars = ax.barh(names, percentages, color=colors)
    ax.axvline(0, color="#2c3e50", linewidth=0.8)
    for bar, value in zip(bars, percentages):
        ax.text(
            value - 0.8 if value >= 0 else value + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}%",
            ha="right" if value >= 0 else "left",
            va="center",
            fontsize=9,
            color="white",
        )
    ax.set_xlabel("Mixture-LoRA vs Single-LoRA")
    ax.set_title("Relative change")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)

    fig.suptitle(f"Task 25 performance change - step {lo}-{hi}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help='格式 "标签:路径"')
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--end", type=int, default=11)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--xmax", type=int, default=15, help="曲线图画到第几个 step")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    runs = []
    for spec in args.run:
        label, _, path = spec.partition(":")
        try:
            runs.append((label, load(path)))
        except FileNotFoundError as error:
            print(f"[skip] {error}")

    if not runs:
        raise SystemExit("no runs loaded")

    curves = plot_curves(runs, args.start, args.end, os.path.join(args.out_dir, "perf-curves.png"), args.xmax)
    bars = plot_bars(runs, args.start, args.end, os.path.join(args.out_dir, "perf-window-means.png"))
    changes = plot_overhead_breakdown(
        runs,
        args.start,
        args.end,
        os.path.join(args.out_dir, "perf-overhead-breakdown.png"),
    )
    print(f"已写出：\n  {curves}\n  {bars}\n  {changes}")

    print(f"\n窗口 step {args.start}-{args.end} 均值：")
    for tag, title, unit, _ in PANELS:
        row = []
        for label, series_by_tag in runs:
            series = series_by_tag.get(tag)
            mean = window_mean(series, args.start, args.end) if series else None
            row.append(f"{label}={mean:,.2f}" if mean is not None else f"{label}=n/a")
        print(f"  {title} ({unit}): " + "  ".join(row))


if __name__ == "__main__":
    main()
