#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Task 25 收尾实验：从 TensorBoard 和 GPU 采样里提取统一口径的性能窗口。

三组（全参 / 单 LoRA / Mixture-of-LoRA）必须用同一个 step 区间比较。
Mixture 200-step 基线的 step time 随训练推进从 ~272s 降到 ~234s（response
长度随 reward 改善而变化），所以拿对照组的前几十步去比基线的最后 50 step
会凭空多出 16% 的差距。

窗口默认改成 step 2-11（原为 10-59）。依据是基线自身的逐步数据：真正的冷启动
只有 step 0 一步（509.6s），step 1 起就落进 259-308s 的正常带，丢 10 步属于
过度保守。窗口收窄后对照组不必跑满 60 step，实测成本降到约 1/5。

代价要说清楚：窗口越短，落在哪几步的影响越大。基线自身 step 2-11 的均值
（281.68s）比 step 10-59（271.56s）高 3.7%，因为那几步恰好偏慢。三组使用
相同 step 下标可以统一统计口径，但不能消除生成和采样波动，所以结果只表示这个
短窗口，不能替代长程平均。

窗口内会剔除发生 checkpoint 保存的 step。对照组用 SAVE_INTERVAL=1000 不存
checkpoint，基线在 step 50/100/150 存过，不剔除的话基线会白白多算保存时间。

用法：
    python3 summarize-perf-window.py <run_dir> [<run_dir> ...] \
        [--start 2] [--end 11] [--json out.json]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import statistics
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 需要落在报告里的 tag。key 是报告用的短名，value 是 TensorBoard tag。
PERF_TAGS = {
    "step_time_s": "perf/step_time",
    "rollout_time_s": "perf/rollout_time",
    "actor_train_time_s": "perf/actor_train_time",
    "update_weights_time_s": "perf/update_weights_time",
    "save_model_time_s": "perf/save_model_time",
    "step_resp_token_per_s": "perf/step_resp_token_per_s",
    "step_token_per_s": "perf/step_token_per_s",
    "actor_train_tok_per_s": "perf/actor_train_tok_per_s",
    "tokens_per_gpu_per_sec": "perf/tokens_per_gpu_per_sec",
    "mfu_actor_train": "perf/mfu/actor_train",
}

STABILITY_TAGS = {
    "grad_norm": "train/grad_norm",
    "raw_reward": "rollout/raw_reward",
    "loss": "train/loss",
}


def load_scalars(run_dir: str) -> tuple[dict[str, dict[int, float]], dict[int, float]]:
    tb_dir = os.path.join(run_dir, "tensorboard_log")
    files = sorted(glob.glob(os.path.join(tb_dir, "**", "events.out.tfevents*"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no tensorboard event file under {tb_dir}")
    # 一个 run 只应该有一份 event 文件；多份时取最后一份并提示。
    if len(files) > 1:
        print(f"[warn] {run_dir}: found {len(files)} event files, using {files[-1]}")
    accumulator = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])
    out: dict[str, dict[int, float]] = {}
    step_wall_times: dict[int, float] = {}
    for name, tag in {**PERF_TAGS, **STABILITY_TAGS}.items():
        if tag in available:
            events = accumulator.Scalars(tag)
            out[name] = {event.step: event.value for event in events}
            if name == "step_time_s":
                step_wall_times = {event.step: event.wall_time for event in events}
    return out, step_wall_times


def window_steps(scalars: dict[str, dict[int, float]], start: int, end: int) -> list[int]:
    steps = sorted(step for step in scalars["step_time_s"] if start <= step <= end)
    saves = scalars.get("save_model_time_s", {})
    kept = [step for step in steps if saves.get(step, 0.0) <= 0.0]
    return kept


def summarize_series(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _parse_timestamp(value: str) -> float:
    return datetime.datetime.fromisoformat(value).timestamp()


def _in_window(row: dict[str, str], window_start: float | None, window_end: float | None) -> bool:
    if window_start is None or window_end is None:
        return True
    timestamp = row.get("timestamp")
    return timestamp is not None and window_start < _parse_timestamp(timestamp) <= window_end


def gpu_summary(
    run_dir: str,
    window_start: float | None = None,
    window_end: float | None = None,
) -> dict[str, Any]:
    """统计指定 step 时间窗口内的显存和利用率。

    峰值显存按单卡取最大值，平均利用率按窗口内全部采样点取算术平均。

    另外统计 GPU 上的进程占用。注意这里**不再判断"卡是否被外部租户抢占"**：
    nvidia-smi 报的是宿主机 PID，容器 PID namespace 里查不到，早期版本用
    `/proc/<pid>` 是否存在来分 self / foreign，结果把自己的训练进程全判成外部，
    full-perf 误报了 392 个"被抢"采样点。本机内核 3.10 没有 NSpid 字段，容器内
    做不了归属判断，所以只如实记录进程总数。独占性由 pipeline.sh 的 gate_idle
    保证（每阶段提交前要求 8 张卡显存峰值 < 2000 MiB，实测均为 4 MiB/卡）。
    """
    result: dict[str, Any] = {}
    metrics_path = os.path.join(run_dir, "gpu_metrics.csv")
    if os.path.exists(metrics_path):
        peak_by_gpu: dict[str, int] = {}
        utils: list[float] = []
        with open(metrics_path, newline="") as handle:
            for row in csv.DictReader(handle):
                if not _in_window(row, window_start, window_end):
                    continue
                index = row["gpu_index"]
                used = int(row["memory_used_mib"])
                peak_by_gpu[index] = max(peak_by_gpu.get(index, 0), used)
                utils.append(float(row["utilization_gpu_percent"]))
        if peak_by_gpu:
            result["peak_memory_mib_max"] = max(peak_by_gpu.values())
            result["peak_memory_mib_min"] = min(peak_by_gpu.values())
            result["gpu_util_mean_percent"] = statistics.mean(utils)
            result["gpu_samples"] = len(utils)

    # 新版监控写 gpu_procs.csv；早期 run 只有 gpu_contention.csv，它的 foreign_procs
    # 实际含义就是"GPU 上的进程总数"，按这个口径重新解读即可，不用重跑。
    procs_path = os.path.join(run_dir, "gpu_procs.csv")
    legacy_path = os.path.join(run_dir, "gpu_contention.csv")
    if os.path.exists(procs_path):
        column, path = "gpu_procs", procs_path
    elif os.path.exists(legacy_path):
        column, path = "foreign_procs", legacy_path
    else:
        return result

    peak_procs = 0
    total = 0
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if not _in_window(row, window_start, window_end):
                continue
            total += 1
            peak_procs = max(peak_procs, int(row[column]))
    result["gpu_proc_peak"] = peak_procs
    result["gpu_proc_samples"] = total
    return result


def summarize_run(run_dir: str, start: int, end: int) -> dict[str, Any]:
    scalars, step_wall_times = load_scalars(run_dir)
    steps = window_steps(scalars, start, end)
    if not steps:
        raise ValueError(f"{run_dir}: no logged step in requested window {start}-{end}")

    first_step = steps[0]
    last_step = steps[-1]
    if first_step - 1 in step_wall_times:
        window_start_time = step_wall_times[first_step - 1]
    else:
        window_start_time = step_wall_times[first_step] - scalars["step_time_s"][first_step]
    window_end_time = step_wall_times[last_step]
    summary: dict[str, Any] = {
        "run_dir": run_dir,
        "window_requested": [start, end],
        "window_steps_used": len(steps),
        "window_steps_excluded_for_save": sorted(
            step
            for step in scalars["step_time_s"]
            if start <= step <= end and scalars.get("save_model_time_s", {}).get(step, 0.0) > 0.0
        ),
        "total_steps_logged": len(scalars["step_time_s"]),
        "window_start_time": window_start_time,
        "window_end_time": window_end_time,
        "metrics": {},
    }
    for name in list(PERF_TAGS) + list(STABILITY_TAGS):
        if name not in scalars:
            continue
        values = [scalars[name][step] for step in steps if step in scalars[name]]
        stats = summarize_series(values)
        if stats:
            summary["metrics"][name] = stats
    if "step_time_s" in scalars:
        summary["first_step_time_s"] = scalars["step_time_s"].get(0)
        summary["window_wall_time_s"] = sum(
            scalars["step_time_s"][step] for step in steps if step in scalars["step_time_s"]
        )
    summary["gpu"] = gpu_summary(run_dir, window_start_time, window_end_time)
    summary["gpu_full_run"] = gpu_summary(run_dir)
    return summary


def format_row(label: str, summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]

    def mean(name: str) -> str:
        if name not in metrics:
            return "n/a"
        return f"{metrics[name]['mean']:.2f}"

    gpu = summary["gpu"]
    peak = gpu.get("peak_memory_mib_max")
    peak_text = f"{peak:,}" if peak is not None else "n/a"
    procs = gpu.get("gpu_proc_peak")
    share = f"{procs}" if procs is not None else "n/a"
    return (
        f"| {label} | {summary['window_steps_used']} | {mean('step_time_s')} | "
        f"{mean('rollout_time_s')} | {mean('actor_train_time_s')} | "
        f"{mean('step_resp_token_per_s')} | {mean('actor_train_tok_per_s')} | "
        f"{peak_text} | {share} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--start", type=int, default=2, help="窗口起始 step，含")
    parser.add_argument("--end", type=int, default=11, help="窗口结束 step，含")
    parser.add_argument("--json", help="把完整结果写到这个 JSON 文件")
    args = parser.parse_args()

    summaries = []
    for run_dir in args.run_dirs:
        try:
            summaries.append(summarize_run(run_dir, args.start, args.end))
        except FileNotFoundError as error:
            print(f"[skip] {error}")

    print(f"\n统计窗口：step {args.start}-{args.end}（含），已剔除发生 checkpoint 保存的 step\n")
    print(
        "| 运行 | 窗口 step 数 | step time (s) | rollout (s) | actor train (s) "
        "| response tok/s | actor train tok/s | 峰值显存 (MiB) | GPU 进程数峰值 |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for summary in summaries:
        print(format_row(os.path.basename(summary["run_dir"].rstrip("/")), summary))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump(summaries, handle, indent=2, ensure_ascii=False)
        print(f"\n完整结果已写入 {args.json}")


if __name__ == "__main__":
    main()
