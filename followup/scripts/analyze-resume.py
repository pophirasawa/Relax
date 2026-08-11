#!/usr/bin/env python3
"""断点续训验收：把恢复后的 rollout 50/51/52 和主任务同 step 逐项对比。

强调一点：RL 的 rollout 是带温度采样的，恢复后重跑同一个 step 不可能得到
逐位相同的数字。所以除了 lr 这种由 scheduler 唯一决定的量要求严格相等之外，
其余指标看的是量级是否一致 —— 如果 optimizer/scheduler 真被重新初始化了，
grad_norm 和 router 分布会出现肉眼可见的跳变，而不是百分之几的抖动。

用法：analyze-resume.py <主任务 tb 目录> <resume tb 目录> [输出 json]
"""

import json
import os
import sys
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

STEPS = [50, 51, 52]
NUM_EXPERTS = 4

# 逐项对比的标量。lr 由 scheduler 唯一决定，要求严格相等；其余看量级。
SCALARS = [
    ("train/lr-pg_0", "学习率", True),
    ("train/grad_norm", "梯度范数", False),
    ("train/molora/aux_loss", "router aux loss", False),
    ("train/molora/balance_loss", "router balance loss", False),
    ("rollout/rewards", "reward", False),
    ("rollout/raw_reward", "raw reward", False),
    ("train/loss", "loss", False),
    ("train/entropy_loss", "entropy loss", False),
]


def load(tb_dir):
    """读取 tb 目录里最新的一个 event 文件。

    resume 目录是从快照整份复制来的，理论上不含旧 event 文件，但复制策略以后
    可能变，这里统一只认最新的那个，避免把两次 run 的曲线混在一起。
    """
    files = [
        os.path.join(tb_dir, f)
        for f in os.listdir(tb_dir)
        if f.startswith("events.out.tfevents.")
    ]
    if not files:
        raise SystemExit(f"no event file under {tb_dir}")
    newest = max(files, key=os.path.getmtime)
    acc = EventAccumulator(newest, size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def series(acc, tag):
    if tag not in acc.Tags()["scalars"]:
        return {}
    return {s.step: s.value for s in acc.Scalars(tag)}


def expert_weights(acc, step):
    """把每个 expert 的 post-topk 平均权重在所有层上取均值。

    单看某一层容易被噪声带偏，全局平均才反映路由整体是否延续了主任务的形态。
    """
    tags = acc.Tags()["scalars"]
    out = []
    for e in range(NUM_EXPERTS):
        suffix = f"expert_{e}_post_topk_mean_weight"
        vals = []
        for t in tags:
            if t.startswith("train/molora/") and t.endswith(suffix):
                v = series(acc, t).get(step)
                if v is not None:
                    vals.append(v)
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def rel_diff(a, b):
    if a is None or b is None:
        return None
    if a == 0 and b == 0:
        return 0.0
    denom = abs(a) if a != 0 else abs(b)
    return (b - a) / denom


def main():
    main_dir, resume_dir = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    m, r = load(main_dir), load(resume_dir)

    resume_steps = sorted(series(r, "perf/step_time").keys())
    report = {
        "main_tb": main_dir,
        "resume_tb": resume_dir,
        "resume_steps_present": resume_steps,
        "scalars": {},
        "expert_weights": {},
        "checks": {},
    }

    # 验收点 1：恢复后第一个 step 是 50，不是 0，也不是把 49 重跑一遍。
    report["checks"]["starts_at_50"] = bool(resume_steps) and resume_steps[0] == 50

    for tag, label, strict in SCALARS:
        ms, rs = series(m, tag), series(r, tag)
        rows = []
        for st in STEPS:
            a, b = ms.get(st), rs.get(st)
            rows.append(
                {"step": st, "main": a, "resume": b, "rel_diff": rel_diff(a, b)}
            )
        report["scalars"][tag] = {"label": label, "strict": strict, "rows": rows}

        if strict:
            ok = all(
                x["main"] is not None
                and x["resume"] is not None
                and abs(x["main"] - x["resume"]) < 1e-12
                for x in rows
                if x["resume"] is not None
            )
            report["checks"][f"strict_equal:{tag}"] = ok

    for st in STEPS:
        report["expert_weights"][st] = {
            "main": expert_weights(m, st),
            "resume": expert_weights(r, st),
        }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
