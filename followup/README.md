# Task 25 Mixture-of-LoRA 验收补充实验

本目录整理 Task 25 的完整实验闭环，覆盖 CUDA 回归、checkpoint 真实续训，以及全参、
单 LoRA、Mixture-LoRA 的统一性能对比。Mixture 200-step 主实验材料位于公开分支的
`mixture-200step/` 目录，本目录补充其余验收证据。

实验代码版本为 `feat/task25-mixture-lora@a82eb62`。性能对比使用 Qwen3-4B、DAPO
math、GRPO、BF16 和 colocate，硬件规格为 8×A800 80GB。

## 文档结构

| 文件 | 内容 |
|---|---|
| `00_环境与前提.md` | 软件环境、最终 workload、受控变量和统计定义 |
| `01_CUDA_pytest.md` | Task 25 CUDA 定向测试、GPU 可见全量测试及 upstream 对照 |
| `02_断点续训.md` | iteration-50 checkpoint 的训练状态、数据状态和路由恢复验证 |
| `03_吞吐实验.md` | 三组实验配置、step 2–11 统一窗口和原始性能数据 |
| `04_三组对比结论.md` | 相对变化、耗时分解、显存解释和优化方向 |
| `results/` | 逐步统计 JSON、checkpoint 对比 JSON 和图片 |
| `scripts/` | TensorBoard/NVML 汇总、checkpoint 对比和绘图脚本 |

## 验收结论

| 验收项 | 结果 | 证据 |
|---|---|---|
| `--lora-num-experts 4 --lora-rank 16` 可启用 | 通过 | 200-step 主实验完成 |
| base 冻结，只更新 expert 与 router | 通过 | 参数分类、梯度测试和训练日志 |
| 单 expert 与原有 LoRA 兼容 | 通过 | `N=1` 回归测试及单 LoRA 12-step 对照 |
| Qwen3-4B DAPO colocate GRPO ≥200 step | 通过 | BF16 连续完成 200 step，无 NaN/Inf |
| 全参、单 LoRA、Mixture 性能对比 | 通过 | 三组统一统计 step 2–11 |
| expert 路由未塌缩 | 通过 | 最后 50 step 的 72 个 site 均满足防塌缩条件 |
| checkpoint 保存与恢复一致 | 通过 | 单测、DP reshard 和 iteration-50 真实续训 |

## 三组性能摘要

| 指标 | 全参 | 单 LoRA | Mixture-LoRA |
|---|---:|---:|---:|
| step time (s) | 218.98 | 219.11 | 281.68 |
| rollout time (s) | 111.72 | 113.79 | 155.61 |
| actor train time (s) | 78.61 | 76.92 | 89.91 |
| response 吞吐 (tok/s) | 3964.85 | 3952.03 | 3075.76 |
| actor train 吞吐 (tok/s) | 11293.67 | 11511.07 | 9851.78 |
| actor 阶段系统显存均值 (MiB/GPU) | 38,259 | 26,669 | 29,158 |
| actor 阶段系统显存 P95 (MiB/GPU) | 46,391 | 30,100 | 32,190 |
| actor 阶段系统显存峰值 (MiB/GPU) | 47,550 | 31,258 | 33,678 |
| step 2–11 系统峰值显存 (MiB) | 61,632 | 61,656 | 63,494 |
| 平均 response 长度 (token) | 6787.07 | 6767.17 | 6773.76 |

单 LoRA 与全参的 step time 相差 0.06%，该窗口内两者系统吞吐接近。当前 dense
Mixture 路径比单 LoRA 慢 28.56%，其中 rollout 贡献 66.8% 的额外 step time。
三组平均 response 长度相差不到 0.3%，生成工作量处于同一水平。

Actor 阶段的 NVML 系统显存均值显示，单 LoRA 和 Mixture-LoRA 相比全参分别减少
11,590 MiB（30.3%）和 9,101 MiB（23.8%）。三组统一设置
`sglang-mem-fraction-static=0.7`，rollout 会预留相近大小的静态显存池，KV cache 和
生成 batch 又会把显存推到该容量附近，因此端到端峰值仍接近 60–62 GiB。两种指标回答不同问题：
actor 阶段统计反映训练时的显存收益，端到端峰值用于判断作业是否会 OOM。

Mixture 的结果反映当前实现成本：训练端与 rollout 端都会计算全部四个 expert，再按
Top-K 权重组合输出。后续 grouped sparse 或融合 kernel 可以在不改变路由语义和参数
格式的前提下减少无效 expert 计算。

## 统一统计口径

- 三组统一统计 step 2–11，共 10 个 step。
- Step 0 的初始化开销不进入窗口。
- 窗口内没有 checkpoint 保存。
- 耗时与吞吐来自 TensorBoard 相同 step 下标。
- Actor 阶段显存根据 TensorBoard 的 actor train 和权重同步耗时重建阶段区间，再汇总
  同期 5 秒 NVML 采样的均值、P95 和峰值。
- 端到端显存取统一 step 窗口内所有 GPU 的单卡最大值。
- 三组使用相同模型、数据、batch、最大 response 长度、并行方式和显存配额。

该窗口用于验收场景的短时性能估计。单次十步窗口无法替代重复实验或长程性能均值，
报告在 `03_吞吐实验.md` 和 `04_三组对比结论.md` 中单独说明这一限制。

## 公开材料范围

公开分支只保存复核结论所需的报告、汇总 JSON、分析脚本、指标 CSV 和图片。训练
checkpoint、完整 TensorBoard、运行目录、机器标识、网络配置、调度系统 job ID 和
环境编排日志不进入公开材料。
