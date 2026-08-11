# Task 25 Mixture-of-LoRA 200-step 实验分析

分析日期：2026-08-10

## 1. 结论

BF16 Mixture-of-LoRA 正式任务成功连续完成 200 step。训练过程中没有 scalar NaN/Inf，reward 明显改善，grad norm 保持有限且稳定，最后 50 step 的所有 72 个路由 site 均未塌缩到单一 expert。

本次结果覆盖 Mixture 200-step、路由指标和 checkpoint 保存。Checkpoint 续训、CUDA
测试及全参/单 LoRA/Mixture 性能对比见 `../followup/`。

## 2. 实验信息

- 状态：`succeeded`
- 实验运行时的功能 HEAD 为 `6d1d751`，工作树包含 BF16 recipe 和中英文文档修改；这些实验配置修改随后提交为 `317cc44`。
- 模型与任务：Qwen3-4B、DAPO math、GRPO、colocate、BF16。
- 并行：8×A800 80GB、TP=2、SP、PP=1、CP=1。
- Mixture 配置：4 experts、rank 16、Top-K=2、router aux coefficient=0.01。
- Batch：rollout batch 16、8 samples/prompt、global batch 128、最大 response length 8192。
- 连续训练时间：2026-08-10 01:50:27 至 15:52:31，共 14:02:04。
- 从提交 job 到训练及最终 checkpoint 结束：约 14:09:46。

## 3. 完整性

- TensorBoard 的 1474 个 scalar tag 均覆盖 step 0–199，共 200 个点。
- `rollout_result/train` 包含 200 个 JSONL 文件。
- 保存了 `iter_0000049`、`iter_0000099`、`iter_0000149` 和 `iter_0000199`。
- `latest_checkpointed_iteration.txt` 为 199。
- step-50 独立快照已完成，tracker 为 49，包含 `iter_0000049` 和 `global_dataset_state_dict_49.pt`。

## 4. 稳定性与训练效果

| 指标 | 全 200 step | 前 50 step | 最后 50 step |
|---|---:|---:|---:|
| raw reward 均值 | 0.0503 | -0.4766 | 0.3475 |
| train loss 均值 | 0.03236 | 0.01556 | 0.04369 |
| grad norm 均值 | 0.01000 | 0.00957 | 0.01019 |
| router aux loss 均值 | 0.01047 | 0.01081 | 0.01022 |
| 全局 Top-K 前归一化熵 | 0.8986 | 0.8895 | 0.9053 |
| 全局 Top-K 后归一化熵 | 0.9322 | 0.9251 | 0.9371 |

- 所有 TensorBoard scalar 中非有限值数量为 0。
- grad norm 全程范围为 0.00380–0.01751，没有梯度爆炸或消失迹象。
- raw reward 从前 50 step 的 -0.4766 提高到最后 50 step 的 0.3475。
- RL surrogate loss 可以为负，实测最小值 -0.00161 不属于异常。
- `train/ppo_kl` 为 0 与本次 `kl-loss-coef=0` 配置一致。

## 5. 最后 50 step 路由

全局 expert 平均激活权重：

| Expert | 平均激活权重 |
|---|---:|
| 0 | 0.2502 |
| 1 | 0.2457 |
| 2 | 0.2501 |
| 3 | 0.2540 |

72 个 site 的检查结果：

- 最大的单 expert 平均激活权重为 0.4110，位于 `decoder.layers.12.self_attention.linear_qkv`。
- 该 site 的四个 expert 平均激活权重为 `[0.0598, 0.2613, 0.2679, 0.4110]`。
- 该 site 最大 selection share 为 0.3953，最大 Top-1 比例为 0.5111。
- 所有 72 个 site 的四个 expert 平均激活权重都大于 0.05。
- 没有 site 的最大 expert 平均激活权重超过 0.80。

因此，最后 50 step 没有出现聚合激活集中到单一 expert 的情况。

## 6. 低熵 site 的解释

最后 50 step 有 10 个后层 site 的逐 token Top-K 前归一化熵低于 0.75。最低的是 `decoder.layers.34.self_attention.linear_qkv`：

- Top-K 前归一化熵：0.3172。
- Top-K 后归一化熵：0.4757。
- 四个 expert 平均激活权重：`[0.2033, 0.1524, 0.2786, 0.3657]`。
- 四个 expert selection share：`[0.1708, 0.2148, 0.2804, 0.3341]`。
- 四个 expert Top-1 比例：`[0.2058, 0.1441, 0.2776, 0.3726]`。

代码中的 entropy 是逐 token 计算后再求均值。低熵表示 router 对每个 token 的选择更确定；只要不同 token 仍分配到不同 expert，就不等于 expert 塌缩。该 site 的聚合权重、选择份额和 Top-1 比例均分布在四个 expert 上，因此这里更符合 token specialization，而不是单 expert collapse。

此前 TODO 中“每个 site 的 Top-K 前熵不低于 0.75”是错误的硬门槛，已删除。熵继续作为解释路由置信度的观测指标，防塌缩以跨 token 聚合激活权重和选择份额判断。

## 7. 性能与显存

| 指标 | 全 200 step | 最后 50 step |
|---|---:|---:|
| step time 均值 | 253.67s | 234.19s |
| 每步 response tokens/s 算术均值 | 2871.57 | 2717.36 |
| 每步 actor train tokens/s 算术均值 | 9926.65 | 9934.79 |
| actor train time | 75.60s | 66.32s |
| update weights time | 4.32s | 4.34s |

- 首步包含 warmup，step time 为 509.63s。
- GPU 峰值显存为 64,084 MiB，即约 62.58 GiB。
- 各卡峰值范围为 63,554–64,084 MiB。
- GPU 平均利用率为 83.92%，该值包含启动、checkpoint 和阶段切换。
- 目前只有 Mixture 数据，尚不能形成相对全参和单 LoRA 的 before/after 结论。

## 8. 非阻塞日志

日志中的 Qwen3-ASR `cache_position` 文档校验消息来自 Megatron-Bridge 的模型注册扫描。当前任务使用 Qwen3-4B，这些消息没有终止训练，也没有影响 200-step 指标和 checkpoint。

## 9. 尚未完成

1. 从 step-50 独立快照启动短续训，检查 iteration、optimizer、scheduler、数据进度和指标连续性。
2. 完成全参 BF16 冒烟和 warmup 后的稳定性能窗口，确认正常路径未受影响，并记录显存与吞吐。
3. 完成单 LoRA BF16 smoke，跑过 warmup 后记录相同口径的显存、step time 和吞吐，不要求运行 200 step。
4. 汇总三组 before/after，并决定是否需要训练端 sparse executor 优化。

当前只能判定 Mixture 200-step 子项通过，不能把 Task 25 的全部实验验收标记为完成。
