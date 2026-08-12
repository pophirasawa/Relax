# 01 CUDA pytest 验证报告

## 1. 目标

默认 CPU CI 会跳过依赖完整 Megatron、SGLang 或 CUDA 的测试。本轮在 8×A800 环境
补跑这些用例，并执行 GPU 可见的 `pytest tests/`，确认 Task 25 没有隐藏的 CUDA
回归问题。

代码版本：`feat/task25-mixture-lora@a82eb62`。

## 2. Task 25 CUDA 定向测试

| 用例 | 覆盖内容 | 结果 |
|---|---|---|
| `test_mixture_lora_peft_wraps_real_column_parallel_linear_when_available` | 真实 Megatron `ColumnParallelLinear` 的 PEFT 包装 | passed，37.70s |
| `test_pipeline_checkpoint_restores_each_stage_parameters` | PP=2 时各 stage expert/router 的保存与恢复 | passed |
| `test_checkpoint_reshards_mixture_parameters_when_data_parallel_size_shrinks` | DP=2 保存、DP=1 加载时的参数 reshard | passed |

两个 distributed checkpoint 用例合计 79.46 秒，结果为 `2 passed`。

定向命令可以在安装完整 Megatron/SGLang 的 GPU 环境中执行：

```bash
python -m pytest -q \
  tests/backends/megatron/test_mixture_lora.py::test_mixture_lora_peft_wraps_real_column_parallel_linear_when_available \
  tests/backends/megatron/test_mixture_lora_checkpoint_distributed.py::test_pipeline_checkpoint_restores_each_stage_parameters \
  tests/backends/megatron/test_mixture_lora_checkpoint_distributed.py::test_checkpoint_reshards_mixture_parameters_when_data_parallel_size_shrinks
```

## 3. GPU 可见全量测试

执行命令：

```bash
python -m pytest tests/ --ignore=tests/autoscale -p no:cacheprovider \
  -q --tb=line -W ignore --continue-on-collection-errors
```

结果：

```text
1312 passed, 15 skipped, 2 failed in 448.80s
```

失败用例为：

- `tests/core/test_registry_sft.py::test_process_role_keeps_rl_path_unchanged`
- `tests/core/test_registry_sft.py::test_process_role_debug_flags_take_precedence_over_sft`

## 4. Upstream 对照

两项失败的断言比较两个同名 enum 成员的对象身份。测试集合运行时，模块重复导入会
生成不同 enum class，因此出现名称一致、对象身份不同的结果。

| 代码版本 | 测试范围 | 结果 |
|---|---|---|
| `upstream/main@5b23011` | `tests/core/` | 2 failed、28 passed |
| `upstream/main@5b23011` | `test_registry_sft.py` 单文件 | 8 passed |
| `feat/task25-mixture-lora@a82eb62` | `test_registry_sft.py` 单文件 | 8 passed |

失败在不包含 Task 25 修改的 upstream 基线上可复现，单独运行对应文件时两边均通过。
因此这两项不计为 Task 25 回归，本 PR 不修改无关 registry 测试。

## 5. 其他验证

| 检查 | 结果 |
|---|---|
| 完整 CPU 可运行测试集 | 1307 passed、17 skipped、5 deselected |
| Task 25、单 LoRA、VPP 定向回归 | 232 passed |
| Pre-commit 与 Ruff | passed |
| Gitleaks tracked-files 扫描 | no leaks found |
| GitHub Python 3.10 / 3.11 / 3.12 CI | passed |
| GitHub Lint / Pre-commit Checks | passed |

## 6. 结论

Task 25 新增的真实 PEFT 包装、PP checkpoint 和 DP reshard CUDA 用例全部通过。
GPU 可见全量测试没有发现由 Mixture-of-LoRA 引入的新失败。

## 7. 合并最新 main 后的复核

开发分支合并 `main@050ab04` 后，在相同的完整后端环境中重新执行回归测试。合并提交为
`b363c18`，随后使用 `5d86614` 补齐上游新增 S3 loader 测试所需的 Mixture 参数桩。

| 检查 | 结果 |
|---|---|
| `pytest tests/` | 1568 passed、12 skipped |
| Task 25 Megatron CUDA 定向测试 | 61 passed，179.96s |
| `tests/test_s3_model_loader.py` | 51 passed |
| 全参、单 LoRA、Mixture-LoRA 两步 BF16 训练 | 三组均通过 |

全量 pytest 首次运行时，一个分布式测试使用的随机端口被占用；对应测试文件单独重跑为
3 passed。该问题未复现，最终结果未发现由本次合并或 Mixture-of-LoRA 引入的回归。
