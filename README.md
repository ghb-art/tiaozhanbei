# DB4AI-EdgeServe

面向云边协同场景的分布式人工智能感知与决策原型，参赛题号 `XH-202606`。项目以 G1-G7 可量化门禁为准，详细指标与阶段退出条件见 `IMPLEMENTATION_PLAN.md`。

## 当前状态

首轮 `G0-CAPMEM` 已完成，结果为 `failed, feasible=0/9`。DeepSeek-R1-Distill-Qwen-1.5B 的 Q2_K_S 产物为 716.71MB，20 次预热 + 100 次测量的峰值总内存为 920.23MB，但首轮 matched capability smoke 未达到 80%，因此当前没有已晋级的主边缘模型。

现阶段只执行 P0-A2：在 DeepSeek 1.5B 内存安全基座上进行 train-only 能力与输出格式恢复，再重新执行相同 G0。历史 v1-v31 结论保留在 `docs/REVISION_LOG.md`，对应启动代码和本地大模型已清理。

## 快速检查

```bash
bash scripts/run_p0a.sh checks
bash scripts/run_p0a2.sh preflight
```

P0-A2 顺序：

```bash
bash scripts/run_p0a2.sh upper-bound-smoke
bash scripts/run_p0a2.sh upper-bound
bash scripts/run_p0a2.sh train
bash scripts/run_p0a2.sh evaluate-adapter
bash scripts/run_p0a2.sh export
bash scripts/run_p0a2.sh build-imatrix
bash scripts/run_p0a2.sh quantize
bash scripts/run_p0a2.sh g0-reentry
```

完整参数、数据隔离规则和产物路径见 `docs/RUNBOOK_P0A.md`。

## 核心目录

| 目录 | 用途 |
|---|---|
| `configs` | 模型、网络、负载、G0 与 P0-A2 冻结配置 |
| `model_compression` | 蒸馏数据、LoRA 训练、合并导出 |
| `scripts` | 数据检查、评测、量化、内存门禁和一键入口 |
| `data/splits` | 冻结的 train/validation/test ID 与 hash |
| `data/distill` | 本地训练/校准数据，不提交 Git |
| `models` | 本地模型、adapter、GGUF，不提交 Git |
| `reports/audit` | 数据、训练、能力、内存和门禁审计 |
| `docs` | 数据策略、状态、运行手册与修改记录 |
| `sql`、`docker` | KWDB schema 与本地服务配置 |

## 硬约束

- G1：Math、Code、NLP 与宏平均保持率均不低于 80%。
- G3：llama.cpp 完整进程树 RSS 加设备内存的推理窗口峰值不超过 1500MB（decimal）。
- 正式 GSM8K test、HumanEval 164 题和 CMMLU test 不得进入训练或选模。
- GGUF 文件大小不能替代运行峰值内存；Dev/smoke 结果不能写成 Final Gate 结果。
- 模型、数据和运行产物由脚本生成，不上传 GitHub；Git 仓库保留代码、配置与可审计小型报告。
