# DB4AI-EdgeServe

面向云边协同场景的分布式人工智能感知与决策原型，参赛题号 `XH-202606`。项目以 G1–G7 可量化门禁为准，详细指标和阶段退出条件见 `IMPLEMENTATION_PLAN.md`。

## 当前状态

首轮 `G0-CAPMEM` 为 `failed, feasible=0/9`。随后完成 DeepSeek-R1-Distill-Qwen-1.5B 的 170 条冻结 Dev：Math `84.38%`、Code `23.81%`、NLP `17.19%`。该结果来自未量化 BF16 基座，Code/NLP 已无合理恢复余量，因此 P0-A2 关闭，不再训练或量化 DeepSeek。

P0-A3 已证明未蒸馏 Qwen3-1.7B HF 在170题上可保持 Teacher 的 `86.11%`，但量化配置仍未形成联合通过证据。当前进入 P0-A4：

1. 当前 Qwen2.5-14B-AWQ 是不可调优的正式分母，官方完整测试清单固定为 GSM8K 1319、HumanEval 164、CMMLU 11582；
2. 单独的 Qwen2.5-14B BF16 使用4卡 ZeRO-3 LoRA调优，只作为经过答案/执行校验的蒸馏Teacher；
3. Qwen3-1.7B先做共享多任务蒸馏，必要时增加Top-1任务Adapter，再以训练集imatrix生成Q4_K_M；Q3_K_M仅保留为能力失败对照；
4. 量化运行固定使用Q8 KV；96题每项/宏平均≥75%，170题每项/宏平均≥80%，最多两个Student版本；
5. 通过≤1400MB开发内存门后，只允许一次Student官方完整测试，逐题结果封存且不再反馈训练。

Student v1的Q4在170题上保持率为Math/Code/NLP/Macro=`90.625/75.000/80.435/82.020%`，
因Code低于80%未晋级；一次提前运行的官方全测也已失败并封存。v2、P0-A4R任务Adapter
和P0-A4R2温和Code Adapter均未产生足够增益，现已停止。

当前唯一开放路线为P0-A4R3：从v1 HF合并基座重新训练共享Code+NLP LoRA，Math冻结并加入
回放防遗忘；Code扩展到至少3500个真正不同且执行通过的MBPP/APPS/CodeContests任务，
NLP扩展到3000个经过标签校验的中文多领域选择题。只保留Rank 8/16两个预注册候选，
使用全新train-only验证集选择，不重复既有正式全测。量化保持Q4_K_M、Q8 KV和训练数据imatrix。

## 快速检查

```bash
bash scripts/run_p0a.sh checks
bash scripts/run_p0a3.sh preflight
bash scripts/run_p0a4.sh preflight
bash scripts/run_p0a4r3.sh structural-check
bash scripts/run_p0a4r3.sh protocol
```

P0-A4 主流程：

```bash
# 终端A：先运行AWQ分母服务；终端B建立96/170分母，正式全量可稍后运行
bash scripts/run_p0a4.sh baseline-serve
bash scripts/run_p0a4.sh baseline-dev
bash scripts/run_p0a4.sh baseline-full

# 安装训练依赖、下载BF16 Teacher并训练1至3个候选
bash scripts/run_p0a4.sh install-training-deps
bash scripts/run_p0a4.sh download-teacher
bash scripts/run_p0a4.sh teacher-train 1
bash scripts/run_p0a4.sh teacher-serve
bash scripts/run_p0a4.sh teacher-validate 1
bash scripts/run_p0a4.sh teacher-select
bash scripts/run_p0a4.sh teacher-distill 1

# Student蒸馏、合并、量化和逐级门禁
bash scripts/run_p0a4.sh student-train 1
bash scripts/run_p0a4.sh student-merge 1
bash scripts/run_p0a4.sh student-quantize
bash scripts/run_p0a4.sh edge-start
bash scripts/run_p0a4.sh student-smoke96
bash scripts/run_p0a4.sh student-170 1
bash scripts/run_p0a4.sh student-memory
bash scripts/run_p0a4.sh student-full
```

当前v2从以下命令开始，所有后续命令都必须保留版本环境变量；完整分阶段命令见运行手册：

```bash
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh student-v2-preflight
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh student-train 2
```

v2共享与原Top-1 Adapter均因96题Code保持率只有`70.83%`停止，最后一次170题机会尚未消耗。
当前修复入口将NLP改为短理由/直接选项混合蒸馏，将Code改为独立canonical任务和执行通过率
选优：

```bash
bash scripts/run_p0a4r.sh preflight
bash scripts/run_p0a4r.sh code-source-rebuild
bash scripts/run_p0a4r.sh code-build
```

当前本地Code构建已由1500个APPS official-train任务与292个MBPP train任务组成，共1792
个独立训练组；另有42个独立执行验证组，组重叠和正式测试引用均为0，数据门状态为
`promotion_eligible=true`。完整流程见`docs/RUNBOOK_P0A4R.md`。

P0-A4R3从以下命令开始，完整流程和停止条件见`docs/RUNBOOK_P0A4R3.md`：

```bash
bash scripts/run_p0a4r3.sh protocol
bash scripts/run_p0a4r3.sh apps-expand
bash scripts/run_p0a4r3.sh code-all
bash scripts/run_p0a4r3.sh nlp-prepare
bash scripts/run_p0a4r3.sh nlp-generate
bash scripts/run_p0a4r3.sh assemble
bash scripts/run_p0a4r3.sh preflight
```

当前P0-A4R3 Code数据门已通过：MBPP 292 + APPS 2500 + CodeContests 1000，共3792
个不同且可执行校验的训练任务；另有256个全新Code train-only验证组。
NLP数据门也已通过：3000个八领域中文训练组和256个新验证组全部经过Teacher答案与训练
标签校验。组合后的共享训练集为Math/Code/NLP=`1000/3792/3000`，两个候选dry-run通过。

AWQ分母和BF16+LoRA Teacher都以一个vLLM端点在 GPU `0,1,2,3` 上执行TP=4，不是四个单卡副本。详细顺序和失败处理见 `docs/RUNBOOK_P0A4.md`；P0-A3历史诊断仍保留在 `docs/RUNBOOK_P0A.md`。

## 核心目录

| 目录 | 用途 |
|---|---|
| `configs` | 模型、网络、负载、P0-A3历史配置和P0-A4冻结配置 |
| `model_compression` | 数据构建、Teacher校验蒸馏、P0-A4 LoRA训练和Student合并工具 |
| `scripts` | 数据检查、同口径评测、量化、内存门禁和启动入口 |
| `data/splits` | 冻结的 train/validation/test ID 与 hash |
| `data/distill` | 本地训练和校准数据，不提交 Git |
| `models` | 本地 HF 模型和独立 GGUF，不提交 Git |
| `reports/audit` | 数据、能力、内存和门禁审计 |
| `docs` | 数据策略、状态、运行手册和修改记录 |
| `sql`、`docker` | KWDB schema 与本地服务配置 |

## 硬约束

- G1：Math、Code、NLP 与宏平均保持率均不低于 Qwen2.5-14B 的 80%。
- G3：llama.cpp 完整进程树 RSS 加设备内存的推理窗口峰值不超过 1500MB（decimal）。
- 正式 GSM8K 1319、HumanEval 164和CMMLU 11582题不进入训练、候选选择或错误修复。
- 96题和170题均须与AWQ分母逐样本一致；170题只暴露任务级汇总，最多两个Student版本。
- GGUF 文件大小不能替代运行峰值；`status=passed` 的执行审计不能替代能力保持率 Gate。
- 模型、数据和运行产物不上传 GitHub，仓库只保留代码、配置和小型审计证据。
