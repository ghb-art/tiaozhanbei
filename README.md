# DB4AI-EdgeServe 挑战杯项目目录说明

本目录用于承载 `DB4AI-EdgeServe` 挑战杯 / 揭榜挂帅项目的代码、数据、模型、实验结果、报告和复现材料。

项目根目录：

```text
D:\Desktop\STUDY\code\tiaozhanbei
```

整体目录可以按功能理解为：

```text
configs              固定配置
scripts              数据准备、环境检查、gate 验证脚本
experiments          实验运行、消融实验、Final Gate 脚本
model_compression    蒸馏、SFT、量化、学生探测、repair 相关脚本
data                 原始数据、预处理数据、split、蒸馏数据
models               预训练模型、adapter、量化模型、各类算法模型
results              实验输出结果
reports              审计报告、预检报告、最终报告中间产物
runtime              运行时状态、缓存、outbox
docs                 文档、修改记录、复现说明
sql                  KWDB / KaiwuDB 建表 SQL
docker               Dockerfile、compose、镜像构建材料
logs                 服务日志、运行日志、错误日志
```

## 基础工程目录

| 目录 | 用途 |
|---|---|
| `configs` | 放固定配置，比如 `network_profiles.yaml`、`workload_profiles.yaml`、`models.yaml`、`final_config.yaml`。 |
| `scripts` | 放准备类脚本，比如下载数据、校验 split、验证 DB、验证 cloud、生成 manifest。 |
| `experiments` | 放实验运行脚本，比如 `run_minimal_system.py`、`run_final_integrated.py`、消融实验脚本。 |
| `model_compression` | 放模型蒸馏、SFT、量化、学生探测、counterfactual repair 相关脚本。 |
| `sql` | 放 KWDB / KaiwuDB 建表 SQL，比如 `cloud_schema.sql`。 |
| `docs` | 放文档和修改记录，比如 `REVISION_LOG.md`、复现说明、阶段记录。 |
| `docker` | 放 Dockerfile、docker-compose、环境镜像构建文件。 |
| `logs` | 放运行日志、服务日志、失败记录。 |
| `runtime` | 放运行时状态文件，不作为最终实验结果。 |
| `runtime/state_cache` | 放 StateCache 快照，如网络状态、队列状态、模型健康状态。 |
| `runtime/outbox` | 放边缘侧离线同步、ack、补偿任务等 outbox 数据。 |

## 数据目录

| 目录 | 用途 |
|---|---|
| `data` | 所有数据根目录。 |
| `data/raw` | 原始下载数据，原则上尽量不修改。 |
| `data/processed` | 预处理后的数据。 |
| `data/splits` | train / validation / test 划分文件、sample ids、split hash 相关材料。 |
| `data/kwdb` | KWDB 容器挂载的数据目录。 |
| `data/distill` | 蒸馏训练数据根目录。 |
| `data/distill/teacher` | 14B 教师模型生成的 trace。 |
| `data/distill/student_probe` | 1.5B 学生模型探测输出。 |
| `data/distill/repair` | 反事实样本、错误修复样本。 |
| `data/distill/planner` | IBEP 证据规划器训练数据。 |
| `data/datasets` | 各公开数据集根目录。 |
| `data/datasets/gsm8k` | 数学能力数据。 |
| `data/datasets/humaneval` | 代码能力数据。 |
| `data/datasets/mmlu` | 英文 NLP 能力数据。 |
| `data/datasets/cmmlu` | 中文 NLP 能力数据。 |
| `data/datasets/mvtec_ad` | 工业缺陷检测数据。 |
| `data/datasets/neu_det` | 工业缺陷检测辅助数据。 |
| `data/datasets/cityflow` | 交通多摄像头轨迹数据。 |
| `data/datasets/ua_detrac` | 交通检测 / 跟踪辅助数据。 |

## 模型目录

| 目录 | 用途 |
|---|---|
| `models` | 所有模型文件根目录。 |
| `models/pretrained` | 原始预训练模型或本地模型引用。 |
| `models/checkpoints` | 训练中间 checkpoint。 |
| `models/adapters` | LoRA / adapter 根目录。 |
| `models/adapters/plain_kd` | 普通 KD baseline adapter。 |
| `models/adapters/cedd_structured` | CEDD 第一阶段结构化蒸馏 adapter。 |
| `models/adapters/cedd_repair` | CEDD repair 后的 adapter。 |
| `models/quantized` | INT4 / GGUF 量化模型。 |
| `models/planner` | IBEP 证据规划器模型。 |
| `models/runtime_state` | TARS-SSM 运行状态模型。 |
| `models/policy` | CPR 路由策略模型。 |
| `models/graph` | ST-HGCI 图模型。 |

## 报告目录

| 目录 | 用途 |
|---|---|
| `reports` | 报告、审计、摘要类文件根目录。 |
| `reports/preflight` | P0-A0 预检报告。 |
| `reports/audit` | 数据泄漏检查、conflict_gt 抽样审计、baseline 公平性审计。 |
| `reports/final` | 最终报告用 JSON、表格、摘要。 |

## 结果目录

| 目录 | 用途 |
|---|---|
| `results` | 所有实验输出根目录。 |
| `results/dev_train` | train split 上的开发运行结果。 |
| `results/dev` | validation split 上的开发运行结果。 |
| `results/integrated_dev` | P1 集成开发运行结果。 |
| `results/regression_dev` | P1 regression 回归结果。 |
| `results/final` | P2 Final Gate 最终结果。 |
| `results/perception` | 感知支撑指标结果。 |
| `results/perception_online_smoke` | 在线感知 smoke test 结果。 |
| `results/p1_multinode_train` | P1 多节点 train replay 结果。 |
| `results/p1_multinode` | P1 多节点 validation replay 结果。 |
| `results/relation_graph_train` | 训练用时空图构造结果。 |
| `results/relation_graph` | 验证 / 评估用时空图构造结果。 |
| `results/ablation` | 所有消融实验根目录。 |

## 消融实验目录

| 目录 | 用途 |
|---|---|
| `results/ablation/evidence_fixed` | 固定摘要 baseline。 |
| `results/ablation/evidence_ibep` | IBEP 证据规划实验。 |
| `results/ablation/state_boolean` | 布尔 readiness baseline。 |
| `results/ablation/state_tars_ssm` | TARS-SSM 状态模型实验。 |
| `results/ablation/path_load_high_delay` | high_delay 下 load-based baseline。 |
| `results/ablation/path_cpr_high_delay` | high_delay 下 CPR 实验。 |
| `results/ablation/path_load_low_bandwidth` | low_bandwidth 下 load-based baseline。 |
| `results/ablation/path_cpr_low_bandwidth` | low_bandwidth 下 CPR 实验。 |
| `results/ablation/path_load_high_loss` | high_loss 下 load-based baseline。 |
| `results/ablation/path_cpr_high_loss` | high_loss 下 CPR 实验。 |
| `results/ablation/path_load_short_disconnect` | short_disconnect 下 load-based baseline。 |
| `results/ablation/path_cpr_short_disconnect` | short_disconnect 下 CPR 实验。 |
| `results/ablation/conflict_vote` | 简单投票冲突 baseline。 |
| `results/ablation/conflict_sthgci` | ST-HGCI 图冲突推理实验。 |
| `results/ablation/trust_single` | 云端单次校正 baseline。 |
| `results/ablation/trust_feedback` | BTCF 可信度反馈实验。 |

## 建议优先填充顺序

后续真正开始实施时，建议先填：

```text
1. configs
2. docs
3. scripts
4. sql
5. model_compression
6. experiments
```

不要一开始就全量实现 Final 系统。应先完成：

```text
P0-A0 Preflight
G-DATA
G-DB
G-CLOUD
```

确认内存、200ms action header、数据 split、冲突组数量都可行后，再进入模型训练和完整实验。

## 两人分工建议

如果拆成两个人协作：

| 角色 | 负责范围 |
|---|---|
| A：工程、数据、评测、复现负责人 | 环境、数据集、manifest、KWDB、运行脚本、指标脚本、Final Gate。 |
| B：模型、算法、实验负责人 | 14B / 1.5B 模型、蒸馏、量化、证据规划、路由、图模型、可信度校准。 |

两个人之间最重要的接口文件：

```text
dataset_manifest.json
conflict_gt_manifest.json
decision_tuple_trace.csv
runtime_state_trace.csv
evidence_planner_trace.csv
policy_router_trace.csv
relation_graph_trace.csv
trust_calibration_trace.csv
manifest.json
```

A 负责格式稳定和指标可信，B 按格式读写并优化模型机制。
