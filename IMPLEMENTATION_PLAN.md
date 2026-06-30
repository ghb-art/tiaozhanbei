# DB4AI-EdgeServe v4.1.0-research-final

## 基于 KWDB/KaiwuDB 的云边端协同轻量大模型推理与一致性决策系统

### 面向"挑战杯"揭榜挂帅 XH-202606 最终实施方案

---

**方案版本:** v4.1.0-research-final（研究机制升级版；数据下载后由 dataset_manifest.json 冻结实际 split 与统计）
**最后更新:** 2026-06-24
**发榜单位:** 山东浪潮数据库技术有限公司
**竞赛题目:** XH-202606

---

## 0. 方案总览

### 0.1 定位

DB4AI-EdgeServe 以 KWDB/KaiwuDB 为云边端数据底座，围绕 14B→1.5B 云边闭环语义蒸馏、信息瓶颈证据规划、多源运行状态建模、约束策略路由、时空异构图冲突推理与节点可信度校准闭环，在工业缺陷检测和交通监控两个场景中实现七项核心指标全部达标。

### 0.2 静态实施方案与动态执行边界

本实验 plan 是预先定义的静态实施方案，用于固定项目的研究结构、实验边界、数据隔离规则、指标计算口径、gate 条件、日志输出和复现要求。它不是对每一个脚本实现细节、数据字段可得性和运行时异常处理的不可变约束。正式实验和 Codex 实施过程中，如遇到数据集字段缺失、公开数据版本差异、本地下载结构不同、模型服务异常、脚本不可运行等具体问题，必须基于实际数据目录、`dataset_manifest.json`、`manifest.json`、trace CSV、运行日志和错误信息进行具体分析，并在不改变核心研究目标的前提下修正实现方案。

**允许动态调整：** 数据集本地目录解析方式、数据字段 fallback 构造方式、脚本参数/文件路径/批大小/运行命令、预处理实现细节、日志字段补充、CityFlow 子集扩展、运行时异常恢复策略、不影响指标口径的工程优化。

**不得随意改变：** 三大研究内容和六项关键技术成果、G1-G7 hard gate、train/validation/test 隔离规则、Final Gate 前冻结原则、Final Gate 失败后不得调参重跑、test split 不得进入蒸馏/planner/policy/graph/calibration 训练或 validation 调参、五方案对比边界、六项成果核心消融关系、dataset_manifest/manifest/conflict_gt_manifest 必须记录最终实际执行状态。

所有动态调整必须记录到 `docs/REVISION_LOG.md` 和 `manifest.json`。若调整影响 hash，必须重新生成并在最终报告中说明属于工程实现修正，不属于测试结果后调参。

### 0.3 三大研究内容、六项关键技术成果与论文链路

```
研究内容一: 面向边缘资源约束的边侧轻量决策能力构建
  ├── 成果1: 面向边缘自治决策的云边闭环语义蒸馏技术 (CEDD)
  ├── 成果2: 信息瓶颈驱动的语义证据规划与风险校准复核技术 (IBEP)
  └── 论文产出: 边缘自治决策轻量大模型构建方法
研究内容二: 面向动态云边环境的状态就绪编排与协同推理调度
  ├── 成果3: 多源运行状态空间建模的云边就绪预测技术 (TARS-SSM)
  ├── 成果4: 约束策略学习驱动的云边路径路由与弱网自治技术 (CPR)
  └── 论文产出: 弱网云边协同推理调度方法
研究内容三: 面向多节点关联任务的全局一致决策技术
  ├── 成果5: 时空异构图驱动的多节点冲突语义推理技术 (ST-HGCI)
  ├── 成果6: 节点可信度后验校准与闭环反馈的一致性优化技术 (BTCF)
  └── 论文产出: 多边缘节点一致性决策方法
```

三篇论文共享数据、模型、KWDB 状态底座和 Final Gate，但研究对象逐级递进: 研究内容一产出可自治边侧模型与证据规划器，研究内容二基于其结构化决策与校准风险进行云边路径选择，研究内容三利用多节点决策 trace 构建时空图并将一致性反馈回流到蒸馏与调度数据池。

### 0.4 自适应机制约束

申报书硬性指标、SLA deadline、网络 profile、workload profile、split 隔离和 Final Gate 规则属于验收与实验条件，可以固定。完整方案的在线机制不得使用人工设定的特征权重、人工相似度加权、人工冲突强度公式、人工复核阈值或人工可信度更新率；这些固定规则只允许作为 baseline/ablation。

完整方案必须采用数据驱动的模型化机制: 蒸馏模型、证据规划器、运行状态模型、路径策略、时空图模型与可信度校准模型由 train trace、教师输出、标注数据和运行反馈训练得到，validation trace 仅用于校准、模型选择和消融评估。Final Gate 前冻结模型结构、训练产物、校准集、策略快照、图模型快照、可信度初始后验和所有 hash；Final Gate 中不再调参，但允许基于真实运行日志更新状态缓存、outbox 和节点可信度后验。

### 0.5 两级 gate 与核心 gate 总表

| Gate | 通过条件 (Final) | 通过条件 (Dev) | 阶段 |
|------|-----------------|---------------|------|
| G1 | Math_ratio≥80%, Code_ratio≥80%, NLP_ratio≥80%, Overall_R_cap≥80% | ≥82% | P0-A-dev → P2-final |
| G2 | high_delay≥75%, low_bandwidth≥75%, high_loss≥75%, weak_avg≥75% | ≥78% | P0-B-dev → P1-Reg → P2-final |
| G3 | P95_RSS ≤1500 MB(decimal); 同时报告 MiB | ≤1450 MB(decimal) | P0-A-dev → P2-final |
| G4 | 4 profile+combined ≥90%; 存在provisional的: correction≥90% | ≥92% | P0-B-dev → P1-Reg → P2-final |
| G5 | P95_E2E industrial≤0.2s, traffic≤0.2s, combined≤0.2s | ≤0.18s | P0-B-dev → P1-Reg → P2-final |
| G6 | post_correction_conflict_ratio ≤5% | ≤4% | P1-dev → P2-final |
| G7 | gt_conflict_resolution_success_rate ≥90% | ≥92% | P1-dev → P2-final |

功能 gate: G-DATA, G-DB, G-CLOUD, G-KD-TRACE, G-Support-Perception

### 0.6 指标层级

| 层级 | 内容 | 性质 |
|------|------|------|
| 第一层 | G1-G7 hard gate | 申报书验收，必须达标 |
| 第二层 | 六项成果支撑指标 + SCU_support | 证明每项机制有效，只报告不设 gate |
| 第三层 | 消融对照指标 | 证明完整机制优于退化版本，只报告不设 gate |

SCU_support (300题结构化理解) 仅写入 support_metrics.csv，不进入 G1 hard gate。

### 0.7 风险与早期验证矩阵

| Gate | 风险等级 | 最早验证阶段 | 早期通过条件 | 失败处理 |
|------|---------|-------------|-------------|---------|
| G1 | 高 | P0-A0/P0-A | 100题 smoke 与正式 dev 子集趋势接近 G1-dev | 优先补 CEDD repair/反事实边界蒸馏, 不进入 P0-B final 路线 |
| G2 | 中 | P0-B | weak profile 下 DB4AI 相对 cloud-only-summary 有稳定 TTFT 差距 | 检查 evidence planning、云端请求率和网络 profile 注入 |
| G3 | 中低 | P0-A0 | base 1.5B INT4 稳态 RSS 明显低于 1500 MB(decimal) | 更换推理后端、收紧 context/KV cache/service wrapper |
| G4 | 高 | P0-B smoke | 200ms 内可解析 action header 成功率接近 G4-dev | 优先启用短字段 constrained decoding 和 provisional decision |
| G5 | 高 | P0-B smoke | P95 business_e2e_ms 接近 0.18s dev 线 | 拆分 explanation/correction 异步路径, 压缩证据与输出 token |
| G6 | 中高 | G-DATA dry-run/P1 | relation_group_count、conflict_group_count 和类型分布满足数据门槛 | 先修 conflict_gt 构造与数据子集, 不训练图模型 |
| G7 | 中高 | P1 | correction + sync + ack 全链路 smoke 通过 | 检查 outbox、trust posterior 初始化和云端校正接口 |

P0-A0/P0-B smoke 是早期风险隔离，不替代 Final Gate。若早期验证未通过，必须在 `docs/REVISION_LOG.md` 记录失败原因、修复动作和重新生成的 hash。

### 0.8 fallback 与动态调整边界

允许的工程 fallback:
- 数据目录解析、文件名映射、公开数据版本差异记录。
- 字段格式转换、时间戳单位转换、缺失非标签字段写入空值或 unknown。
- 日志字段补充、batch size/路径/脚本参数修正、进程异常恢复。
- 不改变标签、profile、metric 口径的 CityFlow 子集扩展。CityFlow 子集扩展只允许在 G-DATA 阶段、Final Gate 前完成，并重新冻结 split_hash；P2 后不得扩展或替换 final test 子集。

禁止的结果影响型 fallback:
- 改变 final test split、final sample_ids 或 split_hash。
- 改变 final label、global_decision_gt、conflict_gt 或 relation_group 难度来适配模型结果。
- 改变 frozen network/workload profile、business deadline、correction/sync deadline 或 gate 统计口径。
- 使用 final test 结果训练、校准或选择蒸馏/planner/policy/graph/trust 模型。
- Final Gate algorithm_failure 后调参重跑。

所有 fallback 必须写入 `manifest.json` 的 `fallback_events` 与 `docs/REVISION_LOG.md`。影响数据 hash、模型 hash 或 metric script hash 的 fallback 必须重新生成对应 hash。

---

## 1. 硬件环境、模型角色、基线边界

设备: Dell T640 (4×RTX 3090, 24GB) @ 192.168.4.178 + Blade (CPU-only, ≥16GB RAM) @ 192.168.4.174

| GPU | 服务 | 模型 |
|-----|------|------|
| 0 | Cloud LLM | Qwen2.5-14B-AWQ (vLLM, TP=1) |
| 1 | High-Edge LLM | Qwen2.5-7B-AWQ (vLLM, TP=1) |
| 2 | SFT + planner/policy/graph 训练 / BCC Arbitrator | 按需 |
| 3 | Teacher data gen / Quant | 按需 |

| 角色 | 模型 | G1/G3 | G2/G4/G5 | G6/G7 |
|------|------|-------|-----------|-------|
| 云端全量教师 | Qwen2.5-14B-AWQ | G1分母 | cloud 路径 | 全局校正 |
| 高算力边缘 | Qwen2.5-7B-AWQ | 不参与 | 异构节点 | 可选 |
| 低算力边侧 | **DB4AI-Edge-1.5B-KD-INT4** | **唯一对象** | edge/coop 路径 | 本地决策 |

已复用: 14B FP16 `/home/qhq/serverless_llm_experiment/models/Qwen--Qwen2.5-14B-Instruct/`, 7B FP16 同上。学生初始化权重: Qwen2.5-1.5B-Instruct (~3GB)。

Cloud-only-summary 基线: 共享 evidence candidates + summary_generation + 相同时延统计。不使用 IBEP 证据规划、TARS-SSM 状态建模、CPR 路由、弱网自治补偿、ST-HGCI 冲突推理、BTCF 节点可信度校准反馈。成果2贡献由消融证明。

---

## 2. 指标计算口径

### 2.1 G1: 能力保持率

```
Math_ratio = Edge_GSM8K_Accuracy / Cloud_GSM8K_Accuracy
Code_ratio = Edge_HumanEval_pass@1 / Cloud_HumanEval_pass@1
NLP_ratio  = Edge_NLPScore / Cloud_NLPScore
  NLPScore = 0.5 × MMLU_accuracy + 0.5 × CMMLU_accuracy

用于 Overall_R_cap 时使用 capped ratio (截断不超过 1.0):
  Math_ratio_cap = min(Math_ratio, 1.0)
  Code_ratio_cap = min(Code_ratio, 1.0)
  NLP_ratio_cap  = min(NLP_ratio, 1.0)
  Overall_R_cap = (Math_ratio_cap + Code_ratio_cap + NLP_ratio_cap) / 3

G1 hard gate (四项全部通过):
  Math_ratio ≥ 80%, Code_ratio ≥ 80%, NLP_ratio ≥ 80%, Overall_R_cap ≥ 80%

原始未截断 ratio 单独报告。

HumanEval: 统一 Python sandbox, timeout=10s/test, 禁止网络访问, pass@1 单候选。
MMLU/CMMLU: 统一 multiple-choice parser, 无法解析为 A/B/C/D 记为错误,
            Edge 与 Cloud 使用相同 prompt/parser/采样参数。
```

### 2.2 G2: TTFT 降低比例

```
P95_TTFT_reduction(profile) = 1 - P95_TTFT_DB4AI(profile) / P95_TTFT_CloudOnlySummary(profile)

通过条件: high_delay≥75%, low_bandwidth≥75%, high_loss≥75%, weak_avg≥75%
同时报告 mean_TTFT_reduction 作为支撑指标。

decision_ttft_ms: 请求进入系统 → 第一个可解析业务决策字段输出的时间
  (JSON: label/action 字段可解析时计入)。所有方案使用统一 decision parser。
```

### 2.3 G3: 推理内存

```
P95_RSS_MB_decimal: 边侧 1.5B INT4 模型的稳态推理 RSS P95 (decimal MB)
P95_RSS_MiB_binary: 同一窗口下的 binary MiB 口径，仅用于辅助报告

测量条件:
  warmup_requests = 20, measure_requests ≥ 100, batch_size = 1
  采样周期 100ms, 只统计模型加载完成后的稳定推理窗口(不含冷启动)
  peak_RSS_MB 记录但不作为 hard gate

G3-final: P95_RSS_MB_decimal ≤ 1500 MB
           同时报告 P95_RSS_MiB_binary, 参考线 ≤1431 MiB
G3-dev:   P95_RSS_MB_decimal ≤ 1450 MB
测量: cgroup memory.stat + psutil.Process.memory_info().rss
```

### 2.4 G4: 弱网保持率

```
business_deadline_ms = 200ms

valid_business_decision: 在 business_deadline_ms 内输出可解析、可执行、非空 action 的决策

business_retention(profile) = valid_business_decisions(profile) / total_tasks(profile)

通过条件 (每个 profile 独立):
  high_delay/low_bandwidth/high_loss/short_disconnect ≥ 90%, combined ≥ 90%

coop: provisional 在 200ms 内输出→计入保持; correction_deadline=1000ms; sync_deadline=2000ms
存在 provisional 的 profile: provisional_correction_rate ≥ 90%
```

### 2.5 G5: 端到端时延

```
E2E_latency = business_decision_output_ts - task_received_ts
包含: edge_preprocess, evidence planning, summary_generation, LLM inference, path routing, correction_waiting

G5-final: P95_E2E industrial≤0.2s, traffic≤0.2s, combined≤0.2s
G5-dev:   P95_E2E ≤0.18s

无效业务决策在 G4 中计为失败;
在 G5 中计为 timeout_ms = max_observation_timeout_ms (固定为 1000ms),
按 1000ms 计入 P95_E2E 计算。

coop: business_e2e_ms + correction_latency_ms 分开记录。
```

### 2.6 G6: 后校正决策冲突比例

```
post_correction_conflict_ratio =
  unresolved_or_inconsistent_relation_groups / total_evaluated_relation_groups

total_evaluated_relation_groups:
  final test split 中所有重叠感知区域或关联任务组, 不仅限于 conflict_gt=true 的真实冲突组

unresolved_or_inconsistent_relation_groups:
  在 correction_deadline 与 sync_deadline 后仍存在 event_type/risk_attr/action 不一致,
  或未能形成唯一全局决策的关联任务组

G6-final: ≤5% | G6-dev: ≤4%

支撑指标:
  gt_unresolved_conflict_ratio = unresolved_gt_conflict_groups / total_gt_conflict_groups
  false_positive_conflict_ratio = false_positive_conflicts / total_evaluated_relation_groups
  若 final test 中 total_gt_conflict_groups < 50 或 total_evaluated_relation_groups < 200 → G-DATA 不通过
  写入 conflict_semantics_metrics.csv 和 support_metrics.csv
```

### 2.7 G7: 冲突解决成功率

```
gt_conflict_resolution_success_rate = successfully_resolved_gt_conflict_groups / total_gt_conflict_groups

successfully_resolved: correction_deadline(1000ms)内校正 + sync_deadline(2000ms)内ack
  + final_decision 与 conflict_gt_manifest.global_decision_gt 一致
  (比较 event_type, risk_attr, action; confidence 不作为一致性硬条件)
  + 反馈已写入 trust_posterior 或 outbox

G7-final: ≥90% | G7-dev: ≥92%
漏检的真实冲突计入分母、不计入分子
```

### 2.8 功能 gate 定义

**G-DATA:** 所有数据集下载 + 版本识别 + split 文件生成 + `dataset_manifest.json`/`manifest.json`/`conflict_gt_manifest.json` 中不存在占位符(`"actual"`, `"实际值"`, `"..."`, `TBD`, `TODO`, `placeholder`, null hash, empty split_hash, empty final_gate_sample_ids_hash) + split_hash 生成 + global_leakage_check 全部通过 + `conflict_gt_manifest.json` 已生成 + hash 已写入 + `conflict_gt_audit.csv`/`conflict_gt_sample_audit.json` 已生成 + validation 与 final test 分别满足 `conflict_group_count≥50`, `relation_edge_count≥200`, `relation_group_count≥200` + manifest 中不存在空 conflict_group_id/event_id/global_decision_gt

**G-DB:** KWDB Docker 启动 + schema 创建 + 写入/查询/CSV 导出成功 + outbox 表创建

**G-CLOUD:** 14B-AWQ serving `/health` 200 + smoke test 首 token <2s + model_hash/prompt_hash 记录

**G-KD-TRACE:** Student-Base → CEDD-Structured → CEDD-Repair → DB4AI-Edge-1.5B-KD-INT4 完整链路 + teacher_trace/student_probe/counterfactual_repair/calibration/quant_behavior hash 全部生成

**G-Support-Perception:** 工业视觉与交通感知支撑指标生成成功。工业场景至少报告 image-level AUROC、F1、Recall、False Alarm Rate；若存在像素标注则报告 pixel-level AUROC/IoU。交通场景至少报告 mAP、MOTA/IDF1 或 event F1、duplicate alert rate。结果写入 perception_results.csv, 并与五方案对比。

### 2.9 200ms 业务决策输出协议

```
business_action_header:
  required fields = task_id, event_type, risk_attr, action, confidence, is_provisional, selected_path

200ms 内的 hard requirement:
  输出可解析 business_action_header, 不要求输出完整自然语言解释或长 rationale

decision_ttft_ms:
  task_received_ts → business_action_header 首次被统一 parser 成功解析的时间

business_e2e_ms:
  task_received_ts → business_action_header 被确认可执行的时间

async fields:
  explanation, long_rationale, cloud_correction, cross_node_consistency_update
  与 business_action_header 分开记录, 不用于 200ms 首响应判定
```

实现约束:
- G4/G5 hard path 仅统计 business_action_header 的生成、解析与确认时延；自然语言解释、长理由、云端校正和跨节点一致性更新不计入 200ms 首响应路径。
- 边侧模型必须优先生成 JSON/grammar constrained 的短字段 action header。
- `max_new_tokens`、stop token、schema parser 和 decision parser 在 Final Gate 前冻结并写入 manifest。
- weak_autonomy 路径允许先输出 provisional decision；云端 correction、trust posterior 更新和 outbox sync 按 G4/G7 口径异步统计。
- 若采用 replay/precomputed perception feature，必须在 latency_breakdown.csv 中标注 `feature_mode=replay|precomputed|online`，不得把离线特征时延伪装成在线端到端时延。
- Final Gate 主结果采用 `feature_mode=replay` 的任务重放口径，online_smoke 用于证明感知模块可运行；若报告在线端到端结果，必须单独列出，不与 replay 主结果混合。

### 2.10 baseline 公平性协议

五方案对比必须使用相同 task set、split、network profile、workload profile、decision parser、business deadline 和 metric scripts:

| 方案 | 允许能力 | 禁止能力 |
|------|---------|---------|
| cloud_only_summary | 云端 14B + 共享 evidence candidates + summary_generation | IBEP, TARS-SSM, CPR, weak_autonomy, ST-HGCI, BTCF |
| edge_only | 边侧 1.5B INT4 + 同一 decision parser | 云端复核、云端校正、图一致性校正 |
| static_split | 固定路径划分 | 状态预测、约束策略路由 |
| load_based | 负载/队列驱动路径选择 | TARS-SSM 后果预测、CPR 约束策略 |
| db4ai_edgeserve | 完整 CEDD+IBEP+TARS-SSM+CPR+ST-HGCI+BTCF | Final Gate 中调参 |

为避免 strawman 质疑，Final 报告额外提供 `cloud_only_full_context` 支撑结果: 使用云端 14B 和完整可用证据，但不进入五方案 hard comparison，仅用于说明 cloud-only-summary 不是人为削弱基线。所有 baseline 均报告 accuracy/latency/communication/resource 四类指标。

---

## 3. 数据集来源、观测量与 profile

### 3.1 数据集矩阵

| 数据集 | 公开规模 | Final test 规模 | seed | 进入蒸馏 | Final Gate |
|--------|---------|--------------|------|---------|-----------|
| GSM8K | train 7473 / test 1319 | 500 | 42 | train | 是 |
| HumanEval | 164 全量 | 164 | — | 不进入 | 是 |
| MMLU | 57 科目 | 1000 (科目分层) | 42 | train | 是 |
| CMMLU | 67 科目 | 1000 (科目分层) | 42 | 非final train/synthetic | 是 |
| MVTec AD | 15类, train 3629(仅正常) / test 1725 | 1725 (官方test全量) | — | train 正常样本 | 是 |
| NEU-DET | 6类×300=1800 | 360 (类别分层) | 42 | train(70%) | 是 |
| CityFlow | CityFlow-Original: 3.25h, 40摄像头, 10路口, 5场景(3train+2test), 666 vehicle ID | 构造脚本统计 | 42 | train场景 | 是 |
| UA-DETRAC | 100视频, >140k帧, 含车辆框/类别/遮挡/截断 | 构造脚本统计 | 42 | 不进入关联图训练 | 是 |

MVTec AD validation 方案 A: 官方 test 不进 validation。validation 使用 train 正常样本固定子集 + NEU-DET val 缺陷样本 + 派生非 final 异常样本。

CMMLU: final sample_ids 写入 manifest。不允许 final evaluation 子集进入蒸馏。synthetic 记录 prompt_hash/generation_seed/teacher_model/sample_hash。

#### CityFlow 子集构造配置

```
preferred_dataset_version: CityFlow-Original
若实际下载为 CityFlowV2 → dataset_manifest.json 记录 dataset_version = CityFlowV2
camera_count: 6
intersection_count: 2
replay_duration_s: 300
replay_tick_s: 1 (annotation-driven replay)
relation_sources: [vehicle_id, camera_id, region_id, timestamp, derived_event]

split: 若官方 test 标签不可用 → 按 scene/camera/time_window 冻结划分 train/val/test
       split_hash 写入 dataset_manifest.json
       validation 与 final test 的 relation_edge_count, relation_group_count 和 conflict_group_count 分别统计
```

### 3.2 dataset_manifest.json

```json
{
  "manifest_version": "1.0",
  "created_by": "scripts/setup_datasets.sh",
  "created_ts": "float",
  "sampling_seed": 42,
  "datasets": [
    {
      "dataset_name": "GSM8K",
      "dataset_version": "实际值",
      "official_scale": "train 7473 / test 1319",
      "used_train_count": "实际值",
      "used_validation_count": "实际值",
      "used_test_count": 500,
      "final_gate_sample_ids_hash": "实际值",
      "split_hash": "实际值",
      "teacher_generated_train_count": "实际值",
      "leakage_check_pass": true
    }
  ],
  "scu_support": {
    "scu_sample_count": 300,
    "scu_sample_hash": "实际值",
    "scu_source": "independent_support_sources",
    "used_for_training": false,
    "used_for_validation": false,
    "used_for_final_support": true
  },
  "global_leakage_check": {
    "test_in_distill": false,
    "test_in_planner_train": false,
    "test_in_policy_train": false,
    "test_in_graph_train": false,
    "test_in_calibration": false,
    "test_in_validation": false
  }
}
```

每个数据集均含 `used_train_count`, `used_validation_count`, `used_test_count`, `split_hash`, `final_gate_sample_ids_hash`, `leakage_check_pass`。CityFlow 额外含 `relation_node_count`, `relation_edge_count`, `relation_group_count`, `conflict_group_count`, `vehicle_id_count`。MVTec AD 额外标注官方 test 是否全量进入 Final Gate。NEU-DET 额外标注每类 train/val/test 数量。

G-DATA 通过后: `dataset_manifest.json`, `manifest.json`, `conflict_gt_manifest.json` 中不得存在 `"actual"`, `"实际值"`, `"..."`, `TBD`, `TODO`, `placeholder`, null hash, empty split_hash, empty final_gate_sample_ids_hash。

### 3.3 观测量、学习状态与日志来源表

| # | 状态/表示 | 关联成果 | 来源 | 产物/用途 | 写入日志 |
|---|----------|---------|------|-----------|---------|
| 1 | teacher_decision_trace | 1 | 14B 教师 + 标注 | 结构化决策蒸馏标签 | semantic_distill_trace.csv |
| 2 | student_probe_trace | 1 | 1.5B 学生在 train replay 上的输出 | 学生在环修复样本挖掘; validation probe 只用于诊断不进入训练 | student_probe_trace.csv |
| 3 | counterfactual_boundary_trace | 1 | 错误样本、低置信样本、冲突样本自动派生 | 工业缺陷边界/交通重复告警/风险等级临界样本 | counterfactual_trace.csv |
| 4 | quant_behavior_trace | 1 | FP16/INT4 学生模型对同一任务的输出对齐 | 量化行为保持与 G3 复核 | quant_behavior_trace.csv |
| 5 | evidence_candidate_set | 2 | 感知模块、标注、教师 trace、历史决策 | IBEP 候选证据集合 | evidence_planner_trace.csv |
| 6 | evidence_plan | 2 | IBEP 证据规划器 | selected_evidence_ids, summary, evidence_sufficiency, review_intent | evidence_planner_trace.csv |
| 7 | calibrated_risk_state | 2,4,6 | 校准集 + 运行反馈 | 边侧自决/协同/云端复核风险估计 | calibration_trace.csv |
| 8 | runtime_observation | 3,4 | StateCache fast/slow + network_snapshot | 网络、队列、模型健康、上下文完整性、outbox、任务风险 | runtime_state_trace.csv |
| 9 | runtime_latent_state | 3 | TARS-SSM | h_t 运行状态表示与路径后果预测 | readiness_metrics.csv |
| 10 | policy_action_trace | 4 | CPR 路由器 | edge/cloud/coop/weak_autonomy 路径选择与预测后果 | policy_router_trace.csv |
| 11 | perception_metrics | 1,2,4 | 工业视觉、交通检测/跟踪模块 | AUROC/F1/mAP/MOTA/IDF1/event F1 | perception_results.csv |
| 12 | decision_tuple | 1,4,5,6 | 边侧/云端/协同路径输出 | event_type, risk_attr, action, confidence, review_intent | decision_tuple_trace.csv |
| 13 | spatiotemporal_graph_state | 5 | 决策元组、对象轨迹、区域、事件、时间窗口 | ST-HGCI 图输入 | relation_graph_trace.csv |
| 14 | conflict_inference | 5 | ST-HGCI 图模型 | 关联组、冲突类型、全局决策分布 | conflict_semantics_metrics.csv |
| 15 | trust_posterior_state | 6 | 冲突校正、ack、复发记录、节点历史 | 节点可信度后验与复核优先级 | trust_calibration_trace.csv |
| 16 | communication_resource_state | 2,4,6 | runtime + DB + network + LLM serving | 上传字节、云端请求率、token、RSS、CPU、KWDB 写入吞吐 | communication_results.csv, resource_results.csv |

#### 3.3.1 观测量构造细则

**对象与轨迹表示:**
- CityFlow 优先使用官方 vehicle_id/camera_id/timestamp 作为轨迹监督信号。
- 缺少 vehicle_id 或跨摄像头 identity 不可靠时，不使用人工加权 fallback。系统从检测框序列、局部图像 crop、事件文本、时间间隔和区域上下文学习 tracklet embedding，由 ST-HGCI 在图中推断潜在关联。
- UA-DETRAC 仅声明同视频内 track_id 或 derived_tracklet；不把同视频轨迹人工外推为跨摄像头 identity。

**事件链与上下文表示:**
- 同一 object/region/time window 内的事件按时间构成 event sequence，由轻量序列编码器生成 event_context_embedding。
- 交通 event_type 包含 normal/slow/congestion/stop/lane_change/abnormal_queue/duplicate_alert_candidate。
- 工业 event_type 包含 normal/defect_candidate/defect_confirmed/uncertain_inspect/reject/pass。

**证据与风险表示:**
- evidence_candidate_set 保留证据来源、时间、模态、内容和解析状态，但完整方案不使用人工特征权重。
- calibrated_risk_state 由 validation calibration set 和运行反馈生成，用于选择性自决/协同/复核；固定人工复核阈值仅出现在 baseline。

### 3.4 研究机制建模

**成果1: CEDD 云边闭环语义蒸馏**
```
z_s, y_s = EdgeStudent_theta(x, S)
z_t, y_t = CloudTeacher(x, S_full)
y = {event_type, risk_attr, action, confidence, review_intent}

theta* 位于多目标 Pareto 前沿:
  minimize (D_decision(y_s,y_t), D_state(z_s,z_t), calibration_error, quantized_behavior_drift)
subject to:
  Math_ratio/Code_ratio/NLP_ratio/Overall_R_cap 满足 G1
  P95_RSS_MB_decimal 满足 G3
  structured_parse_success 与 decision_tuple 完整性通过功能检查
```

CEDD 包含四类训练 trace: 结构化决策蒸馏、过程压缩蒸馏、学生在环修复、反事实边界蒸馏。学生在环修复只从 train split 的 student_probe_trace 自动挖掘错误、低置信、冲突和量化漂移样本，再由 14B 教师复核生成 repair trace；validation probe 仅用于诊断和模型选择，不进入训练数据。Final Gate 前冻结 teacher prompt、student adapter、repair 数据 hash、calibration snapshot 与 INT4 量化行为报告。

**成果2: IBEP 信息瓶颈证据规划**
```
S* = EvidencePlanner_phi(x, E_candidates, runtime_latent_state)

目标: 在候选证据中生成最小充分证据子集, 使边侧结构化决策风险受控
  minimize Pareto tuple: (decision_risk(Y|x,S), evidence_cost(S), communication_bytes(S))
subject to:
  sufficiency(Y|x,S) 通过教师/标注一致性校验
  runtime budget 来自当前 StateCache 与业务 SLA
  calibrated_risk 满足 G4/G5 相关业务要求
```

IBEP 输出 selected_evidence_ids、summary_text、evidence_sufficiency、review_intent。固定摘要、固定 token budget、人工 review threshold 与无风险校准只作为消融。

**成果3: TARS-SSM 多源运行状态空间模型**
```
h_t = F_theta(h_{t-1}, o_t)
p(latency, failure, accuracy_loss, sync_delay | h_t, action) = G_theta(h_t, action)
```

o_t 来自 network_snapshot、queue_depth、model_health、context_completeness、outbox_backlog、task_risk、recent_success/failure feedback。TARS-SSM 不输出人工 readiness 分数，而是预测不同路径在当前状态下的时延、失败、准确率损失和同步延迟分布，供 CPR 路由器使用。

**成果4: CPR 约束策略路由与弱网自治**
```
action_t = pi_theta(h_t), action_t ∈ {edge_only, cloud_only, coop, weak_autonomy}

pi* 选择满足约束的 Pareto 最优路径:
  minimize (P95_TTFT, P95_E2E, communication_bytes, cloud_request_rate)
subject to:
  business_retention 满足 G4
  P95_E2E 满足 G5
  error_rate 不劣于校准集允许风险
  P95_RSS_MB_decimal 满足 G3
```

CPR 使用离线 trace 训练 + validation 校准。弱网自治路径在 200ms SLA 内输出 provisional decision，网络恢复后由云端校正并通过 outbox 同步。人工 static split、load-based、network-only、boolean readiness 只作为 baseline。

**成果5: ST-HGCI 时空异构图冲突推理**
```
G_t = (V_decision, V_object, V_region, V_event,
       E_spatial, E_temporal, E_tracklet, E_semantic, E_feedback)
p(C, Y_global | G_t, D_t) = GraphModel_theta(G_t, D_t)
Y_global* = MAP_Y p(Y_global | G_t, D_t)
```

ST-HGCI 联合推断关联任务组、冲突类型和全局一致决策。图边由轨迹、区域、时间、事件语义和反馈 trace 学习得到，不使用人工 relation_strength、conflict_strength 或相似度阈值。simple vote、text match、manual similarity、no graph 只作为消融。

**成果6: BTCF 节点可信度后验校准与闭环反馈**
```
p(Theta_node | H_t) ∝ p(H_t | Theta_node) p(Theta_node)
p(Y_global | D_t, G_t, Theta_node) = calibrated_graph_decision
```

Theta_node 表示节点在任务类型、区域、网络状态、模型版本下的可靠性和偏差状态；H_t 包含历史决策、冲突、云端校正、ack、复发和同步结果。BTCF 不使用人工学习率更新可信度，而是用后验校准影响复核优先级、全局决策和模型/策略反馈。每次 Final Gate 从 frozen initial_trust_posterior_snapshot 开始，在线只依据真实运行日志更新后验状态。

### 3.5 网络 profile、workload profile、train/val/test 隔离

```yaml
# configs/network_profiles.yaml — 冻结
normal:         {delay_ms:0, jitter_ms:0, loss_pct:0, bandwidth_mbps:null, disconnect:false}
high_delay:     {delay_ms:100, jitter_ms:20, loss_pct:0, bandwidth_mbps:null, disconnect:false}
low_bandwidth:  {delay_ms:0, jitter_ms:0, loss_pct:0, bandwidth_mbps:1, disconnect:false}
high_loss:      {delay_ms:0, jitter_ms:0, loss_pct:5, bandwidth_mbps:null, disconnect:false}
short_disconnect: {delay_ms:0, jitter_ms:0, loss_pct:0, bandwidth_mbps:null, disconnect:true, down_s:5, period_s:60}

# configs/workload_profiles.yaml — 冻结
stable:   {arrival:fixed_rate, rates:[1,2,5,10], duration_s:300}
burst:    {arrival:burst, base_rate:2, burst_factor:3, burst_duration_s:30, total_duration_s:300}
replay:   {arrival:dataset_timestamp, preserve_order:true, duration_s:300}
scale:    {edge_nodes:[2,4,8], relation_ratio:[0.10,0.25,0.40], duration_s:300}
```

上述 profile 仅用于实验复现和压力测试，不作为在线路径策略、证据规划或冲突推理模型的人工规则输入。

train/val/test 隔离硬规则:
- train: GSM8K train · MMLU train · CMMLU 非final train/synthetic · MVTec AD train(仅正常) · NEU-DET train(70%) · CityFlow train
- validation: MMLU val · MVTec AD train正常子集+NEU-DET val派生 · CityFlow val
- test (仅 final gate, seed=42): GSM8K 500 · HumanEval 164 · MMLU 1000 · CMMLU 1000 · MVTec AD test 1725 · NEU-DET test 360 · CityFlow test · UA-DETRAC test
- test 不进入蒸馏/SFT/planner/policy/graph/calibration 训练或 validation 调参

---

## 4. Schema 与冲突 ground truth 协议

### 4.1 八个 Schema

所有 schema 统一含: `schema_version`, `created_by`, `split`, `source_dataset`, `sample_hash`, `created_ts`

**semantic_distill_schema.json:** evidence_items[] + decision_tuple (object_state, event_type, risk_attr, action, confidence, review_intent, short_rationale) + teacher_trace + student_probe_trace + repair_trace + quant_behavior_trace

**evidence_planner_schema.json:** evidence_candidates[] + evidence_plan (selected_evidence_ids, summary_text, evidence_sufficiency, review_intent) + planner_model_hash + calibration_snapshot_hash

**runtime_state_schema.json:** task_id, edge_node_id, network_snapshot, queue_state, model_health, context_state, outbox_state, task_risk, runtime_latent_state_hash, predicted_path_outcomes

**policy_action_schema.json:** task_id, runtime_state_hash, selected_path(edge_only|cloud_only|coop|weak_autonomy), predicted_outcome_distribution, actual_outcome, policy_snapshot_hash

**decision_tuple_schema.json:** decision_tuple_id, task_id, scene, dataset, edge_node_id, decision_ts, object_id, region_id, event_id, relation_group_id, conflict_group_id, network_profile, workload_profile, object_state, event_type, risk_attr, action, confidence, review_intent, is_provisional, selected_path, source

**relation_graph_schema.json:** graph_id, nodes[] (decision/object/region/event), edges[] (spatial/temporal/tracklet/semantic/feedback), graph_model_hash, graph_input_hash, source_split

**conflict_inference_schema.json:** relation_group_id, conflict_group_id, conflict_type_distribution, global_decision_distribution, final_global_decision, conflict_gt, label_source, conflict_gt_manifest_hash, graph_model_hash, event_id

**trust_posterior_schema.json:** edge_node_id, task_type (industrial_defect|traffic_event), posterior_state, posterior_snapshot_hash, correction_history_hash, ack_state, recurrence_state, network_profile, workload_profile

### 4.2 conflict_gt_manifest.json

```json
{
  "manifest_version": "1.0",
  "created_by": "scripts/build_conflict_gt.py",
  "created_ts": "float",
  "datasets": ["CityFlow", "MVTec AD", "NEU-DET"],
  "split": "train|validation|test|all",
  "conflict_groups": [{
    "conflict_group_id": "str",
    "event_id": "str",
    "node_ids": ["str"],
    "conflict_type_gt": "class_conflict|risk_conflict|action_conflict|duplicate_alert|none",
    "global_decision_gt": {
      "event_type": "str",
      "risk_attr": "low|medium|high",
      "action": "pass|reject|inspect|alert|ignore|upload"
    },
    "label_source": "cityflow_annotation|derived_rule|manual_review|industrial_label",
    "source_dataset": "str",
    "time_window_id": "str"
  }],
  "conflict_group_count": "int",
  "conflict_type_distribution": "dict",
  "manifest_hash": "str"
}
```

### 4.3 冲突 ground truth 构造协议

1. conflict_gt 在系统推理前由构造脚本生成
2. 不由 DB4AI-EdgeServe 的冲突检测结果反推
3. 不由被评估的边侧模型输出决定
4. traffic conflict_gt 优先由 CityFlow vehicle_id/camera_id/timestamp/derived_event 派生
5. industrial conflict_gt 由标签、区域、时间窗口和构造事件组派生
6. 每个 conflict_group_id 记录 label_source
7. 构造脚本输出 conflict_gt_manifest.json
8. conflict_gt_manifest.json hash 写入 dataset_manifest.json
9. G6 final 以 final test 中所有重叠感知区域或关联任务组为分母；G7 final 以 conflict_gt=true 的真实冲突组为分母
10. CityFlow traffic/combined 是 G6/G7 主评测来源；industrial conflict_gt 进入 combined 和支撑分析，若 final relation_group_count 足够可单独报告

---

## 5. 研究机制与实验验证映射

| 成果 | 阶段 | 核心消融（正文） | 附录消融 | 核心指标 | 支撑指标 | 关键日志 |
|------|------|----------------|---------|---------|---------|---------|
| 成果1 CEDD | P0-A | Plain SFT/KD vs 云边闭环语义蒸馏 | 无学生在环修复, 无反事实边界蒸馏, 无量化行为保持 | G1,G3 | 结构化解析成功率, 风险/动作一致率, 量化漂移, 复核意图校准误差 | semantic_distill_trace.csv, student_probe_trace.csv, counterfactual_trace.csv, quant_behavior_trace.csv |
| 成果2 IBEP | P0-B | 固定摘要 vs 信息瓶颈证据规划 | 固定 token budget, 人工 review threshold, 无风险校准 | G2,G5 | 证据充分性, 字段保留完整率, 风险证据保留率, 云端复核有效率 | evidence_planner_trace.csv, calibration_trace.csv |
| 成果3 TARS-SSM | P0-B→P1 | 固定布尔检查 vs 多源运行状态空间模型 | 仅网络状态, 仅负载状态, 无历史反馈 | G2,G5 | 路径后果预测误差, 状态缺失识别准确率, 调度失败比例 | runtime_state_trace.csv, readiness_metrics.csv |
| 成果4 CPR | P0-B,P1-Reg | Load-based vs 约束策略路由+弱网自治 | Static-Split, Network-only, 无弱网自治 | G2,G4,G5 | 路径选择比例, 云端请求率, 上传字节数, provisional 校正率 | policy_router_trace.csv, weak_autonomy_trace.csv, communication_results.csv |
| 成果5 ST-HGCI | P1 | 简单投票 vs 时空异构图冲突推理 | 文本匹配, 人工相似度, 无图结构 | G6 | 关联识别 F1, 冲突检测召回率, 冲突类型识别准确率, 全局决策一致率 | relation_graph_trace.csv, conflict_semantics_metrics.csv |
| 成果6 BTCF | P1 | 云端单次校正 vs 节点可信度后验校准+反馈 | 云端投票, 无反馈, 固定节点可信度 | G7,G6 | 重复冲突下降比例, 校准后复发率, ack 成功率, 边缘反馈生效率 | trust_calibration_trace.csv, feedback_effect_metrics.csv |

消融仅在 validation split 执行。正文 1组核心消融/成果。成果1-2: normal; 成果3: normal+high_delay; 成果4: 四类网络; 成果5-6: traffic+combined。

SCU_support: 300题, independent_support_sources 生成, 不与 distill_dataset.jsonl 重合, `used_for_training=false, used_for_validation=false, used_for_final_support=true`。scu_sample_hash 写入 dataset_manifest.json。

---

## 6. 数据库、StateCache 与语义证据规划器

### 6.1 StateCache 分层

```
fast (20-50ms刷新):  network_status, queue_depth, edge_service_health, outbox_backlog
  → TARS-SSM runtime_observation 来源
  字段: RTT, bandwidth, loss_rate, disconnect_flag, pending_task_count, server_healthy

slow (500-1000ms刷新): model_version, policy_version, conflict_history, trust_posterior
  → TARS-SSM context/trust observation 来源
  字段: active_model_version, active_policy_version, related_task_status,
        conflict_risk_state, posterior_snapshot_hash
```

CPR 调度关键路径只读 StateCache 与 TARS-SSM cached prediction，不执行 DB I/O。调度延迟目标 <1ms。异步写回不计入。

### 6.2 KWDB Docker 部署

```bash
docker pull kwdb/kwdb:2.2.0
docker run -d --name kwdb-cloud --privileged --ulimit memlock=-1 \
  -p 26257:26257 -p 8080:8080 \
  -v /home/qhq/DB4AI-EdgeServe/data/kwdb:/kaiwudb/deploy/kaiwudb-container \
  -w /kaiwudb/bin kwdb/kwdb:2.2.0 \
  ./kwbase start-single-node --insecure --listen-addr=0.0.0.0:26257 \
    --http-addr=0.0.0.0:8080 --store=/kaiwudb/deploy/kaiwudb-container
docker exec -i kwdb-cloud ./kwbase sql --insecure --host=127.0.0.1 < sql/cloud_schema.sql
```

边缘侧: PostgreSQL Docker 或 SQLite（outbox 表），sync agent 自实现。

### 6.3 IBEP 语义证据规划器

```
文件: models/ibep_evidence_planner.pt
输入: evidence_candidate_set + task_context + runtime_latent_state + calibrated_risk_state
输出: selected_evidence_ids, summary_text, evidence_sufficiency, review_intent
训练: 14B 教师、标注数据和 student_probe_trace 在 train split 生成 planner_train.jsonl
      validation split 仅用于校准与模型选择; 运行时不调用 14B 教师
manifest: planner_model_path, planner_model_hash, planner_train_split_hash,
          planner_validation_metric, calibration_snapshot_hash, planner_freeze_ts
```

---

## 7. P0-A：基础设施与成果1实验

### 7.0 P0-A0 Preflight

```bash
# 1. 基础 INT4 内存与短字段输出烟测
python3 scripts/preflight_edge_runtime.py \
    --student_init Qwen/Qwen2.5-1.5B-Instruct \
    --quant_format Q4_K_M \
    --measure_requests 100 \
    --output reports/preflight_edge_runtime.json

# 2. 100题能力与结构化 action header smoke
python3 scripts/preflight_capability_smoke.py \
    --teacher_url http://localhost:8000 \
    --student_init Qwen/Qwen2.5-1.5B-Instruct \
    --sample_count 100 \
    --output reports/preflight_capability_smoke.json

# 3. 推理后端对比: RSS、TTFT、business_action_header parse success
python3 scripts/preflight_backend_compare.py \
    --backends llama_cpp,vllm,transformers \
    --output reports/preflight_backend_compare.csv
```

P0-A0 通过条件:
- base/student INT4 稳态 RSS 明显低于 G3-final 1500 MB(decimal), 且记录 KV cache/context 设置。
- business_action_header parse success ≥95% (smoke), P95 action_header_latency ≤250ms 且 P50 ≤200ms；未达到时不得进入 P0-B 完整系统调试。
- G1 smoke 不作为 hard gate, 但若 Math/Code/NLP 任一维度明显低于预期, 必须先补 CEDD 数据与训练策略再进入 P0-A 完整训练。
- 输出 `preflight_report_hash` 写入 manifest.json。

### 7.1 G-DATA

```bash
bash scripts/setup_datasets.sh
python3 scripts/validate_splits.py --check_leakage
python3 scripts/build_conflict_gt.py \
    --dataset_manifest dataset_manifest.json \
    --output conflict_gt_manifest.json \
    --audit_csv reports/conflict_gt_audit.csv \
    --sample_audit_json reports/conflict_gt_sample_audit.json
# G-DATA 通过条件: manifest 无占位符(actual/实际值/TBD/TODO/placeholder/null hash/empty hash);
#   validation 与 final test 分别满足 conflict_group_count>=50, relation_edge_count>=200, relation_group_count>=200;
#   conflict_gt_manifest 无空字段; conflict_gt_audit.csv 记录 label_source/type/group 分布;
#   sample_audit_json 随机抽样检查 conflict groups, hash 写入 manifest
```

### 7.2 G-DB

```bash
docker pull kwdb/kwdb:2.2.0
docker run -d --name kwdb-cloud --privileged --ulimit memlock=-1 \
  -p 26257:26257 -p 8080:8080 \
  -v /home/qhq/DB4AI-EdgeServe/data/kwdb:/kaiwudb/deploy/kaiwudb-container \
  -w /kaiwudb/bin kwdb/kwdb:2.2.0 \
  ./kwbase start-single-node --insecure --listen-addr=0.0.0.0:26257 --http-addr=0.0.0.0:8080
docker exec -i kwdb-cloud ./kwbase sql --insecure --host=127.0.0.1 < sql/cloud_schema.sql
python3 scripts/verify_gate_db.py
```

### 7.3 G-CLOUD

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --quantization awq \
  --max-model-len 4096 --gpu-memory-utilization 0.85 --tensor-parallel-size 1 --port 8000 &
python3 scripts/verify_gate_cloud.py
```

### 7.4 成果1: CEDD 云边闭环语义蒸馏

```bash
# 教师结构化决策与短证据链生成
python3 model_compression/generate_teacher_traces.py \
    --teacher_url http://localhost:8000 \
    --gsm8k_split train --mmlu_split train --cmmlu_split non_final_train_or_synthetic \
    --mvtec_train_only --neu_split train --cityflow_split train \
    --output_distill data/distill/distill_dataset.jsonl \
    --output_teacher_trace data/distill/teacher_decision_trace.jsonl

# 普通 KD 消融
python3 model_compression/train_sft_plain_kd.py \
    --config configs/models.yaml \
    --student_init Qwen/Qwen2.5-1.5B-Instruct \
    --distill_data data/distill/distill_dataset.jsonl \
    --output_dir models/adapters/plain_kd

# CEDD 第一阶段: 结构化决策蒸馏
CUDA_VISIBLE_DEVICES=2 python3 model_compression/train_cedd_structured.py \
    --config configs/models.yaml \
    --student_init Qwen/Qwen2.5-1.5B-Instruct \
    --teacher_trace data/distill/teacher_decision_trace.jsonl \
    --output_dir models/adapters/cedd_structured

# 学生在环探测与反事实边界样本挖掘
python3 model_compression/run_student_probe.py \
    --adapter_path models/adapters/cedd_structured \
    --dataset_manifest dataset_manifest.json \
    --split train \
    --output_probe data/distill/student_probe_trace.jsonl
python3 model_compression/mine_counterfactual_repairs.py \
    --teacher_url http://localhost:8000 \
    --probe_trace data/distill/student_probe_trace.jsonl \
    --output_repair data/distill/counterfactual_repair_trace.jsonl

# CEDD 第二阶段: 修复蒸馏 + 校准蒸馏
CUDA_VISIBLE_DEVICES=2 python3 model_compression/train_cedd_repair.py \
    --config configs/models.yaml \
    --student_init Qwen/Qwen2.5-1.5B-Instruct \
    --teacher_trace data/distill/teacher_decision_trace.jsonl \
    --repair_trace data/distill/counterfactual_repair_trace.jsonl \
    --output_dir models/adapters/cedd_repair

# INT4 GGUF + 量化行为保持验证
python3 model_compression/merge_quantize_and_verify.py \
    --student_init Qwen/Qwen2.5-1.5B-Instruct \
    --adapter_path models/adapters/cedd_repair \
    --output_gguf models/db4ai-edge-1.5b-kd-int4-Q4_K_M.gguf \
    --output_quant_trace data/distill/quant_behavior_trace.jsonl
```

### 7.5 P0-A 通过条件

```
G-DATA ✅  G-DB ✅  G-CLOUD ✅  G-KD-TRACE ✅  G1-dev ✅  G3-dev ✅  →  P0-B
```

---

## 8. P0-B：最小系统与成果2/3/4实验

### 8.1 启动边侧服务 + 最小系统

```bash
bash scripts/start_edge_llm.sh
python3 experiments/run_minimal_system.py \
    --config final_config_dev.yaml \
    --dataset_manifest dataset_manifest.json \
    --split train \
    --network_profiles configs/network_profiles.yaml \
    --workload_profiles configs/workload_profiles.yaml \
    --output_dir results/dev_train
python3 experiments/run_minimal_system.py \
    --config final_config_dev.yaml \
    --dataset_manifest dataset_manifest.json \
    --split validation \
    --network_profiles configs/network_profiles.yaml \
    --workload_profiles configs/workload_profiles.yaml \
    --output_dir results/dev
# 输出: runtime_state_trace.csv, decision_tuple_trace.csv, ttft/e2e preliminary metrics
```

### 8.2 成果2: IBEP 语义证据规划与风险校准复核

```bash
python3 model_compression/build_planner_training_data.py \
    --teacher_trace data/distill/teacher_decision_trace.jsonl \
    --student_probe data/distill/student_probe_trace.jsonl \
    --dataset_manifest dataset_manifest.json \
    --output data/distill/planner_train.jsonl
python3 model_compression/train_evidence_planner.py \
    --train_data data/distill/planner_train.jsonl \
    --output models/ibep_evidence_planner.pt

python3 experiments/run_ablation_evidence.py \
    --config final_config_dev.yaml \
    --dataset_manifest dataset_manifest.json \
    --planner_model models/ibep_evidence_planner.pt \
    --split validation \
    --network_profile normal \
    --workload_profile replay \
    --variant fixed_summary \
    --output_dir results/ablation/evidence_fixed
python3 experiments/run_ablation_evidence.py \
    --config final_config_dev.yaml \
    --dataset_manifest dataset_manifest.json \
    --planner_model models/ibep_evidence_planner.pt \
    --split validation \
    --network_profile normal \
    --workload_profile replay \
    --variant ibep_evidence_planner \
    --output_dir results/ablation/evidence_ibep
# 附录: --variant fixed_token_budget, --variant manual_review_threshold, --variant no_risk_calibration
```

### 8.3 成果3: TARS-SSM 运行状态建模

```bash
python3 experiments/train_runtime_state_model.py \
    --config final_config_dev.yaml \
    --dataset_manifest dataset_manifest.json \
    --runtime_trace results/dev_train/runtime_state_trace.csv \
    --output models/tars_ssm_state_model.pt

python3 experiments/run_ablation_runtime_state.py \
    --config final_config_dev.yaml \
    --dataset_manifest dataset_manifest.json \
    --runtime_model models/tars_ssm_state_model.pt \
    --split validation \
    --network_profile normal \
    --workload_profile replay \
    --variant boolean_readiness \
    --output_dir results/ablation/state_boolean
python3 experiments/run_ablation_runtime_state.py \
    --config final_config_dev.yaml \
    --dataset_manifest dataset_manifest.json \
    --runtime_model models/tars_ssm_state_model.pt \
    --split validation \
    --network_profile normal \
    --workload_profile replay \
    --variant tars_ssm \
    --output_dir results/ablation/state_tars_ssm
# 附录: --variant only_network, --variant only_load, --variant no_feedback_history (high_delay 网络)
```

### 8.4 成果4: CPR 约束策略路由与弱网自治

```bash
python3 experiments/train_policy_router.py \
    --config final_config_dev.yaml \
    --dataset_manifest dataset_manifest.json \
    --runtime_model models/tars_ssm_state_model.pt \
    --runtime_trace results/dev_train/runtime_state_trace.csv \
    --output models/cpr_policy_router.pt

for net in high_delay low_bandwidth high_loss short_disconnect; do
  python3 experiments/run_ablation_policy_router.py \
    --config final_config_dev.yaml \
    --dataset_manifest dataset_manifest.json \
    --runtime_model models/tars_ssm_state_model.pt \
    --policy_model models/cpr_policy_router.pt \
    --split validation \
    --network_profile "$net" \
    --workload_profile replay \
    --variant load_based \
    --output_dir "results/ablation/path_load_${net}"
  python3 experiments/run_ablation_policy_router.py \
    --config final_config_dev.yaml \
    --dataset_manifest dataset_manifest.json \
    --runtime_model models/tars_ssm_state_model.pt \
    --policy_model models/cpr_policy_router.pt \
    --split validation \
    --network_profile "$net" \
    --workload_profile replay \
    --variant constrained_policy_router \
    --output_dir "results/ablation/path_cpr_${net}"
done
# 附录: --variant static_split, --variant network_only, --variant no_weak_autonomy
```

### 8.5 G-Support-Perception

```bash
python3 experiments/run_perception_support.py \
    --config final_config_dev.yaml \
    --dataset_manifest dataset_manifest.json \
    --feature_mode replay \
    --output_dir results/perception
python3 experiments/run_perception_support.py \
    --config final_config_dev.yaml \
    --dataset_manifest dataset_manifest.json \
    --feature_mode online_smoke \
    --output_dir results/perception_online_smoke
# 输出: perception_results.csv, decision_quality_results.csv,
#       perception_latency_breakdown.csv, online_smoke_report.json
```

### 8.6 P0-B 通过条件

```
成果2消融 ✅  成果3消融 ✅  成果4消融 ✅
G2-dev ✅  G4-dev ✅  G5-dev ✅  G-Support-Perception ✅  →  P1
```

---

## 9. P1：完整系统与成果5/6实验

### 9.1 P1 多节点决策重放

```bash
python3 experiments/run_multinode_decision_replay.py \
    --config integrated_dev_config.yaml \
    --dataset_manifest dataset_manifest.json \
    --split train \
    --scenario traffic,combined \
    --network_profile normal \
    --workload_profile replay \
    --output_dir results/p1_multinode_train
python3 experiments/run_multinode_decision_replay.py \
    --config integrated_dev_config.yaml \
    --dataset_manifest dataset_manifest.json \
    --split validation \
    --scenario traffic,combined \
    --network_profile normal \
    --workload_profile replay \
    --output_dir results/p1_multinode
# 输出: p1_multinode_train/decision_tuple_trace.csv 和 p1_multinode/decision_tuple_trace.csv
```

### 9.2 构造 ST-HGCI 时空异构图

```bash
python3 experiments/build_sthgci_graph.py \
    --decision_trace results/p1_multinode_train/decision_tuple_trace.csv \
    --dataset_manifest dataset_manifest.json \
    --conflict_gt_manifest conflict_gt_manifest.json \
    --output_dir results/relation_graph_train
python3 experiments/train_sthgci_graph_model.py \
    --graph_dir results/relation_graph_train \
    --dataset_manifest dataset_manifest.json \
    --conflict_gt_manifest conflict_gt_manifest.json \
    --output models/sthgci_graph_model.pt
python3 experiments/build_sthgci_graph.py \
    --decision_trace results/p1_multinode/decision_tuple_trace.csv \
    --dataset_manifest dataset_manifest.json \
    --conflict_gt_manifest conflict_gt_manifest.json \
    --output_dir results/relation_graph
```

### 9.3 成果5: ST-HGCI 多节点冲突语义推理

```bash
python3 experiments/run_ablation_conflict_semantics.py \
    --config integrated_dev_config.yaml \
    --dataset_manifest dataset_manifest.json \
    --conflict_gt_manifest conflict_gt_manifest.json \
    --split validation \
    --network_profile normal \
    --workload_profile replay \
    --graph_model models/sthgci_graph_model.pt \
    --variant simple_vote \
    --output_dir results/ablation/conflict_vote
python3 experiments/run_ablation_conflict_semantics.py \
    --config integrated_dev_config.yaml \
    --dataset_manifest dataset_manifest.json \
    --conflict_gt_manifest conflict_gt_manifest.json \
    --split validation \
    --network_profile normal \
    --workload_profile replay \
    --graph_model models/sthgci_graph_model.pt \
    --variant sthgci_graph_conflict \
    --output_dir results/ablation/conflict_sthgci
# 附录: --variant text_match, --variant manual_similarity, --variant no_relation_graph
```

### 9.4 构造校正历史

```bash
python3 experiments/build_correction_history.py \
    --decision_trace results/p1_multinode_train/decision_tuple_trace.csv \
    --conflict_gt_manifest conflict_gt_manifest.json \
    --split train \
    --output results/p1_multinode_train/correction_history.jsonl
python3 experiments/build_correction_history.py \
    --decision_trace results/p1_multinode/decision_tuple_trace.csv \
    --conflict_gt_manifest conflict_gt_manifest.json \
    --split validation \
    --output results/p1_multinode/correction_history.jsonl
```

### 9.5 初始化节点可信度后验

```bash
python3 experiments/init_trust_posterior.py \
    --dataset_manifest dataset_manifest.json \
    --correction_history results/p1_multinode_train/correction_history.jsonl \
    --output initial_trust_posterior_snapshot.json
```

若 correction_history 不足以估计节点级后验，使用 train split 的标注先验与运行日志初始化，并在 manifest.json 记录 `initial_trust_posterior_source = train_prior_or_default_init`。该 fallback 只用于初始化，不允许在 Final Gate 后人工调整。

### 9.6 成果6: BTCF 节点可信度校准与闭环反馈

```bash
python3 experiments/run_ablation_trust_calibration.py \
    --config integrated_dev_config.yaml \
    --dataset_manifest dataset_manifest.json \
    --conflict_gt_manifest conflict_gt_manifest.json \
    --graph_model models/sthgci_graph_model.pt \
    --trust_snapshot initial_trust_posterior_snapshot.json \
    --split validation \
    --network_profile normal \
    --workload_profile replay \
    --variant cloud_single_correction \
    --output_dir results/ablation/trust_single
python3 experiments/run_ablation_trust_calibration.py \
    --config integrated_dev_config.yaml \
    --dataset_manifest dataset_manifest.json \
    --conflict_gt_manifest conflict_gt_manifest.json \
    --graph_model models/sthgci_graph_model.pt \
    --trust_snapshot initial_trust_posterior_snapshot.json \
    --split validation \
    --network_profile normal \
    --workload_profile replay \
    --variant bayesian_trust_calibration_feedback \
    --output_dir results/ablation/trust_feedback
# 附录: --variant cloud_vote, --variant no_feedback, --variant fixed_node_trust
```

### 9.7 P1 集成运行

```bash
python3 experiments/run_integrated_dev.py \
    --config integrated_dev_config.yaml \
    --dataset_manifest dataset_manifest.json \
    --conflict_gt_manifest conflict_gt_manifest.json \
    --planner_model models/ibep_evidence_planner.pt \
    --runtime_model models/tars_ssm_state_model.pt \
    --policy_model models/cpr_policy_router.pt \
    --graph_model models/sthgci_graph_model.pt \
    --trust_snapshot initial_trust_posterior_snapshot.json \
    --network_profiles configs/network_profiles.yaml \
    --workload_profiles configs/workload_profiles.yaml \
    --output_dir results/integrated_dev
```

---

## 10. P1-Regression

```bash
python3 experiments/run_regression_dev.py \
    --config integrated_dev_config.yaml \
    --dataset_manifest dataset_manifest.json \
    --planner_model models/ibep_evidence_planner.pt \
    --runtime_model models/tars_ssm_state_model.pt \
    --policy_model models/cpr_policy_router.pt \
    --graph_model models/sthgci_graph_model.pt \
    --network_profiles configs/network_profiles.yaml \
    --workload_profiles configs/workload_profiles.yaml \
    --trust_snapshot initial_trust_posterior_snapshot.json \
    --manifest manifest.json \
    --output_dir results/regression_dev
# 输出: ttft_results_regression.csv, weak_network_results_regression.csv,
#        e2e_results_regression.csv, communication_results_regression.csv,
#        resource_results_regression.csv, regression_gate_summary.json
```

通过条件: G2-reg(high_delay≥78%, low_bw≥78%, high_loss≥78%, weak_avg≥78%) + G4-reg(4 profile+combined ≥92%) + G5-reg(≤0.18s)

---

## 11. P2：Final Integrated Run

```bash
python3 experiments/run_final_integrated.py \
    --config final_config.yaml \
    --dataset_manifest dataset_manifest.json \
    --conflict_gt_manifest conflict_gt_manifest.json \
    --planner_model models/ibep_evidence_planner.pt \
    --runtime_model models/tars_ssm_state_model.pt \
    --policy_model models/cpr_policy_router.pt \
    --graph_model models/sthgci_graph_model.pt \
    --trust_snapshot initial_trust_posterior_snapshot.json \
    --network_profiles configs/network_profiles.yaml \
    --workload_profiles configs/workload_profiles.yaml \
    --manifest manifest.json \
    --schemes cloud_only_summary,edge_only,static_split,load_based,db4ai_edgeserve \
    --output_dir results/final
```

| 失败类型 | 处理方式 |
|---------|---------|
| **algorithm_failure** (指标未达标/结果错误/冲突未解决) | 不允许改参数重跑 |
| **infrastructure_failure** (断电/磁盘满/容器崩溃/进程被杀) | 记录 restart_reason, 从 frozen snapshot 重启 |
| **data_failure** (split泄漏/数据损坏/占位符未替换) | 废弃本次 run, 重新生成 dataset_manifest, 重启 P2 |

Final Gate: G1-G7 全部 Final 标准通过 → 五方案对比 → 报告生成

**final_gate_summary.json:**
```json
{
  "schemes": ["cloud_only_summary","edge_only","static_split","load_based","db4ai_edgeserve"],
  "gate_results": {
    "db4ai_edgeserve": {
      "G1":"pass|fail","G2":"pass|fail","G3":"pass|fail",
      "G4":"pass|fail","G5":"pass|fail","G6":"pass|fail","G7":"pass|fail"
    }
  },
  "comparison_tables": {
    "ttft":"ttft_results.csv","e2e":"e2e_results.csv",
    "weak_network":"weak_network_results.csv",
    "conflict":"conflict_results.csv","resolution":"resolution_results.csv",
    "perception":"perception_results.csv",
    "latency_breakdown":"latency_breakdown.csv",
    "decision_quality":"decision_quality_results.csv",
    "communication":"communication_results.csv",
    "resource":"resource_results.csv",
    "baseline_fairness":"baseline_fairness_report.json",
    "risk_audit":"risk_gate_audit.json",
    "conflict_gt_audit":"conflict_gt_audit.csv"
  }
}
```

### manifest.json 字段模板（扩展字段）

下列 `"..."` 仅表示模板占位，正式 Final Gate 产物必须替换为实际 hash 或实际取值，不得保留 `"..."`、空字符串、null hash 或其他占位符。

```json
{
  "git_commit": "...",
  "final_config_hash": "...",
  "risk_matrix_hash": "...",
  "fallback_policy_hash": "...",
  "preflight_report_hash": "...",
  "baseline_fairness_hash": "...",
  "teacher_model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
  "student_init_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
  "edge_model_name": "DB4AI-Edge-1.5B-KD-INT4",
  "edge_model_sha256": "...",
  "distill_dataset_hash": "...",
  "teacher_trace_hash": "...",
  "student_probe_trace_hash": "...",
  "counterfactual_repair_trace_hash": "...",
  "quant_behavior_trace_hash": "...",
  "sft_config_hash": "...",
  "lora_adapter_sha256": "...",
  "quant_config_hash": "...",
  "planner_model_hash": "...",
  "calibration_snapshot_hash": "...",
  "runtime_state_model_hash": "...",
  "policy_snapshot_hash": "...",
  "graph_model_hash": "...",
  "trust_posterior_snapshot_hash": "...",
  "conflict_gt_audit_hash": "...",
  "conflict_gt_sample_audit_hash": "...",
  "latency_breakdown_hash": "...",
  "online_perception_smoke_hash": "...",
  "data_split_hash": "...",
  "network_profile_hash": "...",
  "prompt_template_hash": "...",
  "policy_config_hash": "...",
  "decision_parser_hash": "...",
  "capability_metric_script_hash": "...",
  "ttft_metric_script_hash": "...",
  "e2e_metric_script_hash": "...",
  "conflict_metric_script_hash": "...",
  "resolution_metric_script_hash": "...",
  "perception_metric_script_hash": "...",
  "communication_metric_script_hash": "...",
  "resource_metric_script_hash": "...",
  "frozen_scu_hash": "...",
  "synthetic_chinese_nlp_hash": "...",
  "fallback_events": [],
  "timestamp": "..."
}
```

---

## 12. 实施阶段与交付

```
P0-A0 (第1周前置): Preflight + conflict_gt dry-run/audit + G1/G3/G4/G5 smoke
P0-A (第1-3周):  G-DATA/DB/CLOUD + G-KD-TRACE + 成果1 + G1/G3
P0-B (第4-6周):  成果2/3/4 + G2/G4/G5 + G-Support-Perception + 200ms action header smoke
P1 (第7-9周):    成果5/6 + G6/G7 + P1-Regression
P2 (第10-12周):  Final Gate + 五方案 + 报告 + 视频 + 打包
```

交付: 作品报告 PDF + 视频 MP4 + 源码 + Docker + 脚本 + 日志 + perception/decision/communication/resource/latency CSV + preflight/risk/baseline/conflict_gt audit + manifest.json + dataset_manifest.json + conflict_gt_manifest.json + README + 报名表

---

*DB4AI-EdgeServe v4.1.0-research-final — 研究机制升级版 — 2026-06-24*
*所有压缩引用已展开。全文档自包含。修改历史见 docs/REVISION_LOG.md*
