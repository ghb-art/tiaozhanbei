# DB4AI-EdgeServe v5.1

## 面向云边协同场景的分布式人工智能感知与决策系统——指标驱动实施计划

**题目编号：** XH-202606
**发榜单位：** 山东浪潮数据库技术有限公司
**文档状态：** 当前唯一有效的实施总纲
**重构日期：** 2026-07-19
**历史路线：** v1-v31 的实验过程、失败原因和产物索引保留在 `docs/REVISION_LOG.md`，不再作为当前实施约束。

---

## 0. 文档原则

### 0.1 最高约束

本项目以赛题原文和最终提交要求为最高约束。本计划、模型规模、算法名称、数据集、训练路线和软件结构均可在 Final Freeze 前调整，但调整必须服务于以下目标：

1. 达到赛题七项核心指标；
2. 形成可运行、可复现、可演示的完整云边协同系统；
3. 至少覆盖两个差异明显的应用场景；
4. 形成作品报告、运行视频、代码、数据和量化实验结果；
5. 不使用测试集标签训练、逐题修复或选择候选，不以不完整内存口径、缩小统计分母等方式制造达标结果。

实施优先级固定为：

```text
指标可达性 > 完整闭环 > 实验可信度 > 演示效果 > 算法复杂度
```

### 0.2 当前真实状态

| 项目 | 当前状态 | 结论 |
|---|---|---|
| G-DATA | 已完成 | 八个数据集、split 和 manifest 已建立，重构后需复核场景最终子集 |
| G-DB | 已完成 | KWDB 建表、写入、查询和导出已有 live gate |
| G-CLOUD | 已完成 | Qwen2.5-14B-AWQ 云端服务已有 live gate |
| G1 能力 | Student v1未通过，v2代码与协议已冻结 | v1 Q4的170题Code保持率75%而失败；提前执行的一次完整测试也失败且已封存。v2只使用170题任务级汇总调整，不使用正式全测反馈 |
| G0 联合门禁 | 历史候选和Student v1均未联合通过 | 当前执行最后一个Student候选v2的共享蒸馏、任务Adapter、96题安全门、170题和内存联合路线 |
| G3 内存 | 已形成 Dev 证据 | Qwen3-1.7B Q3 历史峰值 1502.67MB、IQ2 峰值 1306.14MB、DeepSeek Q2 峰值 920.23MB；下一步只重测 Q3 的 32/16 runtime 配置 |
| G2/G4/G5 | 未形成正式系统结果 | 云边调度、弱网和时延实验待实现 |
| G6/G7 | 未形成正式系统结果 | 多节点冲突仲裁与一致性实验待实现 |
| 两场景闭环 | 未完成 | 工业检测与交通监控需先完成最小闭环再扩展 |

任何报告、README 或答辩材料不得把开发结果写成 Final Gate 已通过。

### 0.3 本次重构的关键决策

- 不再把 3B、INT4、v31 或六个独立学习模型视为必须保留的路线。
- 边缘模型必须同时通过 G1 能力和 G3 内存，单项通过不能进入系统主线。
- DeepSeek 1.5B 的完整 BF16 Dev 为 Math 84.38%、Code 23.81%、NLP 17.19%，P0-A2 已关闭；不得继续训练、合并或量化该路线。
- P0-A4 将Qwen2.5-14B-AWQ固定为不可调优的正式分母；BF16+LoRA 14B仅作为蒸馏Teacher，不替换分母。
- Qwen3-1.7B先做共享多任务蒸馏，再以训练集imatrix量化为Q4_K_M，运行时固定Q8 KV；Q3_K_M因96题代码保持率仅50%作为失败对照保留。Top-1任务Adapter是必须独立通过能力、内存和时延门的可选消融。
- 96题内部烟测要求三任务及capped macro均≥75%；170题选择门均≥80%，只允许任务级汇总反馈且最多两个Student版本。
- v31 已在内部执行验证中失败并终止，不继续编号式 Code 微调。
- 原六项技术成果收敛为三个核心模块：轻量边缘智能、状态感知协同调度、图可信一致性决策。
- 业务决策与自然语言解释解耦：先输出可执行短决策，解释和云端校正异步完成。
- 两场景共用统一事件协议、路由器、数据库和一致性组件，避免建设两套系统。
- KWDB 不仅保存日志，还承担事件状态、断网 outbox、模型版本、策略快照和可信度状态管理。

---

## 1. 赛题指标与验收 Gate

### 1.1 核心 Gate 总表

| Gate | 赛题目标 | 本项目正式通过条件 | 主要证据 |
|---|---|---|---|
| G1 边缘能力 | 满血模型能力的 80%–90% | Math、Code、NLP 三项保持率均 ≥80%，宏平均 ≥80% | `capability_results.csv`、逐样本 trace |
| G2 TTFT | TTFT 减少 75% | 相对公平 Cloud-only 基线，综合及三个弱网 profile 的 TTFT 降低率均 ≥75% | `ttft_results.csv` |
| G3 内存 | 单次推理内存 ≤1.5GB | 边缘完整推理进程在正式推理窗口内的总内存峰值 ≤1500 MB；同时报告 P95、PSS/RSS 和设备内存 | `memory_results.csv`、采样日志 |
| G4 弱网保持 | 基本业务功能保持率 ≥90% | 四个弱网 profile 及综合保持率均 ≥90% | `weak_network_results.csv` |
| G5 端到端时延 | 两类场景平均时延 ≤0.2s | 工业、交通、综合三项 mean E2E 均 ≤200ms；额外报告 P50/P95 | `e2e_results.csv` |
| G6 决策冲突 | 冲突比例 ≤5% | 校正与同步截止后，关联任务组冲突比例 ≤5% | `conflict_results.csv` |
| G7 冲突解决 | 解决成功率 ≥90% | 真实冲突组中形成正确唯一全局决策的比例 ≥90% | `resolution_results.csv` |

Dev Gate 应留出余量：P0-A4先以96题75%作早停，再以170题各单项及capped macro均≥80%晋级；G2 ≥78%、G3峰值≤1400MB、G4≥92%、G5 mean≤180ms、G6≤4%、G7≥92%。这些Dev Gate不改变赛题正式阈值。

### 1.2 G1：能力保持率

固定使用同一 prompt、输出约束、采样参数和评分器比较边缘模型与云端 14B：

```text
Math_ratio = Edge_GSM8K_accuracy / Cloud_GSM8K_accuracy
Code_ratio = Edge_HumanEval_pass@1 / Cloud_HumanEval_pass@1
NLP_ratio  = Edge_CMMLU_accuracy / Cloud_CMMLU_accuracy
Overall    = mean(min(Math_ratio, 1), min(Code_ratio, 1), min(NLP_ratio, 1))
```

正式主结果使用官方完整测试集：GSM8K 1319题、HumanEval 164题和CMMLU 11582题，共13065题。历史500/164/1000结果保留为历史证据但不再作为P0-A4正式主结果。HumanEval使用隔离Python sandbox，单候选pass@1，禁止网络，逐测试超时10秒；无法解析均记为错误。

边缘侧允许使用量化模型及其固定 tokenizer/parser，但不得通过云端调用完成 G1。若使用本地工具增强，必须额外分别报告“纯模型结果”和“工具增强结果”；G1 主结果采用赛前与发榜单位确认后更保守的口径。

### 1.3 G2：TTFT

```text
TTFT = task_received_ts 到首个可解析、可执行 action 字段输出的时间
TTFT_reduction = 1 - TTFT_DB4AI / TTFT_CloudOnly
```

公平性要求：

- 两方案接收完全相同的结构化感知摘要；
- 使用相同 action schema、解析器、任务顺序和网络 profile；
- Cloud-only 也允许只输出短 action，不强制生成长解释；
- DB4AI 的 provisional action 只有满足业务有效性条件才能计入 TTFT；
- 同时报告 mean、P50、P95 和置信区间，主 Gate 使用预先冻结的聚合口径。

### 1.4 G3：内存

G3 是模型选择的前置 Gate，不得留到项目末期。

测量对象包括边缘业务完成一次推理所需的完整常驻组件：模型权重映射、KV cache、tokenizer、运行时 buffer 和决策头。若未来增加额外参数模块，也必须计入；若使用 GPU/NPU，还必须计入设备显存，不得只报告 Python 或 llama.cpp 的 CPU RSS。

固定条件：

```text
batch_size = 1
context_length = 正式业务配置，默认不低于 512
warmup_requests >= 20
measure_requests >= 100
sample_interval_ms <= 100
primary_metric = steady_state_total_memory_peak_mb_decimal
host_prompt_cache_mib = 0
cache_idle_slots = false
```

同时报告：模型文件大小、RSS、PSS、cgroup memory.current、设备内存、P95 和观测最大值。正式 Gate 以完整推理窗口内的总内存峰值为准。llama.cpp主机提示缓存属于可选的重复提示加速状态，边侧低内存方案必须通过`--cache-ram 0 --no-cache-idle-slots`显式关闭并写入审计；请求级`cache_prompt=false`不足以关闭该全局缓存。若采用 mmap，应明确文件页统计方式。任何模型只有 G1 与 G3 同时通过，才可命名为正式边缘模型。

### 1.5 G4：弱网业务保持率

弱网业务成功不能只按“JSON 可解析”判断：

```text
valid_business_task =
  在 200ms 内产生非空、可执行 action
  AND action 满足任务安全约束
  AND action 与 ground truth 一致或落在允许动作集合

business_retention = valid_business_tasks / all_tasks
```

四类弱网环境为 `high_delay`、`low_bandwidth`、`high_loss`、`short_disconnect`，另报告 combined。断网期间系统必须走本地自治路径，网络恢复后通过 outbox 补传，不得阻塞当前业务等待云端。

### 1.6 G5：端到端时延

```text
E2E = business_action_output_ts - raw_event_received_ts
```

包含预处理、感知推理、事件编码、状态读取、路由和业务 action 生成。异步解释与 1 秒内的云端校正单独统计，不并入首个业务动作时延。未在 1 秒观测窗内产生有效动作的请求按 1000ms 计入，避免只统计成功样本。

赛题 Gate 使用各场景 mean ≤200ms；同时以 P95 ≤250ms 作为项目 stretch target，以展示尾延迟稳定性。

### 1.7 G6/G7：一致性

```text
post_conflict_ratio =
  截止时间后仍未形成唯一一致决策的关联任务组数
  / 全部关联任务组数

resolution_success_rate =
  被正确识别、仲裁并同步到相关节点的真实冲突组数
  / conflict_gt = true 的任务组数
```

冲突至少覆盖：同一区域重复感知、跨摄像头关联目标、事件类型冲突、风险等级冲突、动作冲突和过期决策冲突。Ground truth 的构造规则、人工抽样审计和 hash 必须在 Final Freeze 前完成。

---

## 2. 总体技术方案

### 2.1 三个核心模块

#### 模块 A：EdgeLite——资源约束下的边缘感知与轻量决策

职责：

- 在边缘节点完成视频/图像预处理和轻量视觉感知；
- 将视觉、传感器和数据库状态统一编码为 `EdgeEvent`；
- 使用轻量决策头输出毫秒级 `action header`；
- 使用量化轻量大模型完成本地语义理解、异常归因和离线自治；
- 通过冻结 Teacher 对照、结构化 fast path 和 importance-aware Q3 量化满足 G1/G3。

模块 A 不再维护已关闭的 CEDD 专项训练路线；当前只允许冻结基座选型和独立 GGUF 量化。

#### 模块 B：SafeRoute——状态感知的云边协同调度

职责：

- 读取网络 RTT/抖动/丢包/带宽、队列、CPU/内存、模型置信度和任务风险；
- 在 `edge_fast`、`edge_llm`、`cloud`、`edge_then_cloud` 四条路径中选择；
- 高风险或低置信任务异步提交云端，简单任务留在边缘；
- 弱网时强制本地自治，恢复后补传和校正；
- 以时延、准确率、带宽和 SLA 违约组成约束优化目标。

首版采用小型 GBDT/MLP 或上下文 bandit，只有在数据确实充足时才升级复杂状态空间模型。规则路由仅作为 baseline 和安全兜底。

#### 模块 C：GraphTrust——时空图冲突仲裁与可信反馈

职责：

- 将节点、事件、区域、目标和决策构造成时空关系图；
- 识别多个边缘节点的关联事件与语义冲突；
- 综合时间新鲜度、模型置信度、节点历史可靠度和云端复核结果形成唯一决策；
- 将校正结果与可信度后验写回 KWDB，并同步到相关边缘节点；
- 通过 outbox/ack 保障弱网恢复后的最终一致性。

首版使用轻量图特征模型加贝叶斯可信度更新；GNN 只在 validation 明确优于轻量模型时进入完整方案。

### 2.2 分层实时路径

```text
原始视频/传感器
      │
      ▼
边缘感知器 ──> EdgeEvent ──> 毫秒级决策头 ──> provisional action
                             │
                             ▼
                        SafeRoute
                      ┌──────┼────────┐
                      ▼      ▼        ▼
                  edge LLM  cloud   edge→cloud
                      └──────┼────────┘
                             ▼
                    GraphTrust 全局仲裁
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
          最终动作        KWDB 状态       模型/策略更新
```

`provisional action` 必须安全、可执行且可独立完成基本业务；云端响应只允许提升结果或在截止时间内校正，不能成为弱网环境下业务可用的必要条件。

### 2.3 两个应用场景

| 项目 | 场景一：工业质量检测 | 场景二：城市交通监控 |
|---|---|---|
| 数据 | NEU-DET 为主，MVTec AD 支撑 | CityFlow 为主，UA-DETRAC 支撑 |
| 边缘节点 | 产线相机/工位 | 路口摄像头/区域节点 |
| 实时动作 | pass、hold、recheck、stop_line | allow、warn、track、dispatch |
| 云端作用 | 跨批次缺陷分析、复杂归因、模型更新 | 跨摄像头关联、拥堵与事件全局判断 |
| 关联冲突 | 相邻工位对同一工件判断不一致 | 重叠视野/跨摄像头对目标和事件判断不一致 |
| 业务价值 | 降低漏检、停线和上传带宽 | 提升联动效率、弱网连续性和全局一致性 |

两个场景使用同一 `EdgeEvent`、`DecisionTuple`、调度器、冲突协议和指标脚本，只替换感知插件及场景动作表。

### 2.4 KWDB 的不可替代作用

KWDB/KaiwuDB 统一承载：

- 边缘感知事件与设备时序状态；
- 网络、负载和推理性能 trace；
- 未上传事件 outbox、重试和 ack；
- 全局决策、冲突关系和校正历史；
- 节点可信度后验；
- 模型、量化产物、策略和 schema 版本；
- 审计日志和实验结果导出。

必须增加 `file/cache-only` 消融，量化 KWDB 在状态查询时延、断网补传成功率、跨节点关联查询和可追溯性方面的作用，避免数据库仅作为项目名称或普通日志容器。

---

## 3. 边缘模型联合可行性路线

### 3.1 先做 G0，不再串行等待

正式训练前先执行 `G0-CAPMEM`：在相同硬件、context 和评分器下并行比较候选的能力、内存和 action header 时延。

| 候选 | 目的 | 进入主线条件 |
|---|---|---|
| A：现有 v25 3B + 低于 4-bit 的 GGUF | raw IQ2 峰值 1737.83MB，v25 Code 全精度也只有 75% | G0 failed/pruned |
| B：Qwen2.5-1.5B Q4_K_M | 历史保持率 Math 73.09%、Code 65.15%、NLP 81.49%，峰值 1688.24MB | G0 failed；仅保留 Q3 备用可能性 |
| C：Qwen3-1.7B IQ2_XS | G3 为 1306.14MB，但 CMMLU matched smoke 保持率仅 4.76% | G0 failed；禁止恢复 IQ2 |
| D：DeepSeek-R1-Distill-Qwen-1.5B Q2_K_S | G3 为 920.23MB；完整 BF16 Dev Math 84.38%、Code 23.81%、NLP 17.19% | P0-A2 closed/pruned |
| E：Qwen3-1.7B Q3_K_M + Q8 KV | Q4 KV使F16/Q3 logits显著偏移；Q8 KV 24题为3/8、7/8、3/8，接近HF的3/8、7/8、4/8 | P0-A3 primary reopened；完整170条待复验 |
| F：Qwen2.5-1.5B Q3_K_M | 预计比 Q4 获得更大内存余量，但必须独立复验三任务能力 | P0-A3 conditional fallback |

不得先决定模型名称再解释指标。联合得分只用于 Dev 排序：

```text
feasible = G1_dev_pass AND G3_dev_pass
rank = capability_margin + latency_margin + memory_margin
```

### 3.2 止损规则

- v31 内部验证为 `30/64→30/64`，未超过历史门槛 `32/64`，已终止且不得重跑、合成或进入 HumanEval。
- 不再以 HumanEval 正式错题创建 v32、v33 等修复数据。
- 3B 的任意量化格式若模型文件、最小运行 RSS 或设备内存已经超过 1500MB，不再投入正式 G1 评测。
- DeepSeek 的 170 条完整 Dev 已触发止损；不允许用 LoRA、增加 token 上限或正式错题重新开启该路线。
- Qwen3 HF 在冻结 Dev 的任一 Teacher-relative 单项低于 80% 时立即拒绝，不生成 Q3；Q3 Dev 任一单项低于 80% 时也立即拒绝，不运行正式 G0。
- 若Q3在完整、无生成错误且提示模板一致的前提下相对HF异常大幅下降，允许使用F16 GGUF、token/logit对照定位运行参数。已确认Q4 KV会破坏Qwen3 logits，活动配置固定为Q8 KV；F16诊断不参与部署门禁，也不接触正式测试集。
- Qwen2.5-1.5B Q3 只能在 Qwen3 拒绝审计存在时启动，不并行调参两个候选。
- G0 必须在完整系统开发前 3 个工作日内给出主候选和备用候选。
- 2026-07-18 的 G0 结果为 `failed, feasible=0/9`；在能力恢复候选重新通过 G0 前，不得宣称主边缘模型已冻结。
- 当前 CPU llama.cpp 短输出均值仍为 Qwen3 IQ2 `4.08s`、DeepSeek Q2 `2.58s`；这不是两场景正式 G5，但已排除“CPU 大模型同步路径直接达到 0.2s”的假设，必须使用加速设备或结构化 fast path。

### 3.3 量化与能力保护

执行顺序：

1. Qwen2.5-14B Teacher 以单 vLLM 端点在 GPU 0–3 上执行 tensor parallel 4，并在冻结 170 条 Dev 上生成逐样本分母；
2. 未量化候选使用完全相同的 prompt、sample ID、token 上限和 scorer；
3. Math、Code、NLP 和 capped macro 保持率均 ≥80%，生成错误为 0；
4. 只为通过 HF Dev 的候选生成 train-only importance matrix 和Q4_K_M；
5. Q4在相同170条Dev上再次运行完整能力对比；
6. 完成 20+100 请求真实 G3，Dev 目标 ≤1400MB、正式门槛 ≤1500MB；
7. 冻结唯一候选，正式 G1/G0 只运行一次；正式失败后不得返回调参。

量化前后必须记录逐样本行为差异、模型 hash、量化参数、runtime 版本和 context 配置。

### 3.4 P0-A3 冻结 Dev 与判定口径

P0-A3 复用已审计的 170 条 selection-only 数据：CMMLU official dev 64、GSM8K train 内部留出 64、MBPP dev_select 42。训练/验证 group overlap、正式集引用和 HumanEval prompt overlap 均为 0。

Teacher、HF 候选、Q3 候选必须逐样本完全匹配。保持率沿用 G1 口径：

```text
task_ratio = candidate_accuracy_on_170 / teacher_accuracy_on_same_170
pass = all(task_ratio >= 0.80) AND mean(min(task_ratio, 1)) >= 0.80
       AND generation_error_count = 0
```

评测程序执行完成产生的 `status=passed` 不能替代该保持率审计。不得将 170 条错题、答案或响应回灌训练数据。

---

## 4. 数据、协议与版本管理

### 4.1 数据隔离

每个数据源按任务组而不是按单帧划分：

- `train`：模型、路由器、图模型和可信度先验训练；
- `validation`：阈值校准、候选选择、消融与 Dev Gate；
- `test`：Final Freeze 后一次正式评测，不参与训练和选择。

跨摄像头同一目标、同一工件序列、增强样本及近重复文本必须落入同一个 split。所有最终 sample id、来源、license、hash、划分规则和统计写入 `dataset_manifest.json`。

### 4.2 核心协议

`EdgeEvent` 最少包含：

```json
{
  "event_id": "...",
  "scene": "industrial|traffic",
  "node_id": "...",
  "region_id": "...",
  "object_id": "...",
  "event_ts": 0,
  "event_type": "...",
  "risk": 0.0,
  "perception_confidence": 0.0,
  "features_ref": "...",
  "model_version": "..."
}
```

`DecisionTuple` 最少包含：

```json
{
  "decision_id": "...",
  "event_id": "...",
  "action": "...",
  "risk": 0.0,
  "confidence": 0.0,
  "path": "edge_fast|edge_llm|cloud|edge_then_cloud",
  "provisional": true,
  "reason_code": "...",
  "deadline_ms": 200,
  "policy_version": "..."
}
```

### 4.3 模型和策略分发

云端模型更新闭环必须进入原型，而不是只写在报告中：

1. 云端根据 train/运行反馈生成候选模型参数包、量化产物或策略；
2. validation 通过后写入 model registry；
3. 产物分块、SHA256 校验并使用签名元数据；
4. 边缘节点断点续传到 staging；
5. 完整性、兼容性和 smoke 验证通过后原子激活；
6. 失败自动回滚上一版本；
7. 节点 ack 和当前版本写回 KWDB；
8. 灰度发布期间不同版本的决策进入 GraphTrust 版本一致性检查。

正式演示至少包含一次弱网中断、恢复续传、成功激活或失败回滚。

### 4.4 冻结与审计

Final Freeze 前必须冻结：

- git commit；
- train/validation/test split hash；
- 模型、量化产物和 runtime；
- 感知器、路由器、图模型和可信度快照；
- 网络与负载 profile；
- prompt、parser、schema 和动作安全约束；
- 所有指标脚本；
- baseline 配置；
- 随机种子和重复次数。

工程路径修复可以重新运行，但必须记录原因和 hash。正式 test 结果不得用于修改模型、阈值或数据。

---

## 5. 实验设计

### 5.1 公平基线

| 方案 | 描述 |
|---|---|
| Cloud-only | 感知摘要全部上传云端 14B，云端直接给出短决策 |
| Edge-only | 所有任务只在边缘运行，不使用云端校正和全局仲裁 |
| Static-split | 按固定任务类型决定云或边 |
| Load-only | 只根据当前网络/负载阈值动态路由 |
| DB4AI-EdgeServe | EdgeLite + SafeRoute + GraphTrust + KWDB 闭环 |

所有方案共享原始数据顺序、感知结果、动作 schema、硬件资源上限、并发负载、网络 profile 和观测窗口。对不适用组件不得偷偷提供额外缓存或 ground truth。

### 5.2 网络与负载

开发阶段沿用 `configs/network_profiles.yaml` 和 `configs/workload_profiles.yaml`，在 Final Freeze 前根据实测确认并冻结。至少包含：

- 正常网络；
- 100ms 延迟 + 20ms 抖动；
- 1Mbps 低带宽；
- 5% 丢包；
- 每 60 秒断网 5 秒；
- stable、burst、dataset replay；
- 2、4、8 个逻辑边缘节点扩展实验。

网络使用 `tc/netem` 或等价的可审计注入方式，保存实际采样而不是只保存配置值。

### 5.3 重复与统计

- 能力评测：确定性解码，冻结测试集运行一次，保留逐样本结果；
- 系统性能：每个方案×场景×网络 profile 至少 3 次独立重复；
- 每次预热后正式观测不少于 300 秒或不少于 500 个任务；
- 报告 mean、P50、P95、标准差及 95% bootstrap 置信区间；
- 正式结果聚合全部预定重复，不选择最佳轮次；
- 机器时间同步，所有 trace 使用单调时钟计算持续时间。

### 5.4 感知和决策支撑指标

除 G1-G7 外，必须报告：

- 工业：precision、recall、F1、漏检率；
- 交通：事件准确率、目标关联准确率/IDF1（按最终任务定义选取）；
- 云端请求率、上传字节数、带宽峰值；
- 边缘 CPU/内存、云端 GPU 利用率；
- 模型更新成功率、断点续传和回滚时间；
- KWDB 查询延迟、outbox 补传率和冲突追溯耗时。

### 5.5 核心消融

只保留能够回答明确问题的消融：

1. 无蒸馏/量化能力保护：证明 EdgeLite；
2. 固定路由与仅负载路由：证明 SafeRoute；
3. 无弱网自治：证明 G4 的来源；
4. 无关系图与固定多数投票：证明 GraphTrust；
5. 固定节点可信度：证明反馈校准；
6. 无 KWDB outbox/model registry：证明数据底座价值。

不再为每个模块设计大量名称不同但结论重复的变体。

---

## 6. 分阶段实施

### P0：联合可行性与路线冻结（当前 P0-A4）

任务：

- 冻结Qwen2.5-14B-AWQ的96/170及官方完整分母，逐题结果只进入密封区；
- 使用4卡纯GPU ZeRO-3训练最多三个BF16 14B LoRA候选，模型参数和LoRA AdamW优化器
  均在GPU间分片且禁用CPU offload，以独立Teacher验证集选优；
- 只用非正式训练数据生成数学答案校验、代码执行校验和NLP标签校验后的蒸馏数据；
- 训练共享Qwen3-1.7B Student，量化为训练集imatrix Q4_K_M并固定Q8 KV；
- 依次执行96题75%、170题80%和20+100请求≤1400MB门禁，内存门与正式边侧服务均关闭主机提示缓存；
- 最多两个Student版本；正式完整Student评测只运行一次且之后禁止调优；
- 可选Top-1 Adapter只有同时通过相同能力门和完整内存/加载时延测量才可晋级。
- v2共享与首轮Adapter均在96题Code保持率`70.833%`失败后，不再重复233条Teacher通过样本。P0-A4R的NLP短理由修复在量化96题上达到90%保持率，已按哈希冻结；Code修复仅66.67%，因此v2任务Adapter路线终止。冻结的NLP LoRA以v2为基座，不得跨基座挂到v1。
- P0-A4R2回到表现更好的v1 Q4_K_M：Math和NLP使用共享v1，仅Code允许训练新Adapter。Code固定Rank 4、alpha 4、学习率`1e-5`、1轮、四卡DDP；1792个独立APPS/MBPP任务不做行复制，通过逐来源损失权重使两类总训练质量各占50%；checkpoint只使用42题train-only可执行MBPP内部集选择。
- P0-A4R2首次训练已完成：v1内部基线与checkpoint-56均为`24/42`，新增0、回退0，未满足“净增至少1题且回退不超过2题”，因此不得发布或进入量化门禁。当前回退部署仍为不带任务Adapter的v1 Q4_K_M。

退出条件：量化Student在170题上相对AWQ分母的三任务保持率及capped macro均≥80%，生成错误为0，20+100请求峰值≤1400MB；随后一次官方完整测试的三任务及宏保持率均≥80%。Student v1在170题的Math/Code/NLP/Macro保持率为`90.625/75.000/80.435/82.020%`，因Code失败；按显式授权提前运行并封存的官方全测保持率为`82.149/68.571/70.650/73.790%`，同样失败且不得反馈训练。第二次170题候选机会仍未使用，但v2修复路线和P0-A4R2首次v1 Code Adapter均未取得进入该门禁的资格。后续若继续，必须先注册新的train-only Code候选与独立验证口径；不能覆盖既有内部选择、96题或官方完整证据，因此P0仍不退出。


### P1：两场景最小完整闭环（第 1–2 周）

任务：

- 建立工业、交通两个感知插件；
- 落地 `EdgeEvent`、`DecisionTuple` 和动作安全约束；
- 跑通边缘决策、云端复核、KWDB 写入和最终动作返回；
- 建立 2 个边缘节点 + 1 个云节点的可演示部署；
- 先用规则路由/多数仲裁跑通全链路，形成 baseline。

退出条件：两个场景均能离线运行，断云后仍产生有效动作；端到端 trace 字段完整。

### P2：EdgeLite 正式冻结（第 2–3 周）

任务：

- 完成边缘模型蒸馏/量化候选选择；
- 运行 validation G1/G3；
- 冻结模型、parser、context 和运行时；
- 完成正式 G1/G3 与量化前后对比。

退出条件：G1、G3 达标。若只能通过其中一项，模型不得进入最终方案。

### P3：SafeRoute 与弱网实验（第 3–4 周）

任务：

- 采集网络、资源、置信度、风险和路径收益训练 trace；
- 训练小型约束路由器并加入规则安全兜底；
- 实现 provisional action、异步 cloud correction 和 outbox；
- 完成 G2/G4/G5 Dev Gate 与基线对比。

退出条件：两个场景 G2/G4/G5 全部达到 Dev Gate，弱网失败样本可以从 trace 复盘。

### P4：GraphTrust 与模型分发（第 4–5 周）

任务：

- 冻结 relation group 和 conflict ground truth；
- 实现关系图、冲突检测、唯一仲裁、可信度更新和 ack；
- 实现模型 registry、断点续传、原子切换及回滚；
- 完成 G6/G7 Dev Gate 与核心消融。

退出条件：G6/G7 达到 Dev Gate；弱网恢复后决策和模型版本最终一致。

### P5：集成回归与扩展实验（第 5–6 周）

任务：

- 运行五方案公平对比；
- 运行两个场景、五种网络、三种负载和 2/4/8 节点实验；
- 完成通信、资源、稳定性、扩展性与 KWDB 消融；
- 修复工程问题并冻结最终配置。

退出条件：G1-G7 均达到 Dev Gate；没有 placeholder、缺失 trace 或无法复现的指标。

### P6：Final Gate 与交付（第 7 周）

任务：

- 执行 Final Freeze 和 hash 审计；
- 按预定重复次数运行正式集；
- 自动生成 G1-G7 summary 和评分项对照表；
- 形成报告、视频、部署包和答辩材料。

退出条件：交付包可在干净环境按 README 复现核心演示和主要表格。

---

## 7. 工程交付结构

下列为目标结构；当前不存在的文件是待实现交付物，不视为已经完成：

```text
edge/
  perception/                 两场景感知插件
  decision/                   action header、边缘 LLM 和安全约束
cloud/
  inference/                  云端全量模型复核
  model_registry/             模型/策略分发与回滚
scheduler/
  saferoute.py                状态感知路径选择
consistency/
  graphtrust.py               关系图、仲裁、可信反馈
storage/
  kwdb_repository.py          事件、状态、outbox、版本和审计访问层
experiments/
  run_capability_gate.py
  run_memory_gate.py
  run_system_benchmark.py
  run_conflict_benchmark.py
  run_final_gate.py
configs/
  network_profiles.yaml
  workload_profiles.yaml
  actions_industrial.yaml
  actions_traffic.yaml
results/final/
reports/final/
```

所有正式实验入口支持：固定 seed、配置文件、manifest、输出目录、resume 仅限基础设施失败，以及 `--dry-run`。

---

## 8. 风险、触发条件与替代方案

| 风险 | 早期信号 | 立即动作 | 禁止动作 |
|---|---|---|---|
| 3B 无法满足 1.5GB | 文件或最小 RSS 已超限 | 切换 sub-3bit、1.5B 或 2B 候选 | 等 G1 完成后才测内存 |
| 小模型能力不足 | 冻结 Dev 任一 Teacher-relative 单项 <80% | 按预注册顺序从 Qwen3 Q3 转向 Qwen2.5-1.5B Q3；两者均失败则改结构化工具头/硬件方案 | 用正式错题持续修复或用宏平均掩盖单项失败 |
| 200ms 无法生成 LLM 文本 | action header P95 超限 | 决策头先输出，解释异步 | 缩短统计区间或排除失败请求 |
| 弱网保持率虚高 | action 可解析但质量低 | 把正确性/安全性纳入有效任务 | 只统计 JSON 成功率 |
| 冲突数据不足 | validation conflict group <50 | Final Freeze 前扩展真实/可审计关联组 | 根据模型结果改 ground truth |
| 三个学习模块集成过慢 | 两周后仍无双场景闭环 | 先用轻量模型/规则 baseline 完成系统 | 同时新增更多研究模型 |
| KWDB 价值不明显 | 仅作为日志表 | 增加 outbox、registry、关系查询和消融 | 只在标题中强调数据库 |
| 正式实验波动大 | Dev margin 小于 2% | 增加 Dev 余量和预定重复 | 只挑最好一次 |

---

## 9. 评分项—证据映射

| 评分项 | 分值 | 项目证据 |
|---|---:|---|
| 实时性改进 | 15 | 五方案 TTFT/E2E 表、分布图、弱网 trace |
| 感知与决策效果 | 15 | 两场景 F1/任务准确率、G1、G6/G7 |
| 资源与通信效率 | 10 | G3、上传量、带宽、云端负载、请求率 |
| 方案完整性 | 15 | 两场景可运行原型、云边路径、KWDB、模型分发闭环 |
| 可扩展性与适应性 | 10 | 2/4/8 节点和三种负载结果、插件化场景协议 |
| 稳定性 | 10 | 四类弱网 G4、恢复同步、回滚、重复实验波动 |
| 决策一致性 | 10 | G6/G7、关系图消融、冲突案例回放 |
| 创新性 | 10 | EdgeLite、SafeRoute、GraphTrust 三模块及消融 |
| 应用价值 | 5 | 工业和交通业务收益、部署成本与推广路径 |

报告必须围绕评分项组织主表，不以算法名称数量代替量化提升。

---

## 10. Final Gate 清单

### 10.1 数据与模型

- [ ] 数据来源、license、split 和 hash 完整；
- [ ] train/validation/test 组级隔离通过；
- [ ] 边缘模型同时通过 G1/G3；
- [ ] 量化前后行为差异有逐样本审计；
- [ ] 模型、策略、图模型和可信度快照已冻结；
- [ ] 模型分发、签名校验、激活和回滚演示通过。

### 10.2 系统与实验

- [ ] 工业和交通场景均可独立运行；
- [ ] 五方案使用相同输入、硬件、profile 和评分器；
- [ ] G2/G4/G5 覆盖所有冻结弱网 profile；
- [ ] G6/G7 ground truth 经过脚本检查和人工抽样；
- [ ] 所有失败请求进入统计；
- [ ] 正式重复次数、seed 和聚合方式已冻结；
- [ ] G1-G7 可由一个汇总脚本重算。

### 10.3 交付物

- [ ] 作品报告 PDF；
- [ ] 运行效果视频 MP4；
- [ ] 源代码与可执行部署方式；
- [ ] Docker/环境依赖和一键演示脚本；
- [ ] 数据说明、模型说明和第三方 license；
- [ ] 全部核心 CSV、trace、配置、hash 和 manifest；
- [ ] README、部署手册、实验复现手册和答辩 FAQ；
- [ ] 压缩包按“单位-姓名-作品名称-联系电话”命名。

---

## 11. 完成定义

本项目只有同时满足以下条件才视为完成：

1. G1-G7 按本文固定口径全部达到赛题阈值；
2. 工业和交通两个场景在正常、弱网和短时断网下均有可演示闭环；
3. 云端复核、边缘自治、冲突仲裁、模型更新和 KWDB 状态底座均有真实运行证据；
4. 五方案对比、核心消融、稳定性和扩展性实验完整；
5. 结果能够由冻结代码、配置、模型和 manifest 复算；
6. 报告、视频、源码、数据说明和部署材料完整可提交。

任何单项模型实验成功、smoke 通过或开发集达到阈值，都不能替代上述完成定义。
