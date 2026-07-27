# P0-A4R3 共享 Code+NLP 蒸馏运行手册

## 目标与边界

本路线从已合并的Student v1 HF基座重新训练共享LoRA。Math能力冻结，不再进行专项训练；
训练集加入Math回放，候选选择加入Math防遗忘门。Rank-4和v2任务Adapter不再使用。

只允许两个预注册候选：

- Candidate 1：共享LoRA Rank 8、alpha 16、学习率`5e-5`、2轮；
- Candidate 2：共享LoRA Rank 16、alpha 32、学习率`3e-5`、2轮。

旧42题、96题、170题和正式全测逐题结果都不能用于训练、提示词选择或超参数调整。
模型、下载的原始数据、蒸馏数据、逐题评测trace和密封结果均不提交Git。

## 1. 结构和协议检查

```bash
bash scripts/run_p0a4r3.sh structural-check
bash scripts/run_p0a4r3.sh protocol
```

协议审计必须为`passed`，否则后续命令停止。

## 2. 构建新的训练数据

先将APPS执行通过任务扩展到2500组，再下载CodeContests train的parquet分片并构建
1000组训练、256组全新验证任务。CodeContests只接受Python 3程序，参考解答必须在隔离
子进程中通过stdin/stdout测试。

```bash
bash scripts/run_p0a4r3.sh apps-expand
bash scripts/run_p0a4r3.sh code-download
bash scripts/run_p0a4r3.sh code-build
```

NLP请求来自MMLU auxiliary train八个领域。Teacher提示中不包含标签；生成结果必须包含
中文题目、中文选项、中文短理由和独立最终选项，并由构建器使用训练标签再次校验。
请求池预留35%冗余；最终只从校验通过的候选中按八领域均衡冻结3000条训练和256条验证，
不会要求Teacher对候选池达到不现实的100%接受率。
每个领域至少保留完全等额配额的80%，剩余名额在仍有校验通过样本的领域间轮转分配；
不得用复制样本或降低标签校验标准填补配额。
前两次独立判断使用NLP LoRA Teacher，第三次使用同一14B BF16基座复核；提示中始终不包含
标签，最终样本记录实际通过校验的Teacher身份和尝试序号。

```bash
bash scripts/run_p0a4r3.sh nlp-prepare

# 另一个终端先启动4 GPU Teacher端点，再执行：
P0A4R3_TEACHER_ENDPOINT=http://127.0.0.1:8000 \
P0A4R3_TEACHER_MODEL_ID=distill-teacher-v1 \
bash scripts/run_p0a4r3.sh nlp-generate
```

最后组合MBPP/APPS/CodeContests、中文NLP和Math回放：

```bash
bash scripts/run_p0a4r3.sh assemble
bash scripts/run_p0a4r3.sh preflight
```

组合审计必须确认训练集至少Math/Code/NLP=`1000/3500/3000`组，验证集为
`128/256/256`组，训练/验证交集和受保护测试引用都为0。

## 3. 冻结v1基线并训练候选

先在新train-only验证集评测v1。该分数冻结后不得重建验证集。

```bash
bash scripts/run_p0a4r3.sh evaluate-base
bash scripts/run_p0a4r3.sh train 1
bash scripts/run_p0a4r3.sh merge 1
bash scripts/run_p0a4r3.sh evaluate 1
bash scripts/run_p0a4r3.sh select
```

Candidate 1若通过选择门就不训练Candidate 2。若Candidate 1失败，才允许运行第二个也是
最后一个候选：

```bash
bash scripts/run_p0a4r3.sh train 2
bash scripts/run_p0a4r3.sh merge 2
bash scripts/run_p0a4r3.sh evaluate 2
bash scripts/run_p0a4r3.sh select
```

内部晋级条件：

- 生成错误为0；
- Math相对v1保持率不低于95%；
- Code和NLP均不低于v1；
- Code与NLP准确率平均至少净增1个百分点。

## 4. 量化与正式门禁

只有选择审计为`passed`时才能量化：

```bash
bash scripts/run_p0a4r3.sh quantize
```

量化固定为训练数据imatrix、Q4_K_M权重和Q8 KV Cache。之后使用P0-A4既有同口径入口
顺序运行96题、170题和内存门。未通过时不得读取逐题结果生成新监督，也不得重复既有
官方完整全测。

## 5. 系统侧并行验证

开发仿真入口：

```bash
.venv/bin/python scripts/simulate_cloud_edge.py \
  --config configs/cloud_edge_runtime.json \
  --audit reports/audit/gate_cloud_edge_system_simulation.json
```

仿真覆盖工业和交通Fast Path、弱网强制边缘路由、幂等outbox、云失败回退和安全优先
冲突仲裁。报告必须标记为simulation；比赛最终0.2秒、弱网业务保持率≥90%、冲突比例≤5%
和冲突解决率≥90%仍需在正式双场景部署上复测。
