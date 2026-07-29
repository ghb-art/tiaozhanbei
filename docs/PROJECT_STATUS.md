# Project Status

更新时间：2026-07-29

## 当前结论

项目仍处于P0能力模型阶段。正式边侧模型尚未冻结，不能声称已经同时通过能力、内存和系统指标。

当前唯一有效的能力路线为P0-A5：

- Baseline：冻结Qwen2.5-14B-Instruct-AWQ；
- Teacher：Qwen2.5-14B-Instruct BF16，4 GPU ZeRO-3 LoRA；
- Student：Qwen3-1.7B v1 HF合并基座，单一共享LoRA；
- 量化：Q4_K_M，Q8_0 KV；
- 小门禁：Math/Code/NLP各100题的唯一300题门禁；
- 正式测试：GSM8K 1,319、HumanEval 164、CMMLU 11,582。

## 已冻结历史证据

- Baseline-14B-AWQ正式准确率：GSM8K 91.736%、HumanEval 64.024%、CMMLU 75.928%。
- Student v1正式保持率：Math 82.149%、Code 68.571%、NLP 70.650%，未通过。
- P0-A4R3两个共享候选均出现Code明显回退，路线关闭。
- P0-A4R4候选结果只作为历史失败证据，不进入P0-A5选择。
- 旧96题、170题、MBPP、APPS、CodeContests和翻译MMLU协议退出当前代码主线。

历史结果只记录在`docs/REVISION_LOG.md`及已提交审计中，不得反馈到P0-A5训练。

## P0-A5数据

| 领域 | 训练 | 内部验证 | 小门禁 | 正式 |
|---|---:|---:|---:|---:|
| Math | GSM8K 7,173 | 200 | 100 | 1,319 |
| Code | OpenCodeInstruct 20,000 | 1,000 | 100 | HumanEval 164 |
| NLP | COIG-CQIA人工核验分层9,500 | 1,000 | CMMLU dev 100 | CMMLU test 11,582 |

Code必须由本地隔离执行再次验证；NLP必须按五类学科和推理配额选择；Math以30%损失权重回放并加入基座KL保持约束。

## 晋级规则

- 任意领域保持率低于78%：拒绝；
- 三项达到78%但未全部达到82%：只允许第二个预注册候选；
- 三项和截断宏平均达到82%：推荐进入内存验证和正式全测；
- 正式全测三项和宏平均仍必须分别达到80%。

## 当前操作边界

本轮只进行CPU数据构建、代码重构、协议审计和训练dry-run，不启动GPU。

CPU准备完成后的下一条人工命令是：

```bash
P0A5_GPUS=0,1,2,3 bash scripts/run_p0a5.sh teacher-train
```

## 系统侧未完成项

即使能力门通过，仍需完成：

- 工业和交通两个闭环；
- Fast Path平均端到端时延≤0.2s；
- TTFT降低≥75%；
- 弱网业务保持率≥90%；
- 冲突率≤5%；
- 冲突解决成功率≥90%；
- 完整CPU和设备内存统计≤1.5GB。
