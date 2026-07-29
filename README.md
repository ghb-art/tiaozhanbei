# DB4AI-EdgeServe

面向云边协同场景的分布式人工智能感知与决策系统，题目编号XH-202606。

当前唯一有效的能力路线是P0-A5：

- 云端分母：冻结Qwen2.5-14B-Instruct-AWQ；
- 蒸馏Teacher：Qwen2.5-14B-Instruct BF16，4 GPU ZeRO-3；
- 边缘Student：Qwen3-1.7B v1合并基座，单一共享LoRA；
- 量化：Q4_K_M权重、Q8_0 KV；
- 小门禁：Math、Code、NLP各100题；
- 正式测试：GSM8K 1,319、HumanEval 164、CMMLU 11,582。

历史Student v1正式保持率为Math 82.149%、Code 68.571%、NLP 70.650%，未通过。旧96题、170题、MBPP、APPS、CodeContests、翻译MMLU和任务Adapter路线已经退出主线。

## 当前数据

| 领域 | 训练 | 内部验证 | 唯一小门禁 | 正式测试 |
|---|---:|---:|---:|---:|
| Math | GSM8K 7,173 | 200 | GSM8K 100 | GSM8K 1,319 |
| Code | OpenCodeInstruct 20,000 | 1,000 | OpenCodeInstruct 100 | HumanEval 164 |
| NLP | COIG-CQIA人工核验分层9,500 | 1,000 | CMMLU dev 100 | CMMLU test 11,582 |

Code训练题必须独立执行通过并与HumanEval去重。COIG-CQIA必须按考试、科学、社会人文、法商和语言推理五类配额选择。Student训练按Math/Code/NLP=`30/35/35%`损失权重混合，并对Math启用基座KL保持约束。

## CPU准备

```bash
bash scripts/run_p0a5.sh data-download
bash scripts/run_p0a5.sh data-build
bash scripts/run_p0a5.sh preflight
bash scripts/run_p0a5.sh status
```

上述命令不启动GPU。

CPU准备通过后的第一个GPU命令：

```bash
P0A5_GPUS=0,1,2,3 \
bash scripts/run_p0a5.sh teacher-train \
2>&1 | tee logs/p0a5_teacher_train.log
```

完整顺序见`docs/RUNBOOK_P0A5.md`，指标和数据隔离原则见`IMPLEMENTATION_PLAN.md`。

## 300题门禁

```text
任意领域保持率 < 78%  → 拒绝
三项 ≥78%但未全部≥82% → 只允许第二个预注册候选
三项及宏平均 ≥82%      → 推荐进入内存和正式全测
```

正式全测仍要求每项及宏平均≥80%，且正式逐题结果不得反馈训练。

## 系统架构

能力模型之外，项目继续实现：

- 工业缺陷和城市交通两个场景；
- 规则/轻量模型Fast Path；
- 边缘1.7B本地自治；
- 网络和任务复杂度感知的云边调度；
- 弱网outbox和恢复补传；
- GraphTrust多节点冲突仲裁；
- KWDB事件、状态、模型版本和审计存储。

比赛还要求TTFT降低≥75%、单次推理内存≤1.5GB、弱网保持率≥90%、两场景平均端到端时延≤0.2秒、冲突率≤5%和解决成功率≥90%。能力门通过不等同于比赛已经完成。

## 仓库原则

- 模型、原始数据、训练数据和密封逐题结果不进Git；
- Git提交代码、配置、文档、数据重建脚本和必要汇总审计；
- 正式测试集禁止用于训练、提示词选择和错题修复；
- 不覆盖已有模型或密封结果；
- 旧路线结论保留在`docs/REVISION_LOG.md`。
