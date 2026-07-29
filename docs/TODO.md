# TODO

## 当前P0-A5

- [x] 冻结Baseline-14B-AWQ及既有13,065题分母。
- [x] 决定Math/GSM8K、Code/OpenCodeInstruct→HumanEval、NLP/COIG-CQIA→CMMLU。
- [x] 删除旧96/170、MBPP/APPS/CodeContests/MMLU翻译和Adapter主线代码。
- [x] 完成36,673条训练数据、2,200条内部验证和300题门禁的CPU构建。
- [x] 通过P0-A5数据和协议审计。
- [x] 通过Teacher、Student和蒸馏dry-run。
- [ ] 人工启动4 GPU Teacher训练。
- [ ] Teacher端点生成并验证蒸馏数据。
- [ ] 训练最多两个共享Student候选。
- [ ] 合并、训练数据imatrix、Q4_K_M和Q8 KV。
- [ ] 最终量化模型通过唯一300题门禁。
- [ ] 通过≤1.5GB完整内存验证。
- [ ] 达到82%推荐线后运行一次13,065题正式全测。

## 系统指标

- [ ] 工业Fast Path闭环。
- [ ] 交通Fast Path闭环。
- [ ] TTFT降低≥75%。
- [ ] 弱网业务保持率≥90%。
- [ ] 两场景平均端到端时延≤0.2秒。
- [ ] 决策冲突率≤5%。
- [ ] 冲突解决成功率≥90%。
- [ ] 完成报告、视频、代码、数据说明和审计材料。
