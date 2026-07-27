# TODO

更新时间：2026-07-27

历史实验与失败原因见 `docs/REVISION_LOG.md`；本页只保留当前可执行项。

## 已完成

- [x] G-DATA：数据接入、冻结 split、泄漏检查和 manifest。
- [x] G-DB：KWDB schema、写入、查询和导出 live gate。
- [x] G-CLOUD：Qwen2.5-14B-AWQ 服务与 Cloud gate。
- [x] 首轮 G0：9 个候选 `feasible=0/9`。
- [x] DeepSeek BF16 170 条 Dev：Math 84.38%、Code 23.81%、NLP 17.19%，P0-A2 关闭。
- [x] 冻结 170 条 selection-only Dev，组重叠和正式集泄漏均为 0。
- [x] 实现 P0-A3 HF/HTTP 同口径评测、逐任务 Teacher 保持率和安全启动器。
- [x] 将 Qwen3 Q3 设为主候选、Qwen2.5-1.5B Q3 设为条件备用。
- [x] 删除 P0-A3 不再调用的 CEDD/LoRA 训练、adapter 合并和旧 G1 汇总代码。
- [x] 下载 `Qwen/Qwen3-1.7B`（12 个文件，4,079,450,110 bytes），并完成 Teacher 端点与 Qwen3 HF 三任务实机烟测，生成错误均为 0。
- [x] 完成Teacher 170条分母及Qwen3 HF 170条Dev；逐任务与宏保持率均≥80%。
- [x] 使用768条train-only校准文本生成1,073,242,688-byte Q3_K_M，正式测试集引用为0。
- [x] 定位首次Q3 Dev的20/170中断为llama.cpp PEG非法UTF-8 HTTP 500，并以可复现兼容补丁通过原失败题和三任务抽样。

## P0-A4 已完成的历史阶段

- [x] 冻结官方完整13065题独立清单，旧500/164/1000清单保持不变。
- [x] 构建训练集、Teacher验证96、Student烟测96和选择170，全部按组互斥且正式测试引用为0。
- [x] 实现4卡ZeRO-3 Teacher LoRA、正确性过滤蒸馏、Student共享/专家LoRA、合并、Q3量化入口。
- [x] 实现96题75%、170题80%、1400MB和一次正式全测门禁及次数账本。
- [x] 完成AWQ完整分母、Teacher训练/路由、Student v1蒸馏与Q4_K_M量化。
- [x] 完成v1的96、170、内存预检及一次提前授权的官方全测；Code保持率未达标。
- [x] 验证v2、P0-A4R任务Adapter和P0-A4R2温和Code Adapter均无晋级收益并终止。
- [x] 保存P0-A4代码、配置、文档和必要审计快照；模型、原始数据和密封逐题结果排除Git。

## P0-A4R3 当前执行顺序

- [x] 注册v1 HF合并基座共享蒸馏协议；Math冻结回放，禁止继续训练任务Adapter。
- [x] 固定两个候选：Rank 8与Rank 16共享LoRA；保持Q4_K_M、Q8 KV和train-only imatrix。
- [x] 扩展APPS至2500个执行通过任务，并构建1000个CodeContests训练任务及256个新验证任务。
- [x] 生成3000个八领域中文选择题短理由样本，Teacher输出逐条通过训练标签校验。
- [x] 建立Code 256、NLP 256、Math 128的全新train-only验证集并通过去重/防泄漏审计。
- [ ] 冻结v1新验证集分数，训练和评测Candidate 1；只有必要时再运行Candidate 2。
- [ ] 候选通过Math防遗忘和Code/NLP增益门后，合并并按Q4_K_M+Q8 KV量化。
- [ ] 量化候选依次通过96、170和20+100请求内存门；不重复既有正式全测。
- [x] 实现工业/交通Fast Path、弱网自治、动态调度、持久化outbox和冲突仲裁开发骨架与仿真。
- [ ] 在正式双场景部署复测0.2秒、弱网保持率和一致性，仿真结果不得替代该证据。

## G0 通过后

- [ ] P1：工业检测和交通监控的两个边缘节点 + 一个云节点最小闭环。
- [ ] P2：弱网自治、路由、通信与 0.2s 端到端实验。
- [ ] P3：多边缘关系组、冲突检测、唯一仲裁和一致性指标。
- [ ] Final Freeze 后一次性运行正式 G1–G7，生成报告、视频和部署包。

## 不可变约束

- 不使用 GSM8K test、HumanEval 或 CMMLU test 错题训练、选模或反复调参。
- 不恢复 P0-A2 DeepSeek 训练，也不恢复 v24–v31 编号式调参链。
- 96/170/正式全量均须与AWQ分母使用完全一致的sample ID、prompt和scorer。
- G1 各任务及宏平均保持率均须 ≥80%；G3 峰值总内存须 ≤1500MB（decimal）。
- 评测执行 `passed`、模型文件大小和 CPU 秒级短输出都不是比赛指标达标结论。
