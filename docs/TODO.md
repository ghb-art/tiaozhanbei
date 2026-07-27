# TODO

更新时间：2026-07-19

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

## P0-A4 当前执行顺序

- [x] 冻结官方完整13065题独立清单，旧500/164/1000清单保持不变。
- [x] 构建训练集、Teacher验证96、Student烟测96和选择170，全部按组互斥且正式测试引用为0。
- [x] 实现4卡ZeRO-3 Teacher LoRA、正确性过滤蒸馏、Student共享/专家LoRA、合并、Q3量化入口。
- [x] 实现96题75%、170题80%、1400MB和一次正式全测门禁及次数账本。
- [ ] 运行AWQ的Teacher验证、烟测96、选择170分母，并运行一次官方完整分母。
- [ ] 下载BF16 Teacher，训练最多三个候选并在独立验证集选优。
- [ ] 生成校验蒸馏数据，训练Student v1并重新量化Q3_K_M+Q8 KV。
- [ ] 依次通过96、170和20+100请求内存门；失败时最多再训练Student v2。
- [ ] 全部Dev门通过后只运行一次13065题Student正式测试，之后禁止回训。
- [ ] Student冻结后删除确认无用的DeepSeek权重和中间GGUF，只保留拒绝审计与逐样本trace。
- [ ] 并行设计确定性结构化 fast path，但与纯模型 G1 分开报告。

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
