# TODO

更新时间：2026-07-18

历史实验与失败原因见 `docs/REVISION_LOG.md`；本页只保留当前可执行项。

## 已完成

- [x] G-DATA：数据接入、冻结 split、泄漏检查和 manifest。
- [x] G-DB：KWDB schema、写入、查询和导出 live gate。
- [x] G-CLOUD：14B-AWQ 服务与 Cloud gate。
- [x] 首轮 G0：9 个候选的能力-峰值内存联合审计，结论 `feasible=0/9`。
- [x] DeepSeek Q2 基座 G3：716.71MB GGUF、920.23MB 峰值总内存。
- [x] P0-A2 数据冻结：train 7212、Dev 170、组重叠和正式集泄漏均为 0。
- [x] P0-A2 代码：全精度 Dev、受控 LoRA、合并、模型专用 imatrix、Q2 量化和 G0 回归入口。
- [x] 清理已否决 Qwen 权重、历史 adapter 与 v24-v31 启动代码，保留小型审计证据。

## 当前执行顺序

- [x] 运行 `upper-bound-smoke`，验证 DeepSeek HF 输出与评测协议；三任务各 1 条无运行错误，宏准确率 `1/3`，不代表能力达标。
- [ ] 运行完整 `upper-bound`，记录三任务全精度能力上限。
- [ ] 若上限有可达路径，运行一次 `train` 和 `evaluate-adapter`。
- [ ] 只有 frozen Dev 宏准确率至少提升 1 个百分点才合并候选。
- [ ] 为合并模型重新生成 imatrix 并量化 Q2_K_S。
- [ ] 执行 `g0-reentry`；G1/G3 任一失败即停止 P0-A2 晋级。
- [ ] 并行设计确定性结构化 fast path，但分数与纯 LLM G1 分开报告。
- [ ] 在可复现实验硬件上建立 TTFT/E2E 基线。

## G0 通过后

- [ ] P1：工业检测和交通监控的两个边缘节点 + 一个云节点最小闭环。
- [ ] P2：弱网自治、路由、通信与 0.2s 端到端实验。
- [ ] P3：多边缘关系组、冲突检测、唯一仲裁和一致性指标。
- [ ] Final Freeze 后一次性运行正式 G1-G7，生成报告、视频和部署包。

## 不可变约束

- 不使用 GSM8K test、HumanEval 或 CMMLU test 错题训练/选模。
- 不恢复 v24-v31 编号式调参链。
- G1 四项均须 ≥80%；G3 峰值总内存须 ≤1500MB（decimal）。
- CPU 2.58s 短输出不是 0.2s 达标结果。
