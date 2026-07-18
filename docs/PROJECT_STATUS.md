# Project Status

更新时间：2026-07-18

## 阶段结论

项目仍位于 P0。首轮 `G0-CAPMEM` 为 `failed, feasible=0/9`，没有候选同时满足 G1 能力与 G3 峰值总内存，因此不得进入完整云边调度、双场景和 Final Gate。

| 角色 | 模型 | 结果 |
|---|---|---|
| Cloud Teacher | Qwen2.5-14B-Instruct-AWQ | 保留，作为 G1 分母、蒸馏教师和云端复核模型 |
| 历史能力最佳 | Qwen2.5-3B v25 router | Math 87.53%、Code 75.00%、NLP 88.68%；Code 未过 G1，低位量化内存也不可行 |
| 已拒绝候选 | Qwen3-1.7B IQ2_XS | 峰值 1306.14MB，但 matched capability 失败 |
| 当前恢复基座 | DeepSeek-R1-Distill-Qwen-1.5B Q2_K_S | GGUF 716.71MB、峰值 920.23MB；能力失败，尚未晋级 |

CPU llama.cpp 的 16-token 单次推理均值为 2.58s，只能作为排除证据，不能声称满足 0.2s 端到端指标。后续系统路径仍需结构化 fast path 或可复现实验加速设备。

## P0-A2 已落地

- `configs/p0a2_recovery.json` 冻结 DeepSeek 基座、数据来源、训练和量化参数。
- `build_p0a2_recovery_data.py` 生成 7212 条 train-only 数据和 170 条 selection-only Dev 数据。
- 数据审计确认 train/validation group overlap、正式集引用、HumanEval prompt overlap 均为 0。
- `evaluate_p0a2_recovery.py` 支持未量化基座与 LoRA 的同口径 Math/Code/NLP Dev 评测。
- `run_p0a2.sh` 串联 preflight、全精度上限、四卡受控训练、合并、模型专用 imatrix、Q2 量化和 G0 回归。
- 当前 G0 配置只保留已拒绝 DeepSeek baseline 与一个 P0-A2 active candidate。
- `upper-bound-smoke` 已完成三任务各 1 条的真实 GPU 链路验证，无加载或评分异常；仅 GSM8K 正确，宏准确率 `1/3`，说明当前基座输出格式/能力仍需恢复。该 smoke 只证明链路可运行，不作为能力达标结论。

## 保留与清理边界

保留本地 14B Teacher、DeepSeek 1.5B HF 基座、DeepSeek baseline F16/Q2/imatrix 以及小型历史审计。Qwen 1.5B/3B/Qwen3 权重、历史 adapter、v24-v31 启动代码和大规模中间训练数据已不在当前路线；它们可以依据模型 ID、数据来源与审计重新生成。

## 下一退出条件

1. 完成未量化 DeepSeek 的 170 题独立 Dev 上限；
2. 完成一轮受 generation validation 保护的 train-only LoRA 恢复；
3. 合并并使用该模型自身生成的 importance matrix 重新量化；
4. 同一候选重新通过 G1 matched smoke 与 20+100 请求 G3；
5. 仅在 G1/G3 同时通过后进入 P1 两场景最小闭环。

首轮失败证据固定为 `reports/audit/gate_g0_capmem.json`。不得根据正式测试错题制作监督，也不得降低 80% 或 1500MB 门槛来晋级。
