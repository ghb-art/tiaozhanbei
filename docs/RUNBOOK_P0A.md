# P0-A2 DeepSeek 能力恢复运行手册

更新时间：2026-07-18

首轮 `G0-CAPMEM` 已关闭，9 个候选中没有一个同时满足能力与 1.5GB 峰值总内存门槛。当前只允许使用 `DeepSeek-R1-Distill-Qwen-1.5B` 的 Q2_K_S 内存安全基座做一次受控能力恢复；v24-v31 启动链已删除，历史结论和不可变证据保留在 `docs/REVISION_LOG.md` 与 `reports/audit/`。

## 1. 当前边界

- 基座：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`。
- 已知内存：Q2_K_S GGUF 716.71MB，20 次预热 + 100 次测量的峰值总内存 920.23MB。
- 已知阻塞：首轮 matched smoke 能力不达标，不能进入 P1 系统集成。
- 训练数据：GSM8K official train 教师验真数据、MBPP official train、synthetic CMMLU 格式恢复数据。
- 选模数据：GSM8K train 内部留出、MBPP dev_select、CMMLU official dev。
- 正式 GSM8K test、HumanEval 164 题、CMMLU test 不参与训练或选模。

## 2. 无 GPU 预检

```bash
bash scripts/run_p0a2.sh build-data
bash scripts/run_p0a2.sh preflight
```

数据门禁会生成：

- `data/distill/p0a2_recovery_train.jsonl`；
- `data/distill/p0a2_recovery_validation.jsonl`；
- `reports/audit/gate_p0a2_recovery_data.json`。

必须满足训练/验证 group overlap 为 0、正式集引用为 0、HumanEval prompt overlap 为 0。

## 3. 全精度能力上限

先用一题/任务验证环境和输出协议：

```bash
bash scripts/run_p0a2.sh upper-bound-smoke
```

再运行完整的 170 题独立 Dev：

```bash
bash scripts/run_p0a2.sh upper-bound
```

该结果是 DeepSeek 未量化基座的可达上限证据，不使用正式测试集。如果全精度基座在 Math/Code 上也明显无可达路径，应停止训练，转向结构化工具头，而不是从正式错题制造监督。

## 4. 受控 LoRA 恢复

默认使用 GPU 0-3；可通过 `P0A2_GPUS=0,1` 覆盖：

```bash
bash scripts/run_p0a.sh gpu-preflight
bash scripts/run_p0a2.sh train
bash scripts/run_p0a2.sh evaluate-adapter
```

训练固定使用 group 隔离 NLL validation、170 题外部 generation validation、父正确 token 保护、每 16 updates 选模和 best restore。generation macro accuracy 没有至少提升 1 个百分点时恢复 step 0，不允许因为训练 loss 下降而晋级。

## 5. 合并、重要性量化与 G0 回归

```bash
bash scripts/run_p0a2.sh export
bash scripts/run_p0a2.sh build-imatrix
bash scripts/run_p0a2.sh quantize
bash scripts/run_p0a2.sh g0-reentry
```

`g0-reentry` 重新执行同口径 matched capability smoke 和 20+100 请求峰值内存测试。只有 Math、Code、NLP 与宏平均保持率均不低于 80%，且峰值总内存不超过 1500MB，候选才能成为主边缘模型。

## 6. 常用检查

```bash
bash scripts/run_p0a.sh checks
bash scripts/run_p0a2.sh checks
bash scripts/run_p0a.sh g0-summary
```

不要恢复已删除的 v24-v31 启动命令，不要根据正式能力评测错误继续调参，也不要把 GGUF 文件大小当成运行峰值内存。
