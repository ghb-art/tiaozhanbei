# P0-A3 边缘模型重新选型运行手册

更新时间：2026-07-18

DeepSeek P0-A2 已由完整 BF16 Dev 关闭：Math 84.38%、Code 23.81%、NLP 17.19%。当前只执行能力优先的 P0-A3：Qwen3-1.7B Q3_K_M 为主候选，Qwen2.5-1.5B Q3_K_M 为条件备用。

## 1. 无模型预检

```bash
bash scripts/run_p0a3.sh preflight
bash scripts/run_p0a3.sh checks
```

预检要求 170 条 Dev 完整、训练/验证组隔离、正式集引用为 0，并核对 DeepSeek 拒绝证据。`run_p0a2.sh` 已禁用，任何旧训练命令都会直接退出。

## 2. 恢复主候选

```bash
bash scripts/run_p0a3.sh download-primary
```

只下载 `Qwen/Qwen3-1.7B`。不要恢复 Qwen3 IQ2_XS；它虽然峰值为 1306.14MB，但 matched capability 已失败。

## 3. 建立同口径 Teacher 分母

终端 A：

```bash
bash scripts/run_p0a.sh teacher-plan
bash scripts/run_p0a.sh teacher-serve
```

终端 B：

```bash
bash scripts/run_p0a3.sh teacher-dev
```

默认 `P0A_GPUS=0,1,2,3`，启动一个端口为 8000 的 vLLM 服务，并将 Qwen2.5-14B-AWQ 以 tensor parallel 4 分布到四张 GPU；不是启动四个单卡副本。`teacher-serve` 会先检查四卡显存，旧单卡 Teacher 尚未停止时会拒绝启动。`teacher-plan` 只打印 GPU 组和最终 vLLM 参数，不占用显存。

默认访问 `http://127.0.0.1:8000`。如端口不同，设置 `P0A3_TEACHER_URL`。评测器会通过 `/v1/models` 自动发现 vLLM 实际发布的模型 ID，避免服务使用绝对路径、客户端使用相对路径时产生 HTTP 404；只有服务同时发布多个模型时，才需要显式设置 `P0A3_TEACHER_MODEL_ID`。Teacher、HF 候选和 Q3 候选必须使用 `p0a2_recovery_validation.jsonl` 中完全相同的 170 个 sample ID。

## 4. Qwen3 未量化能力 Gate

```bash
bash scripts/run_p0a3.sh qwen3-hf-smoke
bash scripts/run_p0a3.sh qwen3-hf
```

Qwen3 使用 `enable_thinking=false`，三任务 token 上限固定为 CMMLU 256、GSM8K 512、Code 512。`qwen3-hf` 结束后自动计算：

完整 `teacher-dev` 结束后，在终端 A 按 `Ctrl+C`，或在另一终端运行 `bash scripts/run_p0a.sh teacher-stop`，停止四卡 Teacher。确认显存释放后再运行 Qwen3 HF。HF 评测仍会从 `nvidia-smi` 自动选择空闲显存最多的 GPU；需要固定设备时可使用 `P0A3_EVAL_GPU=1 bash scripts/run_p0a3.sh qwen3-hf`。四卡 Teacher 运行期间不得同时加载本地候选。

```text
Math_ratio = Qwen3_GSM8K / Qwen14_GSM8K
Code_ratio = Qwen3_MBPP / Qwen14_MBPP
NLP_ratio  = Qwen3_CMMLU / Qwen14_CMMLU
```

三项和 capped macro 均须 ≥80%，生成错误须为 0。任一失败即停止，不训练、不量化，并留下 `gate_p0a3_qwen3_1p7b_hf_retention.json`。

## 5. Importance-aware Q3 与同集复验

仅在 HF Gate 通过后：

```bash
bash scripts/run_p0a3.sh prepare-qwen3
bash scripts/run_p0a3.sh qwen3-q3-dev
```

校准文本只来自 train-only 数据。Q3 使用Q8 KV cache，并必须在同一170条上再次通过逐任务80%保持率，不能以平均分掩盖单项失败。Q4 KV已被F16/Q3首token logits对照明确否决，不得为了节省内存恢复。

项目固定llama.cpp提交 `2d973636e292ee6f75fadcf08d29cb33511f509f`，并由 `setup_llama_cpp.sh` 应用 `patches/llama_cpp_chat_utf8_sanitize.patch`。该补丁避免Q3偶发非法UTF-8字节触发PEG HTTP 500，只把非法字节保留为显式替代字符，不改logits或评分。若Q3 trace不足170条，必须先排除运行错误，不能把缺失任务的0分当成模型拒绝。

若完整 Q3 运行无生成错误但能力相对 HF 异常大幅下降，先运行同后端 F16 对照：

```bash
bash scripts/run_p0a3.sh qwen3-f16-control
```

该命令固定使用F16 GGUF与F16 KV，只验证GGUF转换和llama.cpp能否复现HF。原F16+Q4 KV结果仅保留为失败配置证据；它与HF的token ID完全一致，但logits明显偏移，不能用于否决GGUF或Q3权重。F16对照不是边侧候选，不参与内存Gate，也不能解锁正式G0。

## 6. 内存与一次正式 G0

```bash
bash scripts/run_p0a3.sh qwen3-memory
bash scripts/run_p0a3.sh qwen3-formal-g0
```

内存固定 20 次预热 + 100 次测量、50ms 采样、batch 1、context 512、CPU llama.cpp 完整进程树。P0-A3 Dev 要求峰值 ≤1400MB，为正式 1500MB 留出余量。正式 G0 只有在 HF Dev、Q3 Dev 和内存 Dev 全部通过后才可运行一次。

## 7. 条件备用路线

仅当 Qwen3 的 HF Dev、Q3 Dev 或 1400MB 内存 Gate 之一形成明确拒绝审计时：

```bash
bash scripts/run_p0a3.sh download-fallback
bash scripts/run_p0a3.sh qwen25-hf
bash scripts/run_p0a3.sh prepare-qwen25
bash scripts/run_p0a3.sh qwen25-q3-dev
bash scripts/run_p0a3.sh qwen25-memory
bash scripts/run_p0a3.sh qwen25-formal-g0
```

备用路线使用完全相同的数据、Teacher 分母、量化和内存协议。不得同时调两个候选，也不得根据正式 G1 错题返回修改模型。

## 8. 结果解释

- `evaluate_edge_candidate_dev.py` 中的 `status=passed` 只在配置准确率阈值时代表绝对能力门；P0-A3 的晋级结论以 `summarize_edge_candidate_dev.py` 保持率审计为准。
- HF GPU 显存不是 G3；G3 使用 `verify_gate_g3_gguf.py` 的完整进程树 RSS + 设备内存。
- 170 条是 selection-only Dev，不是正式 GSM8K test、HumanEval 或 CMMLU test。
- 任何候选只有 G1/G3 同时通过才可写入正式边缘模型名称。
