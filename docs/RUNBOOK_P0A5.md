# P0-A5 单门禁共享蒸馏运行手册

## CPU准备

```bash
bash scripts/run_p0a5.sh data-download
bash scripts/run_p0a5.sh data-build
bash scripts/run_p0a5.sh preflight
bash scripts/run_p0a5.sh status
```

这些命令不启动GPU。预期通过：

- `reports/audit/gate_p0a5_data.json`
- `reports/audit/gate_p0a5_protocol.json`
- `reports/audit/gate_p0a5_teacher_preflight.json`
- `reports/audit/gate_p0a5_student_preflight.json`

## 第一个GPU步骤

确认没有其他GPU训练任务后：

```bash
P0A5_GPUS=0,1,2,3 \
bash scripts/run_p0a5.sh teacher-train \
2>&1 | tee logs/p0a5_teacher_train.log
```

实时查看：

```bash
tail -F logs/p0a5_teacher_train.log
```

训练完成后再执行：

```bash
bash scripts/run_p0a5.sh teacher-plan
```

该命令只打印Teacher服务计划，不启动服务。

## 后续顺序

在单独终端启动Teacher：

```bash
P0A5_GPUS=0,1,2,3 bash scripts/run_p0a5.sh teacher-serve
```

另一个终端生成蒸馏数据：

```bash
bash scripts/run_p0a5.sh distill-generate \
2>&1 | tee logs/p0a5_distill.log
```

随后：

```bash
bash scripts/run_p0a5.sh student-preflight 1
P0A5_GPUS=0,1,2,3 bash scripts/run_p0a5.sh student-train 1
bash scripts/run_p0a5.sh student-merge 1
bash scripts/run_p0a5.sh imatrix-corpus
```

量化后的Q4_K_M模型必须使用Q8_0 KV运行唯一300题门禁。只有审计中的`recommended_full=true`，才允许进入内存测试和13,065题正式全测。

## 禁止事项

- 不再运行旧P0-A4、P0-A4R、96题或170题命令；
- 不使用MBPP、APPS、CodeContests或MMLU翻译数据；
- 不根据正式逐题结果训练；
- 不用HF/F16门禁替代Q4_K_M门禁；
- 不覆盖已有模型、密封结果或训练目录。
