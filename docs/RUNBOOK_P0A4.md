# P0-A4 Teacher–Student运行手册

## 1. 不可变口径

- 正式分母固定为`Qwen2.5-14B-Instruct-AWQ`，4张GPU组成一个TP=4端点，不参与调优。
- 蒸馏Teacher为单独的Qwen2.5-14B BF16快照，使用4卡ZeRO-3 LoRA训练。参数和
  `adamw_torch` LoRA优化器均由ZeRO-3在GPU间分片，不启用CPU参数/优化器offload，
  从而避免依赖本机CUDA工具链编译`DeepSpeedCPUAdam`。
- Student为Qwen3-1.7B共享蒸馏模型，部署格式固定Q4_K_M，KV Cache固定Q8。Q3_K_M保留为失败对照，不再进入170或正式全测。
- 官方完整集为GSM8K 1319、HumanEval 164、CMMLU 11582；旧500/164/1000清单不得覆盖。
- 正式逐题trace存入`reports/sealed/p0a4`，训练器硬拒绝该目录、`data/eval`和官方完整split。

## 2. 准备与分母

```bash
bash scripts/run_p0a4.sh preflight
bash scripts/run_p0a4.sh baseline-plan
# 终端A
bash scripts/run_p0a4.sh baseline-serve
# 终端B
bash scripts/run_p0a4.sh baseline-dev
bash scripts/run_p0a4.sh baseline-full
```

`baseline-full`启动四个确定性评测分片，共用一个TP=4 vLLM端点。合并器要求13065个ID无重复、无遗漏、无额外项，随后将trace改为只读并登记一次性账本。

## 3. Teacher

```bash
bash scripts/run_p0a4.sh install-training-deps
bash scripts/run_p0a4.sh download-teacher
bash scripts/run_p0a4.sh teacher-train 1

# 终端A，设置版本后启动BF16基座+LoRA
P0A4_TEACHER_VERSION=1 bash scripts/run_p0a4.sh teacher-plan
P0A4_TEACHER_VERSION=1 bash scripts/run_p0a4.sh teacher-serve

# 终端B
bash scripts/run_p0a4.sh teacher-validate 1
bash scripts/run_p0a4.sh teacher-select
bash scripts/run_p0a4.sh teacher-distill 1
```

Teacher推理采用显式任务Top-1路由：数学、代码使用未合并LoRA的BF16基座，中文NLP使用
对应版本LoRA。该策略来自独立Teacher验证集，不接触正式测试集；验证trace会逐题记录实际
`endpoint_model_id`，蒸馏trace也会记录实际`served_model_id`，避免把路由结果误记成单一模型。

当前v1路由后的96题结果为Math/Code/NLP=`32/32、26/32、31/32`，宏准确率`92.708%`，
生成错误为0，Teacher选择门已通过。通常不需要再训练v2，可直接生成蒸馏数据。若确需训练
后备v2/v3，配置已限制代码小样本的最大重复倍数，并降低LoRA rank、学习率和训练轮数，
避免复现v1第2轮验证损失上升及代码能力回退。最多仍只允许三个Teacher候选。

蒸馏生成器只接受数学最终答案一致、代码执行通过、NLP选项标签一致的样本；启动时会检查
基座和路由Adapter的模型ID均由端点实际提供，缺失时直接失败，不再静默回退到列表中的
第一个模型。

训练池中的NLP数据必须是唯一问题：当前使用排除170题选择集后的CMMLU-dev中文训练题，
并用MMLU auxiliary非正式训练题补足到512条。蒸馏门按任务分别检查唯一题数和接受率，
禁止用数学样本的高通过率掩盖代码或NLP不足。当前正确性过滤结果为Math/Code/NLP=
`1952/233/440`，对应接受率`98.39%/79.79%/85.94%`。Student共享训练最多将最少类
上采样2倍，实际平衡为每类466条。

## 4. Student、量化与门禁

```bash
bash scripts/run_p0a4.sh student-train 1
bash scripts/run_p0a4.sh student-merge 1
bash scripts/run_p0a4.sh student-check 1

# 可选Adapter-MoE消融
bash scripts/run_p0a4.sh student-experts 1
bash scripts/run_p0a4.sh prepare-adapters 1

bash scripts/run_p0a4.sh student-quantize
bash scripts/run_p0a4.sh build-llama-cuda
bash scripts/run_p0a4.sh edge-start
bash scripts/run_p0a4.sh student-smoke96
# 当前Q4_K_M候选通过96题后，先执行关闭主机提示缓存的5+20内存预检
bash scripts/run_p0a4.sh student-memory-precheck-q4
bash scripts/run_p0a4.sh student-170-check 1
bash scripts/run_p0a4.sh student-170 1
bash scripts/run_p0a4.sh student-memory
bash scripts/run_p0a4.sh student-full
bash scripts/run_p0a4.sh edge-stop
```

当前v1训练和合并已通过；共享模型最佳权重来自checkpoint-21。`student-quantize`会先复核训练、
合并模型哈希，再从Math/Code/NLP各128条纯训练数据生成均衡imatrix，禁止使用96/170或正式
测试数据。量化完成后同一入口还会核对Q4、imatrix和上游合并模型哈希。Q4的96题保持率为
Math/Code/NLP=`87.5%/75%/95%`、宏保持率`85.83%`，关闭主机提示缓存后的5+20内存峰值为
`1211.20MB`；旧Q3代码保持率仅`50%`，不得误用。

v1随后在170题上得到Math/Code/NLP=`90.625%/75.000%/80.435%`、宏保持率`82.020%`，
因Code低于80%而失败。此前按用户显式要求绕过170和完整内存门运行的一次官方全测已经封存，
保持率为Math/Code/NLP=`82.149%/68.571%/70.650%`、宏`73.790%`；该结果不参与v2调参，
官方全测账本名额也已经消耗。

### 4.1 Student v2（当前下一步）

v2只读取v1的170题三任务汇总，不读取逐题输出或正式全测结果。共享训练把蒸馏样本数冻结为
Math/Code/NLP=`466/932/880`，重点补强Code和NLP；同时训练三个低秩任务Adapter。共享路线与
Top-1 Adapter路线都重新通过同一96题后，只有每项及宏保持率均达到85%的路线才可冻结，
以保护仅剩的一次170题候选机会。

```bash
# 已完成时可跳过；该命令只冻结协议并dry-run，不启动训练
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh student-v2-preflight

# 4卡DDP训练共享v2，然后合并、训练任务Adapter并量化共享基座
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh student-train 2
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh student-merge 2
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh student-experts 2
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh prepare-adapters 2
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh student-quantize 2

# 共享Q4路线：终端A启动，终端B评测，然后停止并做独立内存预检
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh edge-start
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh student-smoke96
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh edge-stop
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh student-memory-precheck-q4

# Adapter Top-1路线：同样重新跑96题和全Adapter常驻内存预检
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh edge-start-adapters
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh student-adapter-smoke96
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh edge-stop
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh student-adapter-memory-precheck 2

# 用96题汇总冻结唯一路线；不通过85%安全线就停止，不消耗最后一次170题机会
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh student-v2-select-route

# 通过后：终端A启动冻结路线，终端B消耗最后一次170题机会并做20+100内存门
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh edge-start-v2-selected
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh student-v2-170
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh edge-stop
P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh student-v2-memory
```

`edge-start`会写入受管服务审计，绑定实时PID、Q4模型哈希、Q8 KV、关闭思考和关闭主机提示缓存等参数。`student-170-check`不占用170题候选名额；它会同时复核Q4量化来源、96题trace、冻结的14B分母、内存预检、实时服务身份和候选账本。只有显示`READY`后才执行`student-170`。
边侧服务默认用`P0A4_EDGE_GPUS=0`绑定一张RTX 3090，CUDA服务并发为4，总上下文6144，保证每个槽位仍有1536 token。服务审计必须在`nvidia-smi`中找到同一PID的CUDA显存分配，否则拒绝评测。

96题要求Math/Code/NLP及capped macro全部≥75%。170题要求全部≥80%，只可读取任务级汇总；失败后只允许基于三任务汇总调整一次v2，禁止查看逐题错误。量化模型必须重新过门，HF成绩不能代替。

所有Qwen3 Student评测都必须在请求中显式设置`enable_thinking=false`，并在审计中记录
`disable_thinking=true`和`kv_cache_type=q8_0`。启动参数`--reasoning off`不能替代请求级
审计；任一参数缺失时，运行脚本会在保持率计算前拒绝该评测报告。

内存门使用20次预热、100次测量、batch 1、context 512、50ms采样，统计进程树RSS与设备内存，开发峰值≤1400MB。边侧服务和内存门都固定`--cache-ram 0 --no-cache-idle-slots`，关闭llama.cpp默认的主机提示缓存；该缓存属于可选的重复提示加速状态，不计入本方案部署能力，开启时会为不同提示持续保留上下文副本。请求级`cache_prompt=false`不能替代服务端关闭。审计必须记录`host_prompt_cache_mib=0`和`cache_idle_slots=false`。可选Adapter需把所有已加载Adapter、请求切换和加载时延计入；超限时rank 8降为4，仍失败则回退共享Student。

## 5. 正式失败规则

`student-full`只有在170题及内存审计均通过后解锁，账本只允许一个Student版本。正式保持率任一任务或宏平均低于80%时，保存失败报告并结束当前路线；不得根据正式逐题结果生成训练数据、修改提示词或训练第三个Student。
