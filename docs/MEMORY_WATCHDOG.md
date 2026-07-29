# 60% 内存看门狗

`scripts/run_with_memory_guard.sh`用于保护训练、评测或模型服务任务。默认每
2秒检查一次主机 RAM；主机 RAM 连续两次达到或超过60% 时，先向被保护
任务的独立进程组发送 `SIGTERM`，10秒后仍未退出则发送`SIGKILL`。GPU
显存不参与采样、判断或任务终止。

看门狗只会终止由它启动的任务及其子进程，不会扫描或终止服务器上的其他
用户作业。任务必须通过该脚本启动才能获得主动保护。

```bash
bash scripts/run_with_memory_guard.sh \
  .venv/bin/python your_long_task.py
```

`scripts/run_p0a5.sh teacher-train`和`student-train`内部已经接入该看门狗，
不要再在外层重复套一层。

查看当前主机 RAM 使用率，不启动任务：

```bash
.venv/bin/python scripts/memory_watchdog.py snapshot
```

每次运行的状态文件写入`reports/runtime/memory_watchdog_*.json`，事件日志
追加到`logs/memory_watchdog.log`。因内存超限终止时，脚本退出码为75。

如果训练命令已经在独立进程组中运行、但原看门狗因终端或调度器退出而丢失，
可以在不重启训练的情况下接管其主PID：

```bash
.venv/bin/python scripts/memory_watchdog.py monitor \
  --pid TRAIN_MAIN_PID \
  --threshold-percent 60
```

接管模式会记录PID启动时间，避免PID复用后误杀其他任务；只有持续达到阈值时
才终止目标进程组。主动停止接管监控不会终止训练。

如需调整采样频率或连续次数，可设置环境变量，但正式运行保持60%阈值：

```bash
MEMORY_GUARD_INTERVAL_SECONDS=1 \
MEMORY_GUARD_CONSECUTIVE_SAMPLES=3 \
bash scripts/run_with_memory_guard.sh COMMAND
```
