# test/70.r32_tpr_perf — R=32 TPR=16/64/256 性能

- 日期：2026-10-05；机器：本机 202（C600UL，32 卡 Available、全 0%）
- 二进制：`./bench`（当前 src，kNSlice=4；`setarch x86_64 -R bash scripts/build.sh` 确定性构建）
- 口径：`perf.md` 现行 R=32 基准——`--ranks 32 --nodes 20 --graph 0 --perf-only 1
  --compute 1 --perf-pattern 0 --tokens T --max-tokens 512 --win-cap 128`
  （balanced；`full` = graph 链 mega_kernel(+REDUCE) us/kernel，min-of-5）
- 约束：每档单次 `timeout -k 3 10`（AGENTS 10s 规则），每档每次运行 2 遍取均值
- 脚本：`run.sh`；日志：`run1/t*.log`（重编译前）、`t*.log`（重编译后）

## 重编译一致性

| 项 | 值 |
|---|---|
| 重编译前 bench md5 | `37cfbbb4812ea08236270da4f8c266c0` |
| 重编译后 bench md5 | `37cfbbb4812ea08236270da4f8c266c0` |
| 编译命令 | `setarch $(uname -m) -R bash scripts/build.sh`（cucc，~13s，仅 warning） |

> 结论：确定性构建，binaries **逐字节一致**（md5 相同）。

## 结果（run2 = 重编译后）

| TPR | rep1 (µs) | rep2 (µs) | 均值 (µs) | 数值校验 |
|---:|---:|---:|---:|:--|
| 16 | 344.717 | 342.925 | **343.82** | 3 行 PASS / [总体] PASS |
| 64 | 652.237 | 658.432 | **655.33** | 3 行 PASS / [总体] PASS |
| 256 | 1913.933 | 1913.190 | **1913.56** | 3 行 PASS / [总体] PASS |

## 两次运行对照

| TPR | run1 均值 (µs) | run2 均值 (µs) | 差异 | perf.md v1 |
|---:|---:|---:|---:|---:|
| 16 | 347.73 | 343.82 | −1.1% | 345.9 |
| 64 | 651.74 | 655.33 | +0.6% | 655.1 |
| 256 | 1917.24 | 1913.56 | −0.2% | 1914.8 |

> 结论：重编译后 6/6 次运行 rc=0、[总体] PASS、口径 graph（无降级）；
> 两次运行各档差异 ≤1.1%，且均落在 perf.md v1 基线 ±1% 内 → 性能一致、无回归。
> rep 间波动（如 TPR=64 的 652.2 vs 658.4）为 202 机共享环境会话漂移。
