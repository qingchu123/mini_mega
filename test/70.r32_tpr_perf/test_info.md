# test/70.r32_tpr_perf — 测试涉及文件清单

- 日期：2026-10-05；机器：本机 202（C600-UL，32 卡）
- 测试内容：R=32 下 TPR=16/64/256 性能（口径见 `results.md`）
- 工程根：`zzzz/projs_deepseek/projs/01.fp8xfp8_moe_kernel/`
- git HEAD：`4382e5182fa2d926fb1daeddd4f4d4d80c951ef1`

## 1. 测试文件（本目录）

| 文件 | 作用 |
|---|---|
| `run.sh` | 测试驱动：3 档 ×2 次，逐次 `timeout -k 3 10` |
| `results.md` | 结果表与结论（含重编译一致性） |
| `test_info.md` | 本文件 |
| `t16_r1.log` / `t16_r2.log` | 重编译后 TPR=16 两次运行原始日志 |
| `t64_r1.log` / `t64_r2.log` | 重编译后 TPR=64 两次运行原始日志 |
| `t256_r1.log` / `t256_r2.log` | 重编译后 TPR=256 两次运行原始日志 |
| `run1/t*.log` | 重编译前首次运行的 6 个原始日志（对照用） |

### run.sh 实际执行命令

```
MACA_VISIBLE_DEVICES=0,1,...,31 timeout -k 3 10 stdbuf -oL ./bench \
  --ranks 32 --tprs 16,64,256 --nodes 20 --replays 2
```

## 2. 源文件（编译进 `./bench`）

`scripts/build.sh` 用 `cucc -O3 -std=c++17 -Isrc ... -o bench src/bench.cu` 单次编译，
include 展开后参与编译的源文件如下（md5 为本机当前版本）：

| 文件 | 角色 | md5 |
|---|---|---|
| `src/bench.cu` | host harness：参数解析、路由生成、显存分配/审计、正确性与数值校验、CUDA graph 链计时、TPR 扇形；`run_aligned` = 对齐 `mega/src/bench/bench_mega_fp8.cu`（单进程、均衡、graph nodes/replays、默认不校验） | `ef1003ff50159382f2abfa2c6454e8cc` |
| `src/common.cuh` | 场景常量（`kEPR=8,kTopk=8,kHidden=6144,kIntermediate=2048,kGrid=32,kNSlice=4,kRing=8,kM=16`）与工具 | `292d984029abc93b658d28e17e77fe4e` |
| `src/kernel.cuh` | 主 kernel `mega_kernel` 相位编排（前序→dispatch→L1→L2→COMBINE/REDUCE） | `718bdf4563481b055bc5e63a1a11f05d` |
| `src/prologue.cuh` | ROUTE/PUBLISH/XBAR/TASK_SETUP 前序相位 | `30711c075e1cc91432c595c1e41b54f8` |
| `src/compute.cuh` | 计算链（dispatch、小任务路径、任务循环） | `e7c95404f962e9609324fd92d267c757` |
| `src/compute_big.cuh` | 大任务路径（r24/A-smem、L1/L2 大 count） | `6d09e5bc9bdef39d6be7c4fe24bade0d` |
| `src/combine.cuh` | COMBINE / 融合推送 | `4d5c9d776d012f29e232a174ff52b946` |
| `src/flags.cuh` | 旗区协议（epoch/checksum/任务就绪） | `be4912cfba1a2518425eb2491127d6af` |
| `src/host_layout.cuh` | host/device 同源 scratch 布局偏移 | `3487d7ade401d4a90cd1fe94b514d4f6` |
| `src/bench_check.cuh` | 行级校验与 `w_lane_get` 权威权重公式 | `8fba1343308a05922dcde03fe68be17c` |

### include 依赖图

```
bench.cu
├── common.cuh
├── kernel.cuh
│   ├── common.cuh
│   ├── flags.cuh ── common.cuh
│   ├── prologue.cuh ── common.cuh, flags.cuh
│   ├── compute.cuh ── common.cuh, flags.cuh ── compute_big.cuh
│   │                                   ── common.cuh, flags.cuh, combine.cuh
│   └── combine.cuh ── common.cuh
├── host_layout.cuh ── common.cuh, <deep_gemm/layout/mega_moe.cuh>（外部）
└── bench_check.cuh ── common.cuh, host_layout.cuh
```

## 3. 构建与外部依赖

| 项 | 路径 / 值 |
|---|---|
| 构建脚本 | `scripts/build.sh`（`cucc -O3 -std=c++17`，产物 `./bench`） |
| 外部头 include 根 | `-I$REPO/mxmegakernel/include`，`REPO=/home/lchen1/data/projs/MXMOEMega` |
| 外部头 | `$REPO/mxmegakernel/include/deep_gemm/layout/mega_moe.cuh`（md5 `c0562ad7e9d80c35920166ecc8049371`） |
| 平台头 | `/opt/maca/include`、`$MACA_PATH/tools/cu-bridge/include` |
| 运行库 | `mcr/mc_runtime.h` + MACA 运行时（`MACA_PATH` 默认 `/opt/maca`） |

## 4. 被测产物与文档

| 文件 | 说明 | md5 |
|---|---|---|
| `bench` | 本次实际运行的二进制（`bash scripts/build.sh`，2026-10-05 改造后重编） | `c353addee74b9b49a4b7f4746c5fa79a` |
| `perf.md` | R=32 性能基准口径与 v0/v1 表（本次对照基线） | — |
| `docs/02_测试流程与性能标准.md` | 性能口径/正确性协议/运行规则权威定义 | — |
| `scripts/run.sh` | 一键正确性+性能（未用于本次纯性能测试） | — |

> 注：`src/bench.cu.orig`、`src/` 下历史 `bench_*` 二进制、`test/` 其他目录均非本次测试路径。
