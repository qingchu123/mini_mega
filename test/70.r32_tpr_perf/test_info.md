# test/70.r32_tpr_perf — 测试涉及文件清单

- 日期：2026-10-05；机器：本机 202（C600-UL，32 卡）
- 测试内容：R=32 下 TPR=16/64/256 性能（口径见 `results.md`）
- 工程根：`zzzz/projs_deepseek/projs/01.fp8xfp8_moe_kernel/`
- git HEAD：`4382e5182fa2d926fb1daeddd4f4d4d80c951ef1`

## 1. 测试文件（本目录）

| 文件 | 作用 |
|---|---|
| `run.sh` | 测试驱动：单次跑全档，共 2 次，每次 `timeout -k 3 10` |
| `results.md` | 结果表与结论 |
| `test_info.md` | 本文件 |
| `r1.log` / `r2.log` | 两次运行原始日志（含 R/TPR/us-kernel 表） |

### run.sh 实际执行命令

```
MACA_VISIBLE_DEVICES=0,1,...,31 timeout -k 3 10 stdbuf -oL ./bench \
  --ranks 32 --tprs 16,64,256 --nodes 20 --replays 2
```

## 2. 源文件（分层）

`scripts/build.sh` 分两步：(1) `src/megakernel.cu -fPIC -shared -o megakernel.so`；
(2) `src/bench.cu` 链接 `./megakernel.so`（rpath `$ORIGIN`）。md5 为本机当前版本：

### 2.1 接口层 / kernel 实现（编入 `megakernel.so`）

| 文件 | 角色 | md5 |
|---|---|---|
| `src/megakernel.h` | host 公共接口：`fp8mega::MegaLaunchArgs` 与 `launch_{prelude,compute,reduce,full}` / `launch_empty` / `fill_weights_lane`（bench 唯一依赖） | `9c8de46588901dca9acae4abb90e1af0` |
| `src/megakernel.cu` | 接口实现，**唯一**发射 `mega_kernel`/`mega_reduce_kernel`/`mega_empty_kernel`/`fill_w_lane_kernel` | `d3c9fa15f189845b6ae2f0531a90e886` |
| `src/common.cuh` | 场景常量（`kEPR=8,kTopk=8,kHidden=6144,kIntermediate=2048,kGrid=32,kNSlice=4,kRing=8,kM=16`）与工具 | `292d984029abc93b658d28e17e77fe4e` |
| `src/kernel.cuh` | 主 kernel `mega_kernel` 相位编排（前序→dispatch→L1→L2→COMBINE/REDUCE） | `718bdf4563481b055bc5e63a1a11f05d` |
| `src/prologue.cuh` | ROUTE/PUBLISH/XBAR/TASK_SETUP 前序相位 | `30711c075e1cc91432c595c1e41b54f8` |
| `src/compute.cuh` | 计算链（dispatch、小任务路径、任务循环） | `e7c95404f962e9609324fd92d267c757` |
| `src/compute_big.cuh` | 大任务路径（r24/A-smem、L1/L2 大 count） | `6d09e5bc9bdef39d6be7c4fe24bade0d` |
| `src/combine.cuh` | COMBINE / 融合推送 | `4d5c9d776d012f29e232a174ff52b946` |
| `src/flags.cuh` | 旗区协议（epoch/checksum/任务就绪） | `be4912cfba1a2518425eb2491127d6af` |
| `src/host_layout.cuh` | host/device 同源 scratch 布局偏移 | `3487d7ade401d4a90cd1fe94b514d4f6` |

### 2.2 调用层（编入 `bench`）

| 文件 | 角色 | md5 |
|---|---|---|
| `src/bench.cu` | host harness：参数解析、路由生成、显存分配/审计、正确性与数值校验、CUDA graph 链计时、TPR 扇形；`run_aligned` = 对齐 `mega/src/bench/bench_mega_fp8.cu`；**只 include `megakernel.h`，无任何 `<<<>>>`** | `87d2cb22af808afd9bee458335d8d0d2` |
| `src/bench_check.cuh` | 行级校验与 `w_lane_get` 权威权重公式 | `8fba1343308a05922dcde03fe68be17c` |

### include 依赖图

```
bench.cu
├── common.cuh
├── host_layout.cuh ── common.cuh, <deep_gemm/layout/mega_moe.cuh>（外部）
├── bench_check.cuh ── common.cuh, host_layout.cuh
└── megakernel.h（接口；链接 megakernel.so）

megakernel.cu（-> megakernel.so）
├── megakernel.h
├── common.cuh
└── kernel.cuh
    ├── common.cuh
    ├── flags.cuh ── common.cuh
    ├── prologue.cuh ── common.cuh, flags.cuh
    ├── compute.cuh ── common.cuh, flags.cuh ── compute_big.cuh
    │                                   ── common.cuh, flags.cuh, combine.cuh
    └── combine.cuh ── common.cuh
```

## 3. 构建与外部依赖

| 项 | 路径 / 值 |
|---|---|
| 构建脚本 | `scripts/build.sh`（`cucc -O3 -std=c++17`；产物 `./megakernel.so` + `./bench`，bench 链接 .so） |
| 外部头 include 根 | `-I$REPO/mxmegakernel/include`，`REPO=/home/lchen1/data/projs/MXMOEMega` |
| 外部头 | `$REPO/mxmegakernel/include/deep_gemm/layout/mega_moe.cuh`（md5 `c0562ad7e9d80c35920166ecc8049371`） |
| 平台头 | `/opt/maca/include`、`$MACA_PATH/tools/cu-bridge/include` |
| 运行库 | `mcr/mc_runtime.h` + MACA 运行时（`MACA_PATH` 默认 `/opt/maca`） |

## 4. 被测产物与文档

| 文件 | 说明 | md5 |
|---|---|---|
| `megakernel.so` | host 接口层 + kernel 发射（bench 经 rpath `$ORIGIN` 加载） | `bfc00c0d6a499bdeccb30e04974cd805` |
| `bench` | 本次实际运行的二进制（`bash scripts/build.sh`，2026-10-05 接口化后重编） | `9ab44acbaea95f32ca655e1e6ca7473e` |
| `perf.md` | R=32 性能基准口径与 v0/v1 表（本次对照基线） | — |
| `docs/02_测试流程与性能标准.md` | 性能口径/正确性协议/运行规则权威定义 | — |
| `scripts/run.sh` | 一键正确性+性能（未用于本次纯性能测试） | — |

> 注：`src/bench.cu.orig`、`src/` 下历史 `bench_*` 二进制、`test/` 其他目录均非本次测试路径。
