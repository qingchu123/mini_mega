# mini_mega

fp8xfp8 MoE Mega Kernel 的**最小可复现子集**，从
`megakernel/zzzz/projs_deepseek/projs/01.fp8xfp8_moe_kernel` 抽取本次
R=32 TPR=16/64/256 性能测试（test/70）所涉及的全部文件。

## 内容

```
mini_mega/
├── src/
│   ├── megakernel.h            # host C++ 接口（公共头，bench 唯一依赖）
│   ├── megakernel.cu           # 接口实现 + kernel 发射 -> megakernel.so
│   ├── bench.cu                # benchmark：只调接口，不直接发射 kernel
│   └── *.cuh                   # kernel 实现（kernel/prologue/compute/combine/flags/...）
├── include/deep_gemm/          # 最小外部头（mega_moe.cuh + common/{math,exception}.cuh）
├── scripts/build.sh            # 先编 megakernel.so，再链接 bench
├── docs/02_测试流程与性能标准.md
├── perf.md                     # R=32 性能基准口径与基线
├── test/70.r32_tpr_perf/       # 均衡 TPR 基准
│   ├── run.sh  results.md  test_info.md
│   └── r{1,2}.log                # 2 次运行原始日志
├── test/71.r32_tk_imbalance/   # 真实 TK_INFO 不均衡路由基准
│   ├── run.sh  results.md  run.log
└── README.md
```

## 分层

- **接口层**：`src/megakernel.h`（声明 `fp8mega::MegaLaunchArgs` 与
  `launch_prelude/compute/reduce/full`、`launch_empty`、`fill_weights_lane`）。
- **实现层**：`src/megakernel.cu` 是**唯一**发射 `mega_kernel` /
  `mega_reduce_kernel` / `mega_empty_kernel` / `fill_w_lane_kernel` 的地方，
  编译为 `megakernel.so`。
- **调用层**：`src/bench.cu` 只 `#include "megakernel.h"` 并链接 `megakernel.so`，
  不包含 `kernel.cuh`、不出现任何 `<<<>>>`。

## 构建

```bash
setarch $(uname -m) -R bash scripts/build.sh   # -> ./megakernel.so + ./bench
```

依赖：MACA（`MACA_PATH=/opt/maca`）、`cucc`、`mcr` 运行时、系统头 `cute`
（位于 `/opt/maca/include`）。无需 torch / 容器。运行 `./bench` 时经 rpath
`$ORIGIN` 自动加载同目录的 `./megakernel.so`。

## 测试（R=32 TPR=16/64/256）

```bash
bash test/70.r32_tpr_perf/run.sh
```

口径（对齐 `mega/src/bench/bench_mega_fp8.cu`）：单进程 `run_aligned`，均衡路由，
graph 链 `nodes=20`、`replays=2`（取 min）、默认不做正确性校验：

```bash
MACA_VISIBLE_DEVICES=0,...,31 ./bench --ranks 32 --tprs 16,64,256 --nodes 20 --replays 2
```

`./bench` 不带参数 = 默认全档 R=32 TPR={1,4,16,32,33,64,128,256}（单次约 2.9s，
满足 10s 约束）。`--aligned 0` 可退回旧的单档/扇形口径。

## 不均衡测例（真实 TK_INFO）

读取内核抽样日志 `0928_tk_info.log`，复现某次 launch 的每专家收行分布并建图测量：

```bash
bash test/71.r32_tk_imbalance/run.sh
./bench --tk-info test/70.r32_tpr_perf/0928_tk_info.log --tk-launch 853000
```

日志行 `[TK_INFO] rank=r launch=n : c0..c7`；`ci` = rank r 物理专家 i 收到的 pair
数。`--tk-launch N` 指定 launch（缺省自动选「单专家最大收行」最大的全 rank
launch）。实测单次全链：均衡 354.6µs → 不均衡 1.81/3.46/4.72ms（max 176/362/512），
详见 `test/71.r32_tk_imbalance/results.md`。

## 基准（本机 202，C600UL 32 卡）

| TPR | 16 | 64 | 256 |
|---:|---:|---:|---:|
| full (µs/kernel) | 343.5 | 651.0 | 1918.3 |

详见 `test/70.r32_tpr_perf/results.md`。

## flag 结构体改造（local_flags / ranks_flags + current_time）

按参考实现 `mega/src/kernel/mega_fp8.cuh` 重构同步标志（2026-10-05，未提交 git）：

- 新增 `src/flags_defs.cuh`：
  - `struct alignas(256) local_flags { int64_t current_time[kGrid][64];
    uint32_t bar[1032]; int out_total[kGrid]; int out_processed[kGrid]; int dbg[8];
    uint32_t ck_dbg[kEPR*16]; unsigned long long pre_ticks[kGrid*16];
    float l1_chk[kGrid]; float l2_chk[kGrid]; }`（普通显存，本 rank 私有）；
  - `struct alignas(256) ranks_flags { uint32_t xbar_barrier[kGrid][kGrid];
    uint32_t combine_sums[kTopk][kMaxCombineRows][kCombineSlicesMax]; }`
    （FG 显存，跨 rank peer 可写；kMaxCombineRows=512、kCombineSlicesMax=32）。
- `MegaArgs` / `MegaLaunchArgs` 的 `bar_flags` 改 `local_flags* lf`，
  `epoch_ctr/last_release` 由每 block 单调 launch 计数 `lf->current_time[b][*]` 取代；
  xbar 跨 rank 旗从 sym 迁到 `rf->xbar_barrier`。
- 第二轮（本提交）：把控制/诊断缓冲 `out_total/out_processed/dbg/ck_dbg/
  pre_ticks/l1_chk/l2_chk` 从 `MegaArgs`/`MegaLaunchArgs` 独立指针/独立 mcMalloc
  收进 `local_flags`，kernel/host 读写一律走 `lf->...`；删除已无引用的
  `local_flags.last_ready`。跨 rank COMBINE 完成标记 `combine_sums` 从 sym 的
  `combine_sums_off` 区迁入 `ranks_flags::combine_sums[k][token][slice]`，
  COMBINE 推送/标记与 REDUCE 等待读取改经 `rf_peers[dst]->combine_sums`；
  `combine_rows`（BF16 数据）仍留在 sym，sym 分配随之去掉 sums 区。
- 行为不变：`bar[]` 索引逐一对齐原 `kBar*Idx`；可见性顺序（payload→fence→ready）
  与自旋/超时语义保持；`MX_DBG_COUNT=0` 默认零开销路径不变。

正确性（旧口径 R=2/32，`--perf-only 0 --compute 1`）全部 `[总体] PASS`；
性能（test/70，graph，2 次采样）：TPR=16/64/256 = 335~340/646~648/1909~1911
µs/kernel，相对基线 345~353/637~654/1901~1915 全在 ±3% 内（见 `perf.md` v3）。
test/71 不均衡 1788/3483/4749µs vs 基线 1776~1810/3458~3502/4686~4717，全在 3% 内。

## flag K 片轮转（M_CYCLE，对齐 mega_fp8.cuh）

时间标签 `[261005_22_06]`（未提交 git）。引入 `constexpr int M_CYCLE = 4`：
`local_flags::bar` 加 `[M_CYCLE]` 旋转维；`ranks_flags::xbar_barrier` 与
`combine_sums` 同理（后者 +1.5MB/rank，FG 显存）。每相位按单调 launch 计数
`current_time[b][*]` 派生 `c=ct%M_CYCLE`，只用第 c 片；内核入口对
`ct2=(c+2)%M_CYCLE` 槽 device 侧预清（graph replay 安全）。REDUCE 独立 kernel
由 `current_time[0][1]-1` 派生同一 c。可见性顺序与自旋/超时语义不变。
性能：test/70 TPR16/64/256 = 337.6~340.6/637.0~639.5/1904.1~1906.7µs（全在
基线 ±3% 内，TPR64 略快）；test/71 全档 1787/3448/4721µs（±3% 内）；R=2/32
全 `[总体] PASS`。
