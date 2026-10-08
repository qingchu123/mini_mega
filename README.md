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
├── scripts/build.sh            # 产物输出到 test/（so / bench / mybench / check / whl）
├── scripts/run.sh              # 统一启动（mpirun，每进程一张卡；默认持 GPU 锁）
├── test/mybench.cu             # MPI 性能基准（-s 1 均衡 / -s 2 TK 实录）
├── test/check.cu               # MPI 正确性测试（--smoke / --all）
├── test/mega_common.cuh        # 两者共用的设备资源 / 布局 / 路由构造
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

产物统一输出到 `test/`（二进制与 `megakernel.so` 同目录，靠 `rpath=$ORIGIN` 定位）。

```bash
scripts/build.sh so        # test/megakernel.so     （接口层 + kernel）
scripts/build.sh mybench   # test/mybench           （每进程一张卡的 MPI 性能基准）
scripts/build.sh check     # test/check             （MPI 正确性测试）
scripts/build.sh bench     # test/bench             （单进程多卡 benchmark，旧口径）
scripts/build.sh whl       # torch/dist/*.whl       （需 python+torch，PYTHON= 可指定）
scripts/build.sh all       # = so bench mybench check（不含 whl）
```

不带目标时等价 `all`；`mybench`/`check`/`bench`/`whl` 会自动先编 `so`。
`whl` 不进 `all`：它依赖 torch 环境，本机未装 torch 时会必然失败。

依赖：MACA（`MACA_PATH=/opt/maca`）、`cucc`、`mcr` 运行时、系统头 `cute`
（位于 `/opt/maca/include`）。`so`/`mybench`/`check`/`bench` 无需 torch / 容器。

> 运行前必须 `export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:$LD_LIBRARY_PATH`：
> 系统 `ld.so.cache` 中含 `/opt/maca-3.8.4/lib`，否则会与 `/opt/maca` 混版，
> 报 `libmcpti.so: undefined symbol: TRACER_RECORD_NUM`。`scripts/build.sh`
> 已内置该导出；`scripts/run.sh` 亦已内置。

## 运行

`scripts/run.sh [程序] [-R N] [参数...]`，程序为 `mybench`（默认）| `check` | `bench`。
内部用 `mpirun -np R`（每进程一张卡），默认持 `.agents/work/lock/lock_gpu` 避免多 Agent 争抢。

```bash
# 性能（mybench）
scripts/run.sh mybench -R 32 -s 1 -tprs 1,4,16,32,33,64,128,256,512   # 均衡（默认场景）
scripts/run.sh mybench -R 32 -s 2 -top 50                              # TK 实录前 50 个
scripts/run.sh mybench -R 32 -s 2 -cases 804900,807500                 # 指定 TK launch

# 正确性（check）
scripts/run.sh check --smoke     # 冒烟，<10s
scripts/run.sh check --all       # 全量（均衡 TPR 表 + 全部 TK launch），不限时
```

实测（R=32，2026-10-08）：

| 命令 | wall |
|---|---|
| `build.sh so` / `mybench` / `check` / `all` | 35s / 38s / 39s / 48s（后三者含重编 so） |
| `run.sh mybench -R 32 -s 1 -tprs 1..512` | 7.2s |
| `run.sh mybench -R 32 -s 2 -cases 804900,807500` | 6.7s |
| `run.sh mybench -R 32 -s 2 -top 50` | 13.1s |
| `run.sh check --smoke` | 8.0 ~ 9.0s（3 次） |
| `run.sh check --all` | 约 4 分钟（762 用例） |

`check --all` 结果见 `test/77.correctness/测试流程.md`（L0/REDUCE/graph 全 PASS；
36 例 TK 用例 L1 未解决）。

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
launch）。实测单次全链：均衡 334.9µs → 不均衡 0.656/0.670/0.682ms（max 176/362/512，
v9 前序复用后；改前为 1.78/3.46/4.71ms），详见 `test/71.r32_tk_imbalance/results.md`。

## 基准（本机 202，C600UL 32 卡）

| TPR | 16 | 64 | 256 |
|---:|---:|---:|---:|
| full (µs/kernel) | 321.8 | 603.0 | 1800.7 |

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

## 真实不均衡路由长尾优化（窗口切分 + COMBINE 标记修复）

时间标签 `[261005_23_23]`（未提交 git）。根因：`win_cap` 缺省被抬到
`max_tokens=512`，收到 512 行的专家 `ceil(512/512)=1` 个窗口 → 整段串行落在
单个 block（长尾 4.7ms）。另外 `win_cap<max_tokens` 时各 rank 的 `merge/
nslice_eff` 可不一致，而 COMBINE 只写 1 个物理 slice 的完成标记，源 rank 的
REDUCE 按自身 `nslice_eff` 等待会漏槽 → 4M 级自旋超时（多节点 graph 累积为
秒级“伪死锁”）。改动：

1. `combine.cuh`：`combine_push_pair` / `combine_mark_rows` 把物理任务覆盖的
   **全部逻辑 slice**（`slice*merge + [0,merge)`）标记写满（`merge=1` 逐字节等价）。
2. `bench.cu` `run_aligned`：逐测例检测长尾不均衡（`max_expert ≥ 4×非零中位数`）
   时把 `win_cap` 收缩到 `round16(cap_need/kRing)`（约 kRing 个窗口）；均衡分布
   保持 `max(128,max_tokens)` 不变。

性能：目标 `--tk-launch 853000`（max512）**4708.8 → ~1295µs（−72.5%，<2000）**；
804300（max362）3479→1030、804400（max176）1784→1289；均衡 test/70 三档
345/646/1915µs（±3% 内）。正确性 R=2 与 R=32 全 `[总体] PASS`（见 `perf.md` v5）。

### v6：长尾再优化（merge 不粗化，不均衡专项）

在 v5 基础上给 `MegaArgs`/`MegaLaunchArgs` 增加 `merge_grid`：不均衡长尾测例传大值
→ `merge=1`（不做 `nslice_eff` 粗化），并把窗口目标调到 45 行；大专家 512 行切成
12 窗×4 片的小任务摊满 32 block。均衡测例保持原粗化（零回归）。
目标 853000 **1295 → ~1034µs（再降 20.2%）**；804400=967、804300=877。正确性
（含强制 merge=1）R=2/R=32 全 `[总体] PASS`（见 `perf.md` v6）。

## torch 层（pybind + Python 包，可安装 wheel）

时间标签 `[261006_12_40]`（未提交 git）。新增 `torch/`：pybind11 扩展
`mxmegakernel._C` + Python 包 `mxmegakernel`，对齐 sglang megamoe 后端
（C600UL fp8xfp8）所需的最小 API 面，并真实绑定 mini_mega kernel。

| 能力 | 状态 |
|---|---|
| `get_token_alignment_for_mega_moe` | 已实现 |
| `select_fp8_mega_moe_window`（FP8 AUTO 表） | 已实现 |
| `get_symm_buffer_size_for_mega_moe` / 布局偏移 | 已实现 |
| `transform_weights_for_mega_moe`（fp8xfp8） | 已实现（与源项目逐字节等价） |
| `SymmBuffer` 多 rank FG/IPC + 建档预热 + 输入视图 | 已实现 |
| `_C.mega_launch_empty` / `fill_weights_lane` | 已实现（真实 kernel 绑定） |
| `fp8_mega_moe` 融合发射（`fp8mega::launch_full`） | 已实现 |
| `bf16/fp8_fp4` 路线、`transform_sf_into_required_layout` | not_implemented |

构建：`cd torch && MACA_PATH=/opt/maca python setup.py bdist_wheel`
→ `torch/dist/mxmegakernel-0.0.0+mini.mega.20261006b-cp310-cp310-linux_x86_64.whl`。

数值自检：`tests/smoke.py`（单卡）→ `SMOKE OK`；
`torchrun --nproc_per_node=32 tests/test_fp8_mega_moe.py --ranks 32`（71，真实
R=32/E=256 几何，随机权重 vs 参考）→ 32/32 PASS；本地 R=16 window=64 亦全过。

71 跑通：GLM-5.2-FP8（tp32/dp32/megamoe/fp8）服务 ready（`/health=200`，
`max_total_num_tokens=148160`），`/generate` 输出连贯文本；并发压测
48 请求 / 并发 8：`output_throughput=114.5 tok/s`。需要三个 sglang 侧补丁
（权重原位接管、SwiGLU 缩放契约修复、预热），详见 `torch/README_torch.md`
与 `torch/sglang_patches/`。
