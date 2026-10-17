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

## 测试环境

> 本节记录 `perf.md` / README 中性能数据的采集环境；换机复现时请逐项核对。
> 最近核对：2026-10-17。

### 1. 内核基准（`mybench` / `check`，宿主机直跑）

| 项 | 配置 |
|---|---|
| 采集机 | `10.0.82.202`（README 中"本机 202"），单机 32 卡 |
| GPU | 32 × MetaX C600-U（MXC600-U），**28 SM/卡** |
| 驱动 | `3.101.0-20261011-153`（MX-SMI 3.101.0，BIOS 1.2.7.0） |
| MACA SDK | `20261011`（`/opt/maca` → `~/opt/maca-sdk-c600u-20261011`；`mxcc`/`cucc` 1.0.0） |
| OS / kernel | Ubuntu 22.04.5 LTS / `5.15.0-119-generic` |
| Python / PyTorch | conda `py310-maca`（`/home/fuyx/miniconda3/envs/py310-maca`），PyTorch 2.10（MetaX 构建） |
| EP 后端 | `deep_ep` / `mxmesh` `1.0.0+metax20260909.303torch2.10`（默认 wheel 目录 `/home/fuyx/maca-pkg/wheels/20260909-303/wheel`；`--ep-backend=auto` 走 mxmesh） |
| IB / 网络 | `mlx5_10..13`（`tests/mx_test/test_env.sh` 默认 `MXSHMEM_HCA_LIST` / `MCCL_IB_HCA`），`MCCL_SOCKET_IFNAME=enP1s136np0` |
| 构建开关 | `KERNEL_CFLAGS=--offload-arch=xcore1600`；默认 `MX_GROUP_SCHED=1`（自动 `MX_GROUP_WEIGHTS=1`，**GS 默认开**；`-DMX_GROUP_SCHED=0` 关回基线） |
| 运行口径 | `-R 32`（每进程一张卡）、`fp8xfp8`、E=256 / topk=8 / H=6144 / I=2048、50-in-1 CUDA Graph；MXMOEMega 对比口径 MTP 384（mega）/ 256（baseline） |

### 2. 容器 / 服务侧（sglang 模型测试）

| 项 | 配置 |
|---|---|
| 镜像 | `harbor.sh.mxcr.io/ai-release/maca/sglang:0.5.14-maca.ai3.8.2.109-c600u-torch2.10-py310-ubuntu22.04-amd64` |
| 采集机 | 同系列 C600UL 单机 32 卡（`10.0.104.82`） |
| 容器内 MACA | 3.8.2.19（驱动 `3.101.0-20261011-153`） |
| 模型 | GLM-5.2-FP8（E=256 / topk=8 / H=6144 / I=2048，78 层 MoE） |
| 服务口径 | `--mem-fraction-static=0.85`、MTP on、CUDA graph on；profile 流程为 PD-decode + fake-prefill |
| GS | 容器脚本显式 `MXK_GS_GROUP_GATHER=0`（组权重聚合关闭 → **GS 运行期不生效**；内核仍按默认 `MX_GROUP_SCHED=1` 编译） |

**sglang 测试注意事项（[261017] 实测沉淀）：**

1. **CUDA graph 必须开**：`--disable-cuda-graph` 是 kernel trap 排查用的调试开关，不是性能口径。禁图会让每步几千个 kernel 单独发射，MACA 栈上 TPOT 被放大 ~8–9×（实测 deepep 715→77ms、mega 656→80ms）。vendor 性能脚本默认注释该开关；若历史脚本里它是激活状态，先复位再测。
2. **开图 + MTP 时 `--mem-fraction-static ≤ 0.85`**：0.89 下 deepep 会因 DeepEP buffer + 图私有内存在图捕获阶段 OOM（`Capture cuda graph failed: 512MiB... 0 bytes free`），大 prefill 的 deep_gemm masked GEMM 也可能 OOM（768MiB）。0.85 下两边 KV 池约 119k，本负载（16K × 99% 共享前缀、并发 120）刚好够用。
3. **GS 在容器里是"编译开、运行关"**：容器脚本显式 `MXK_GS_GROUP_GATHER=0` → 不做组权重 all-gather → `group_weights_ready=0` → 内核 `gs_on = 倾斜门控 && ready` 恒为假，走 base 路径。要在容器验证 GS，需 `MXK_GS_GROUP_GATHER=1` 并预留组权重显存（4× 权重副本），且只在倾斜路由下才有效。
4. **MTP 口径要固定**：MTP on/off 不影响精度，但 TPOT = 每步耗时 ÷ accept_len；跨后端比较必须同 MTP 配置，或改比"每步耗时/吞吐"。内核对照建议 MTP off，服务口径再单独开 MTP。
5. **profiler 必须卡在稳态**：`POST /start_profile` → 5s → `POST /stop_profile`；要在**第 2 轮**（第 1 轮 warmup）且**确认最近有 decode 批次**后再触发——5s 窗口可能落在 round2 起始的准入空窗（实测遇到：整分钟 0 条 decode 日志，trace 里既无 step span 也无模型 kernel）。trace 落在 `SGLANG_TORCH_PROFILER_DIR`，每 rank 一份 `.trace.json.gz` + `.csv`。
6. **确认是否真的走了图**：看 decode 日志里的 `cuda graph: True`；trace 里应有 `full_cuda_graph_backend.replay`（每步约 3 次：draft / target-verify / draft-extend）。
7. **PD-decode + fake-prefill 口径**：`--disaggregation-mode=decode --disaggregation-transfer-backend=fake` + 客户端 `--pd-separated --fake-prefill` 是**纯 decode** 测试，TTFT 无意义（只是准入），只看 TPOT/ITL/吞吐；16K 输入是 99% 共享前缀，不代表真实 16K 独有上下文。
8. **NUM_MAX 与 chunk 的关系**：`SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` 必须 ≥ chunked-prefill/dp（chunk 8192、dp32 → 256）；该值越大对称缓冲越大，2048 建不出来（`mcMalloc` OOM），decode 实际只需 ~16–20。
9. **megamoe 在容器里需要 [FOLD]/[RAW-X] shim 才能出正确结果**：容器自带 sglang 不做 checkpoint block-scale 折叠，JIT pre_dispatch 会做 UE8M0 归一化；未修时输出乱码（logprob −12~−16）。用本仓库 kernel 接 sglang 前先确认 shim 版本与 fold 开关（E/F=8）。
10. **容器可能被外部停止**：长跑前后检查 `docker ps`；被停后 `docker start` 再重启服务（本环境实测约每小时可能被停一次）。

**sglang 测试文件与流程（宿主机 `~/`，容器内同路径挂载）：**

| 类别 | 文件 |
|---|---|
| 服务启动（本机适配） | `serve_megamoe_local.sh`（A/B 基线）、`serve_megamoe_graph_on.sh`（开图）、`serve_megamoe_graphon_mtp.sh`（开图+MTP）、`serve_megamoe_acc_local.sh`（acc 口径）、`serve_megamoe_pddecode_261017.sh`（PD-decode + profiler 流程）；deepep 侧 `serve_deepep_local.sh`、`serve_deepep_local_c8192.sh`、`serve_deepep_graph_on.sh`、`serve_deepep_graphon_mtp.sh`、`serve_deepep_pddecode_261017.sh` |
| 官方（vendor）脚本 | `decode_fake_test_megamoe_0927.sh`（mega 性能/PD-decode）、`decode_fake_test_megamoe_acc_0927.sh`（mega 精度）、`decode_fake_test_unified_acc.sh`（deepep 精度）、`bench_serving.sh` + `client_loop_test_decode.sh`（PD-decode 两轮客户端）、`profiler.sh`（5s 抓拍）、`run_gsm8k.sh`、`client.sh`（位于 `~/sw_home/m01182/test/c600ul/megamoe/`） |
| 性能客户端（本机） | `bench_mega_256x512.sh` / `bench_deepep_256x512.sh`（端到端 256 prompts / 512 out / conc128 / gsp 1013+10）、`bench_lowconc.sh`（conc16） |
| 精度 | `run_gsm8k.sh`（200 题 / 5-shot / completion / temp 0）、`client.sh`（单语料连贯性检查） |
| 修复/兼容 | `mxmegashim.py`（site-packages；[FOLD]+[RAW-X]；旧版备份 `mxmegashim.py.bak_261016_orig`） |
| 报告/证据 | `logs/serving_ab_summary_261016.md`、`logs/perf_acc_summary_final_261017.md`；profiler 目录 `logs/profiler/{megamoe_pddecode2,deepep_pddecode}`；服务/客户端日志 `logs/*.log` |

**标准流程：**

1. 启动服务：`NODE_IP=<ip> MEGA_MOE_MEM_FRACTION=0.85 bash serve_xxx.sh fp8xfp8`（容器内）；等 `/health` 200（boot 含权重加载/图捕获，约 5–8 分钟）。
2. 探针：`POST /generate` 短文，确认输出连贯、MTP accept_rate 正常。
3. 性能（端到端 A/B）：`bash bench_mega_256x512.sh` / `bash bench_deepep_256x512.sh`；记录 Output tok/s、Mean TPOT、TTFT P50/P90、并发；两轮取稳态（第 1 轮冷缓存）。
4. 性能（低并发）：`bash bench_lowconc.sh`（conc16，验证 20–30ms 口径）。
5. 官方 PD-decode + profiler：`bash bench_serving.sh decode <ip> 8400`（两轮 512→1024 prompts、bs128、16K/1K、fake-prefill）；第 2 轮确认最近 10s 有 decode 日志后，`POST /start_profile` → sleep 5 → `POST /stop_profile`；trace 落 `SGLANG_TORCH_PROFILER_DIR`（每 rank 一份 trace + csv）。
6. 精度：`bash run_gsm8k.sh <ip> 8400`（200 题；1 题 = 0.5%，0.955–0.970 视为噪声带）；单语料 `bash client.sh <ip> 8400`。
7. 结果归档：客户端日志 + 服务日志（含 `cuda graph: True/False`、accept len/rate）+ trace，写入 `logs/serving_ab_summary_261016.md` 与 `logs/perf_acc_summary_final_261017.md`。

**闸门（每轮必查）：**

- 两侧启动脚本 diff 只允许后端本质差异（`moe-a2a-backend` / `deepep-mode` / `DEEPEP_NUM_MAX` / mega 开关）＋ profiler 输出目录；
- 服务日志 `cuda graph: True`（或 trace 内 `full_cuda_graph_backend.replay`）确认走图；
- `grep -iE "out of memory|exception|crashed|assert"` 与 `find ~/mxlog -name 'trapInfo*'` 确认零 OOM / 零 trap；
- 记录 warmup/稳态轮次、KV 池、accept_len、并发，避免跨口径比较。

### 3. 测试文件与流程

**主要测试文件**

| 文件/目录 | 用途 |
|---|---|
| `test/mybench.cu` | MPI 性能基准：`-s 1` 均衡 TPR；`-s 2` TK 实录（`-top N` / `-cases ...`） |
| `test/check.cu` | MPI 正确性测试：`--smoke`（10 例，<10s）/ `--all`（762 例，约 4 分钟） |
| `test/mega_common.cuh` | mybench/check 共用的设备资源、布局、路由构造 |
| `test/70.r32_tpr_perf/` | 均衡 TPR 基准（`run.sh`、`results.md`、`r1/r2.log`） |
| `test/71.r32_tk_imbalance/` | 真实 TK_INFO 倾斜基准（`run.sh`、`results.md`、`run.log`） |
| `test/77.correctness/测试流程.md` | 正确性判据与流程（L0/L1/L2/REDUCE/graph 50-in-1） |
| `test/73/74/75/76/78/79/80/82/91/92/93/94.*` | 专项 / 微基准（各含 `run.sh` + `results.md`） |
| `scripts/build.sh` / `scripts/run.sh` | 构建 / 统一启动（mpirun、每进程一卡、GPU 锁） |
| `docs/02_测试流程与性能标准.md` | 性能口径与正确性协议的权威说明 |
| `perf.md` | 性能记录（每行标注提交时间/号） |

**标准流程**

1. 构建：`scripts/build.sh all`（或 `so` / `mybench` / `check`）；增量构建，产物在 `test/`。
2. 冒烟正确性：`scripts/run.sh check --smoke`（10 用例，实测 7.5s，目标 <10s）。
3. 全量正确性：`scripts/run.sh check --all`（762 用例，约 4 分钟；已知 36 例 TK 用例 L1 FAIL——同一 launch 单跑 PASS，疑似跨用例 pool/权重槽残留）。
4. 均衡性能：`scripts/run.sh mybench -R 32 -s 1 -tprs 1,2,...,256`（实测 7.2s）。
5. 实录倾斜性能：`scripts/run.sh mybench -R 32 -s 2 -top 50`（13.1s）或 `-cases 804900,807500`。
6. 65-case 对比（MXMOEMega 口径）：`run_mega_moe_all.sh --num-processes 32 --routing-fixed-cases all --ep-backend=auto`；`OUT_DIR` 产出 `profile.log` / `run_metadata.txt` / `summary.tsv`（+ `stages.tsv`）；`speedup = baseline / mega`。
7. 结果归档：`perf.md` 追加一行（标注提交时间/号），结论同步 README / `test/*/results.md`。

**口径与判据要点**

- 正式性能口径：单张 CUDA Graph 内 N 个同构节点、host 1 次提交/replay、独立 event 对、预热 2 次、5 次 replay 取 **min-of-5**、`per-kernel = replay/nodes`；`rank_barrier` 在 event 窗口外。不要用 host wall 或单次 event。
- 跨会话数值不直接对比；A/B 优化必须同会话。
- 分相计时（`phase_ticks`/`clock64`，按 2000MHz 折算）只作诊断，正式性能只认 graph 链。
- 正确性判据：L0 全 rank 全字段相等；L1 rel 7e-2 / abs 2e-3；L2 rel 1e-2 / abs 1e-3；REDUCE rel 1e-5 / abs 1e-6；graph 50-in-1 vs eager rel ≤1e-3。
- 路由三形态：balanced / hot / mixed-invalid；改可见性/校验协议必须过 `--stress`，checksum on 对照须 0 失败（历史 checksum off 出现 290/300、273/300 失败）。
- 已知坑：`bench.cu` 曾构造重复专家 topk（非法）；`win < max_tokens` 在 max_tokens=512 时丢 topk 贡献（已统一 `win = max_tokens`）。

### 4. 换机注意事项

- **IB 设备名/网卡名随机器不同**（例如 `10.0.104.82` 为 `mlx5_0..6`），需覆盖 `MXSHMEM_HCA_LIST` / `MCCL_IB_HCA` / `MCCL_SOCKET_IFNAME`；
- **Python 路径**默认 `/home/fuyx/miniconda3/envs/py310-maca/bin/python`，不存在时用 `PYTHON=` 指定；
- **SDK 版本**：`/opt/maca` 必须指向与驱动匹配的 SDK，避免 `libmcpti.so` 混版；当前宿主机已更新到 `20261016`（`mx-smi` 报 MACA `3.0.2098.20261016`），复现历史数据时注意与采集时的 `20261011` 对齐；
- **SM 数**：非 C600-U 的卡用 `MXK_KERNEL_GRID=<SM数>` 覆盖；
- **GS 口径**：本仓库默认构建 GS 开（仅倾斜路由、门控通过时生效）；若要与"base 路径"对照，用 `KERNEL_CFLAGS="-DMX_GROUP_SCHED=0"` 重编，并在结果里注明。

## 关键编译期参数（`src/common.cuh`）

| 参数 | 值 | 说明 |
|---|---|---|
| `KNSLICE_OVERRIDE`（`kNSlice`） | 4 | 每窗口任务切片数；NS=4 时 L1 每片 4 组、L2 每片 6 组。NS=8 已验证可用（`g_l2=3` 需走 `compute_big.cuh` 的 `[R35]` 单批 3 组分支）；NS=16 暂不可用（L2 N 组几何不整除，待设计） |
| `kRing` | 8 | 同专家并发窗上限 |
| `kEPR` / `kTopk` / `kHidden` / `kIntermediate` | 8 / 8 / 6144 / 2048 | 场景维度 |
| `kGrid` | 32 | grid 块数（= SM 数） |

切片/L1 进度旗区容量由 `include/deep_gemm/layout/mega_moe.cuh` 定义
（`kBarMaxNSlice=16`，两旗区各 1024 项，承载 EPR×ring×NSLICE≤1024）。
NS=8 下 `r19_small_task` 的 L2 `g_l2=3` 走单批 3 组路径（`[R35]`，见
`compute_big.cuh`），不能走 `[O3]` 的 3+3 两批。

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

构建为**增量**：`so` 依据各 TU 的 `-MMD` 依赖按 mtime 跳过未变动对象
（改动只影响部分 TU 时仅重编该部分），`mybench`/`check` 仅在自身源码或
`megakernel.so` 变化时重链。无改动重跑 `all` <0.1s；`KERNEL_CFLAGS` 或
包含路径变化时自动全量重编（指纹存于 `build/obj_so/.buildflags`）。

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
scripts/run.sh mybench -R 32 -s 1 -tprs 1,2,4,6,8,12,18,24,30,36,42,48,54,60,66,72,78,84,90,128,256   # 均衡（默认场景）
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
| `run.sh check --smoke` | 7.5s（10 例：均衡 + 实录 case0/case2） |
| `run.sh check --all` | 约 4 分钟（762 用例） |

`check --all` 结果见 `test/77.correctness/测试流程.md`（L0/REDUCE/graph 全 PASS；
36 例 TK 用例 L1 未解决）。

## 冒烟新增实录 case2（`809200`，对齐 MXMOEMega `--routing-fixed-cases 2`）

时间标签 `[261013_12_32]`。`--smoke` 原覆盖实录 case0（`804900`）与均衡窗；本次把
MXMOEMega 固定路由语料的 **case 2 = launch `809200`**（32 卡 × 16 token × topk8 = 4096
pair，实录 skew 14.56x / 40 个零专家 / 单专家 max 233 / 每 rank 收行 17..450）也纳入
冒烟，逐例跑前序/eager/graph×1 与 graph×50（`test/check.cu`，`fixed_case_spec`）。

- 复现命令（无 python）：`scripts/run.sh check --smoke`；用例数 8→10，仍 **<10s**。
- 正确性：**10/10 PASS**（7.51s，R=32）；case2 的 L0/L1L2/REDUCE 全 PASS，
  `graph×1`/`graph×50` 的 `rel=0.00e+00`（旧 wheel 曾在此例报
  `50-in-1 graph diverges rel_max_diff=1.021e+00`，当前 C++ 内核已不复现）。

## torch (`mxmegakernel`) gs 组权重 all-gather 修复（`[261013_13_45]`）

时间标签 `[261013_13_45]`。修复 torch 层 `--routing-fixed-cases 2` 的
`50-in-1 graph diverges rel_max_diff=1.021e+00`。

- **根因**：`torch/csrc/apis/mega.hpp` 未 include `common.cuh`，`MX_GROUP_WEIGHTS` 在
  该 TU 内未定义（当作 0），`ensure_mega_weights` 走了 `#else` 非组分支——**所有 rank
  都写到 `w1g/w2g` 第 0 块**并忽略 `member` 块，且只拷本地 1 块；gs 远端 helper /
  `member>0` 的 owner 因此读到错/零权重，结果与调度绑定，graph 与 eager 发散。
- **修复**：`flags_bridge.cu` 暴露 `mxk_group_ranks()`；`mega.hpp` 的
  `ensure_mega_weights` 增加"组块数"参数、按 `w1g[0..kGE)` 连续写满；`python_api.cpp`
  权重 `numel` 校验放宽为每 rank 权重字节数的整数倍并透传块数；`mega/__init__.py`
  新增 `SymmBuffer.group_weights()`——按 gs 组（`rank%ng` 的 4 个成员、member 序）
  `all_gather` 打包权重成 `(kGE,...)`，capture 前建缓存 / capture 内复用，
  `fp8_mega_moe` 与 `fp8_mega_moe_compute` 调用。
- **验证**：MXMOEMega `--skip-baseline --routing-fixed-cases 0..7`（graph×50）
  `compare-graph-check` **全部 `rel_max_diff=0.000e+00`**；`check --all` 762/762、
  `check --smoke` PASS。
- **回归探针**：`MT_GS_MEMBER_ONLY=1 scripts/run.sh check --smoke`（~5s）会复现同类
  发散（check 用"仅本 rank 块"权重模拟缺 all-gather）。

## gs-on 模型 ATU 修复（`[261014_06_00]`，commit `b82bbd5`）

GLM-5.2 模型上 `MXK_GS_GROUP_GATHER=1`（gs 默认开）持续报
`Xnack Error/ATU Fault(0x8)`（`mega_kernel<true,MegaPhase::0>`，集中在同一 gs 组
rank 5/13/21/29），bench 无法跑完。

- **根因**：`[MX_GROUP_SCHED]` 下 kernel 动态 smem 布局为
  `s_count[8]+s_cum[9]+s_prefix[kGE*(R+1)]+s_gscount[kGE]+s_ge[512]+s_gw[512]`
  = **2129 int = 8516B**（R=32、kGroupRanks=4），而 torch 发射侧
  `torch/csrc/apis/layout.hpp::prod_smem`（`mega.hpp:259` 按 `window=8` 调用）只给
  **8448B** → `s_prefix/s_gscount/s_ge/s_gw` **溢出到越界 smem** → 读到垃圾 `ge`/owner
  → `unk_peers[owner]` 远端越界 → ATU。**bench/test 走 `src/host_layout.cuh` 的 40KB
  分支故不复现**（本地单进程共享内存，物理上也无法复现 IPC/ATU）。
- **修复**：`layout.hpp::prod_smem` 与 `host_layout.cuh` 一致，取 `max(原公式, 40960)`；
  另加 `[GS-CAP]` 钳位（`gs_schedule_member` 只限写入不限返回，`s_nwin>kGsMemberCap`
  时回退 base 路径）；`MX_GS_SKEW_NUM` 默认 3→0（恒启用 gs）。
- **验证（模型 10.0.82.71，GLM-5.2-FP8，EP=32，MTP=512，win48）**：
  `gs-on` bench **跑完、ATU=0**，Mean TPOT **26.78ms**、Output **4525 tok/s**
  （优于 base 30.62ms / 4002 tok/s）；`check --all` **762/762**。
- **排查过程**：否定跨 rank desync（32 rank round 全同步）；否定本地复现；
  `kMaxTPR 512→768`（精确==模型 `slot_cap`）使 ATU 从 round ~16000 推迟到 ~141000
  （量化了"越界随轮数漂移"）；`MX_GROUP_RANKS=1`（无跨 rank 远端读）→ ATU=0，把问题
  锁定到 gs 跨 rank 路径。

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

## 块级代价感知 LPT（`-s2` 不均衡专项，默认关）

新增编译开关 `R19_BLOCK_LPT`（默认 0，`-DR19_BLOCK_LPT=1` 开启）。动机：`-s2`
每 rank 内各 block 的 L1+L2 GEMM 代价（∝ 窗口行数）差异大，而现有 FCFS
（`atomicAdd(task_counter)`）只均衡任务「个数」。开启后 `dispatch_pull_body` 在
tid0 建表阶段对 `raw_tasks = s_cum[kEPR]*kNSlice` 个任务按代价
`cost=min(c_e-w*win_cap, win_cap)` 做**降序稳定计数排序**（smem `s_order`），
领号仍是原 `atomicAdd`，但 `task = s_order[counter]`（降序发放 + 动态领取 ≈ LPT）。
每任务解码 `(e,w,slice)`、`pool_slot`、ring 门、`l1_prog`/`l2_done` 协议完全不变
（`src/compute.cuh:679-726`, `:772-788`）。

- 正确性：开关开/关均 `check --all` **762/762 PASS**（含 win=32..128 自适应窗）。
- 性能（R=32, `-s1 tprs 32/128/256` 与 `-s2 top50 replays=3`，两次均值）：
  | 指标 | OFF(基线) | ON | 变化 |
  |---|---:|---:|---:|
  | s1 TPR32 | 488.8 | 496.4 | −1.6% |
  | s1 TPR128 | 944.6 | 964.6 | −2.1% |
  | s1 TPR256 | 1673.4 | 1692.9 | −1.2% |
  | s2 top50 mean | 765.2 | 766.5 | ±0.2%（中性） |
- 结论：**未达 ~400µs 目标**。根因是**任务量化**：s2 每 rank 重任务切片数常为
  32 的非整数倍（如 rank28 有 44 个满窗切片 / 32 block → 鸽笼必有 12 block 做
  2 个满窗），LPT 只能压低较空 block、无法降低最忙 block 的上界，故墙时不变。
  最忙 block 的 L1 max/min 由基线 2.51× 变为 2.71×（未收敛）。真正要降需**更细
  粒度派发**（每窗按 `kNK` 切 K 段，`raw_tasks = s_cum*kNSlice*kNK`）后再 LPT。

## `-s2` 细粒度派发实验（NS=8 与跨 block K 切，均默认关、零行为变更）

时间标签 `[261012_03_04]`。承接上节：直接尝试两种「更细粒度」派发，**结论均为
净回退**（未达 ~400µs），开关默认关、关时逐字等价基线。改动仅
`src/compute.cuh`、`src/compute_big.cuh`。

### Stage 1：N 切 NS=8（`-DKNSLICE_OVERRIDE=8` ± `-DR19_BLOCK_LPT=1`）

- 原理：把每窗任务数由 4 增到 8（N 组更细）。每片 A 重读 ×N、权重不重读、无需归约。
- 实测（R=32，`-s1 tprs 32/128/256 -nodes 3 -replays 2`；`-s2 top50 -replays 3`）：

  | 指标 | 基线 NS=4 | NS=8 | NS=8+LPT |
  |---|---:|---:|---:|
  | s1 TPR32 | 485.4 | 743.6 | 762.9 |
  | s1 TPR128 | 946.1 | 1208.8 | 1225.0 |
  | s1 TPR256 | 1678.6 | 2239.6 | 2277.8 |
  | s2 top50 mean | 767.9 | 1086.8 | — |

- **否决**：s1 回退 +28~53%、s2 回退 +42%。根因：N 切把 A 装载次数放大 NS 倍，
  且每窗协议任务数翻倍；「更细」的收益被访存放大吞掉。

### Stage 2：跨 block K 切（`-DR19_BLOCK_KSPLIT=1`，`R19_BLOCK_KSPLIT_WINMAX=64`）

- 实现（`src/compute.cuh` + `src/compute_big.cuh`）：把每个 `(e,w,slice)` 的 L1 沿
  K=H 再切 `kNK` 段、跨 block 派发（`raw_tasks = s_cum[kEPR]*kNSlice*kNK`，
  仅在 `win_cap ≤ 64`（TK/长尾小窗）+ fragment 可用时启用；大窗走原路径）。
  - **owner（kseg==0）**执行 pull/权重 staging/分片旗、等 `l1_prog`、跑 L2/tail、
    发布 `l2_done`；其余段只算 K 部分和 —— 严禁每段跑协议。
  - 各段把 gate/up 原值以 **FP32 atomicAdd** 累加到 `l1_partial[slot][slice]`；
    owner 在置分片旗前把该部分和缓冲与 `l1_part_cnt[slot][slice]` 清零（每代一次）。
  - `l1_part_cnt` 的 atomicAdd 判「最后一段」（`old==kNK-1`）→ `r19_l1_reduce`
    （SwiGLU→FP8→ffn_hidden/frag2）→ 由该段发布 `l1_prog`。
  - 新增函数：`r19_l1_ksplit_iso<kMT>`（`compute_big.cuh:556-625` 区）、
    `r19_l1_wait_all`（owner 只等不发布）；`r19_pull_flags` 增加 `kseg` 参数
    （非 owner 跳过 pull、只等分片旗）；任务体 `r19_big/small_task` 增加 `kseg` 并
    按 owner 决定 L2/tail。
- 正确性：开/关均 `check --all` **762/762 PASS**；`check --smoke` **8/8 PASS**
  （K 切开启 <10s）。
- 性能（R=32，同口径）：

  | `kNK` | s2 top50 mean | 相对基线 |
  |---:|---:|---:|
  | 1（基线） | 767.9 | — |
  | 2 | 1003.6 | +31% |
  | 4 | 1419.1 | +85% |
  | 8 | 1970.2 | +157% |

  s1（`kNK=8`）：TPR32 = 1246.8（+157%）、TPR128 = 979.1（win=128 未启用 K 切，
  ≈基线）、TPR256 = 1690.5（≈基线）。
- **否决**：K 切随段数单调变差。根因（实测 + 分析）：
  1. **每窗任务数 = NS×NK 把窗口「独占」整张 grid**（32 任务/窗），窗口由基线
     「8 窗并行」退化为「1 窗串行」；窗内 pull→各段→reduce→L2 的串行段无法被
     其它窗的延迟掩盖。`kNK` 越小任务/窗越少、并行窗越多，回退越小 —— 但始终为负。
  2. **FP32 部分和 atomicAdd 流量**：每 `(e,w,slice)` 写 `count×kL1SliceCols×4B`
     ×`kNK` 次（另加清零/读回）。小窗（`count≤48`）下这是不可忽略的 HBM 原子流量。
  3. 块级 L1 max/min：804900 rank28 由 K 切的 2.22×（min 212k / max 472k）
     **未收敛到目标 ≲1.2×**：段代价仍 ∝ 窗口行数，且 FCFS 按窗连续发牌。
- 结论：**未接近 400µs**；两阶段均默认关（`R19_BLOCK_KSPLIT=0` 编译期消除，
  关时 `check --all` 762/762、s1/s2 与基线逐字一致）。

## 方向1：B（权重）smem 异步双缓冲（`-DR24_BSMEM_B=1`，默认关）

时间标签 `[261012_09_40]`。动机：消融 `R24_DIAG_NOB` 显示 B 装载裸露 ≈211µs
（占 `-s2 top50` 的 28%）。实现：`r24_l1_big`/`r24_l2_big` 增加编译期 `kBsmB`
环深；B 经 `__builtin_mxc_ldg_b128_bsm` 搬入 smem 环、`r24_lds_b` 读，与 A 的 bsm
流水对齐。**关键坑**：该 builtin 的拷贝计入 **gvmcnt**（非 bsmcnt！）——`arrive(4096)`
等不到 B，必须 `arrive(64+N)=gvmcnt(N)`；A 可见性仍由 stage 顶 `bsmcnt(0)+__syncthreads`
保证。smem：默认保持 A 双缓冲 64KB 不变，开启时扁平数组放大到 112KB（A 48KB+B 64KB）。
```sh
scripts/build.sh so check mybench           # 默认关：逐字等价基线
KERNEL_CFLAGS="-DR24_BSMEM_B=1" scripts/build.sh so check mybench
```
- 正确性：关 `check --all` **762/762**；开 `check --all` **762/762**；`check --smoke` 均
  **8/8 PASS**（<10s）。
- 性能：s2 top50 **749.5→763.1µs（+1.8%）**；s1 TPR 32/128/256 = 483.5/967.9/1681.0
  （基线 482.1/947.4/1686.2）。**净回退，默认关**。
- 诊断：B-smem 把 `NOB` 暴露从 **211µs 降到 93µs**（延迟确实被隐藏 ~118µs），但
  `NOB` 地板由 544→677µs（+133µs）——B 经 smem 多一遍写+读，占用 smem 端口带宽。
  B 实为**带宽受限**（≈191µs@1.4TB/s），隐藏延迟不减少字节数。详见 `perf.md`。

## 方向3：通信/REDUCE 与计算重叠（`-DMXK_FUSE_REDUCE=1`，默认关，净中性/微回退）

时间标签 `[261012_11_10]`。承接 `-s2 top50` ~751µs → 400µs 目标，尝试「把通信/REDUCE
与计算重叠」。先把「通信地板」逐项消融量化（R=32，`us_mean`）：

| 消融（`-s2 top50`） | mean (µs) | 边际成本 |
|---|---:|---:|
| 基线 | 753.3 | — |
| `CM_SKIP_GEMM=1`（纯通信+REDUCE） | 174.9 | 地板（**非**与 GEMM 相加项） |
| `MX_SKIP_REDUCE=1` | 739.0 | REDUCE ≈ 14.3 |
| `PULL_LOCAL_DIAG=1` | 740.4 | A 远读 ≈ 13.0 |
| `R30_DIAG_NOPUSH=1` | 805.7 | 推送已重叠（去之反而更慢） |

**关键结论**：`CM_SKIP_GEMM` 的 174.9µs 地板**不等于可回收量**——A pull / COMBINE
推送 / 跨 block wait-all 大部分已被 GEMM 掩盖；s2 真正暴露的通信边际只有
REDUCE≈14µs + A 远读≈13µs ≈ **27µs（3.6%）**，方向3 上限约 749→722µs。s1 主要
暴露 A 远读 45~65µs（`PULL_LOCAL_DIAG`：TPR128 949.4→884.9、TPR256 1670.2→1625.4）。

候选1 实现（`MXK_FUSE_REDUCE`）：`reduce_body` 拆出 `reduce_tile_body`
（`src/combine.cuh:481`）+ wrapper；`mega_kernel` 任务循环后就地按 `(token,half)`
轮分执行（`src/kernel.cuh`）；`launch_full` 不再单发 `mega_reduce_kernel`
（`src/megakernel.cu:231`）；host 经新增 `MegaArgs::y`（`src/compute.cuh:78`）传输出。
```sh
scripts/build.sh so check mybench                         # 默认关：逐字等价基线
KERNEL_CFLAGS="-DMXK_FUSE_REDUCE=1" scripts/build.sh so check mybench
```
- 正确性：关 `check --all` **762/762**（78.6s）；开 `check --smoke` **8/8 PASS**（<10s）。
- 性能：s2 = 749.1→**755.0µs（+0.8%）**；s1 TPR32 485.2→492.8（+1.6%）、
  TPR128 950.5→983.8（+3.5%）、TPR256 1675.5→1717.1（+2.5%）。**净中性/微回退，默认关**。
- 根因：`MT_DUMP_RES=1` 显示 `full regs=256 smem=66160`——SM/寄存器已被 mega 占满，
  reduce 无并发窗口；且融合把 REDUCE 并行度由 `2*tokens` block 降到 `kGrid=32`
  （s1 大档降 16×）。跨 launch side-stream 重叠亦被 harness 阻断：本机
  `BeginCapture(ThreadLocal)` 下未 join 的侧流令 `EndCapture` 返回 904 且无图产出。

详见 `perf.md` [261012_11_10]。

## 方向4：win=512 + 跨 block 扁平池 NS4×NK8 L1 split-K（`-DR19_BLOCK_KSPLIT=1
-DR19_BLOCK_KSPLIT_WINMAX=512`，默认关，净回退）

时间标签 `[261012_13_55]`。目标 `-s2 top50` 734µs → 逼近 400µs：假设 win=512（每专家
1 窗）消除按窗重复读权重，再用 `(e, N片 s, K段 c)` 共 32 个等代价子任务填入同一扁平池
恢复负载均衡。**已用逐块相位诊断核实这是真正的块级派发**（非块内 K 循环）：

| 口径（s2 top50 mean, µs） | 默认(自适应win) | win=512 无切 | win=512 + NS4×NK8 |
|---|---:|---:|---:|
| 完整 | **733.9** | 1992.1 | 2381.7 |
| 纯通信+协议（`CM_SKIP_GEMM=1`） | 171.2 | 486.9 | — |

- **块级分布**（rank17, case 806200）：win=512 无切仅 **12/32** block 有 L1
  （热专家 4 个 N 片任务独占 b=24..27，L1=2.33M cyc，`max/min=7.66`）；加 NS4×NK8 后
  **32/32** block 有 L1（每块 5~8 任务，`max/min=1.46`）。均衡达成，但每块 L1 均值
  1.33M vs win=64 的 ~0.4M → **绝对耗时 ×3.3**。
- **根因**：K 切把「M（窗）并行」换成「K 并行」，A-smem 双缓冲只剩 3 段流水
  （`nstage=chunk/kKS`），每任务固定开销（流水未热身 + B 直装 + 协议）不随 K 缩小；
  本机权重**非瓶颈**（`CM_SKIP_GEMM` 仅 171µs），win=512 省下的权重重复读 < 失去并行的代价。
- **归约非瓶颈**：win512 base（无 atomicAdd）≈ ksplit；`r19_l1_reduce` 仅 ~227µs。
  「非重叠 `l1_partial[e][s][c]`」需 ×NK 缓冲且末段 reduce 读量 ×NK，**更慢，未采用**。
- 门禁：默认关 `check --all` **762/762**（75.4s）；开 `check --all` **762/762**（134.8s）、
   `--smoke` 8/8。**未接近 400µs**；两开关保持默认关。详见 `perf.md` [261012_13_55]。

## 方向5：fp4(e2m1) 权重 + packed 预取环（`-DMX_FP4_W=1`，默认关）

时间标签 `[261012_18_58]`。目标：W1/W2 以 fp4 打包存储（HBM 字节减半），kernel 用
`__builtin_mxc_cvt_pk8_f4tof8_scale_lo` 解包回 fp8 fragment 复用现有 fp8 MMA。

- **原型问题**：把 cvt 放进 B 预取环（`load→cvt` 即刻产出 fp8 fragment）时，
  `-s2 top50` 从 695µs 恶化到 877µs（B 暴露 216→427µs）。微基准（同会话，本机
  C600-UL 32SM）显示该 cvt 为**延迟受限**：128×256 时 362 Gcvt/s，32×256（=本 kernel
  的 4 warps/SM）仅 123 Gcvt/s。`MX_FP4_NOCVT` 隔离实验（只删 cvt，留 8B 装载）
  得 660µs < 基线 → **代价全在 cvt 的调度暴露，不在字节/装载宽度**。
- **修法（`MX_FP4_RING`，默认 1）**：预取环只缓存 **8B packed fp4**（寄存器减半），
  解包推迟到消费点，并按 k-step 批量展开 `bexp[kBatch]` 后再喂 MMA，给编译器更大
  调度窗口。应用到 r19/r24 的 L1/L2 四条 GEMM 路径。
- **A/B（同会话，us_mean）**：

| 口径 | 基线 | fp4+RING | 变化 |
|---|---:|---:|---:|
| `-s2 top50` mean（3 轮） | 698.2 | **581.6** | **−16.7%** |
| `-s2` B 暴露（`R24_DIAG_NOB`） | 216.4 | **134.0** | **−38%** |
| s1 TPR=32 | 446.9 | 396.9 | −11.2% |
| s1 TPR=64 | 620.1 | **494.2** | **−20.3%** |
| s1 TPR=128 | 915.3 | 890.9 | −2.7% |
| s1 TPR=256 | 1597.3 | 1562.5 | −2.2% |

- 门禁：`check --smoke` **PASS 8/8**，6.4s（<10s）；默认（`MX_FP4_W=0`）逐字等价基线。
- 遗留：距 500µs 仍差 ~82µs；`MX_FP4_W=1 + R24_BPF=4 + R19_L1_PF=8` 可再降至 ~567µs，
  但需做成 **fp4 条件默认**（否则回退默认 fp8 路径，历史 `-s2` TPR128/256 回退），未纳入。
  权重 fp4 量化/reorg 在 torch 侧尚未接入（本环境无 torch）。

## 方向5b：fp4 条件默认加深 B 预取环（`#if MX_FP4_W`）

时间标签 `[261012_19_24]`。承接上节遗留：把 `R24_BPF=4 + R19_L1_PF=8` 用
`#if MX_FP4_W` 做成 **仅 fp4 路径** 的默认（`src/compute_big.cuh`），`#else` 分支与
`-D` 覆盖语义逐字不变。

| 口径 | fp4 基线 (BPF2/PF2) | fp4 条件默认 (BPF4/PF8) | 变化 |
|---|---:|---:|---:|
| `-s2 top50` mean（3 轮） | 583.7 | **569.9** | **−2.4%** |
| s1 TPR=32 / 64 / 128 / 256 | 392.2 / 505.5 / 895.6 / 1562.2 | 389.2 / 494.8 / 894.4 / 1579.0 | −0.8% / −2.1% / ≈0 / +1.1% |

门禁：`check --smoke` PASS 8/8（6.2s，<10s）；`MX_FP4_W=0` 经 `static_assert` 探针确认
默认仍为 `R24_BPF=2 && R19_L1_PF=2`（逐字等价基线），`-D` 覆盖仍生效。详见
`perf.md` [261012_19_24]。

## 方案 Stage 1：peer 可见缓冲（`expert_tokens` / `l2_out` 迁入 FG `ranks_data`）

时间标签 `[261012_22_54]`。组内均衡方案（窗口按组内最小负载分配到组内 rank）需要
远端 rank 读取 A 行、回收 L2 输出，因此先把两块原 rank 私有缓冲迁入 `unk->rd`（FG，
peer 可读）。**纯数据放置，不改任务分配**，默认 fp8 路径。

| 项 | 变化 |
|---|---|
| `local_data.expert_tokens` / `local_data.l2_out` | 删除，改到 `ranks_data`（FG） |
| `rwk_data` 大小 | 1040MB → 637MB（迁出 402MB，FG 增加同量） |
| kernel 引用 | `S.pool_base`、`combine_push_pair` 与 `fuse.out_base` 等改为 `a->unk->rd.*` |

| 口径 | 基线 (9ce5306) | Stage 1 | 变化 |
|---|---:|---:|---:|
| `-s2 top50` mean（2 轮） | 698.4 ~ 698.8 | 690.4 ~ 691.3 | ≈ −1.1% |
| `-s1 TPR=16`（3 轮交替） | 320.8 ~ 327.7 | 316.0 ~ 322.6 | ≈ −1.5% |
| `-s1 TPR=32 / 256`（单轮） | 420.9 / 1597.2 | 416.2 / 1595.0 | ≈ 0 |

门禁：`check --smoke` PASS 8/8（6.9s，<10s）。FG 放置对 pool 读写无可见开销。

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

---

## C600UL 优化实录（2026-10-14，`agent/opt/preclear-gs` 分支）

线上口径：GLM-5.2 形状（H=6144 / I=2048 / E=256 / topk=8 / R=32）、28 SM/C600-U、
backend=mxmesh。性能口径为 MXMOEMega 的 `run_mega_moe_all.sh`：
50-in-1 CUDA Graph、bench_kineto 核时求和、rankavg、`rel=0` 数值门禁。

### 提交链（可复现）

```
c122068 — 环境原点
580ff40 — 环境基线（FIXED-WIN=48 / GS-ON 组权重 / MXK_GROUP_RANKS 透传）
92eed92 — preclear 跨 block 分片
222ac89 — GS 判定+LPT 改 block0 广播（mailbox）
a9af9ee — 28-SM 适配（kGrid 单一来源 src/grid_config.h，默认 28）
1082058 — pre_ticks 诊断读口（symm_rwk_ptr/rwk_ticks_off/ticks_count）
058f769 — _FIXED_WIN 48→64（65 个 case 实测最优全局窗口）
```

### 65 fixed-case 对比（R=32，rankavg，ms）

| case | baseline | mega 原版 | mega 优化 | 加速 原版→优化 |
|---|---:|---:|---:|---:|
| 804900 | 0.7038 | 0.6898 | 0.5582 | 1.01x → **1.26x** |
| 806600 | 1.8576 | 1.1411 | 1.1129 | 1.63x → **1.65x** |
| 807500 | 0.7305 | 0.8538 | 0.6405 | 0.84x → **1.15x** |
| 811800 | 0.6934 | 0.8397 | 0.6721 | 0.82x → **1.07x** |
| 813900 | 0.7158 | 0.7616 | 0.5657 | 0.95x → **1.28x** |
| 814500 | 0.6023 | 0.6744 | 0.5731 | 0.91x → **1.08x** |
| 814900 | 0.6587 | 0.8151 | 0.5533 | 0.81x → **1.20x** |
| 853900 | 0.6000 | 0.8540 | 0.6080 | 0.71x → **1.00x** |
| 855800 | 0.6233 | 0.6601 | 0.5442 | 0.94x → **1.16x** |
| 860900 | 0.9500 | 0.8791 | 0.6683 | 1.09x → **1.41x** |
| 870000 | 0.5638 | 0.6815 | 0.5617 | 0.82x → **1.01x** |
| 892800 | 0.6346 | 0.7792 | 0.5567 | 0.82x → **1.15x** |
| **全量** | — | **13/65，0.914x** | **64/65，1.176x** | 全 case 提升 |

- 重跑复现：基线 12/65@0.916x、优化 64/65@1.176x（同一二进制两次跑，mega 差 <1%）。
- 审计：图结构/预热/计时（双方核时求和）/输入张量全部同口径；mega 侧有约
  1–3% 优势（replay 前 `_sleep`+对齐、输入拷贝核不进求和）。`bench_kineto`
  发现的核名确认包含 `mx_fp8_*` + `mega_reduce_kernel`（REDUCE 计入）。

### 复现方法

```bash
# 环境（SDK 重组后：/opt/maca 与 /opt/maca-20261011 是指向
#   ~/opt/maca-sdk-c600u-20261011 的软链；mybench 的 mpirun 需要它）
export MACA_PATH=/opt/maca CUDA_HOME=/opt/maca/tools/cu-bridge \
       CUCC_PATH=$CUDA_HOME CUBRIDGE_HOME=$HOME \
       DG_JIT_NVCC_COMPILER=$HOME/cucc_maca_include.sh
export LD_LIBRARY_PATH=$MACA_PATH/ompi/lib:$CUDA_HOME/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/lib
export KERNEL_CFLAGS="--offload-arch=xcore1600"   # 缺省会报 ErrorNoBinaryForGpu

# 1) kernel + wheel（grid 28 缺省；换卡用 MXK_KERNEL_GRID=<SM数>）
cd mini_mega && bash scripts/build.sh mybench        # so（注意：test/megakernel.so）
cd torch && python setup.py bdist_wheel             # wheel（_C + 随包 megakernel.so）

# 2) 65 fixed-case 对比（32 卡，持 GPU 锁）
cd ~/MXMOEMega
DG_TEST_PACKAGE_ROOT=<解包的mxmegakernel wheel目录> \
PYTHONPATH=<同一目录>:/home/fuyx/pylibs-20261011 \
OUT_DIR=~/rd_x MASTER_PORT=34xxx \
bash scripts/mx_scripts/run_mega_moe_all.sh --num-processes 32 \
  --routing-fixed-cases all --ep-backend=auto
```

### 优化原理

1. **preclear 分片**（`mx_preclear_ct2`）：32 block 各清全量 → grid-stride 恰好一次
   （src_index_table 64MB/launch 清零量 ÷32）。清零幂等 + ct2 槽两轮后才读，
   分片安全。
2. **GS 广播**（`gs_mail` mailbox）：block0 做 1024 次 FG 读的倾斜判定 + LPT，
   经 `(epoch,~epoch)` 双字握手广播；27 个 block 的重复读全消。倾斜 case 省
   200–400µs；均衡 case 付 ~16µs（换 65-case 大幅收益，值得）。
3. **28-SM**（`src/grid_config.h`）：`kGrid`/`kWinGrid`/`kMegaGrid()` 单一来源，
   默认 28=设备 SM 数。实测 28-block 比 32-block 在 8/8 case 更快（−11~−75µs）。
4. **win=64**：65 case 扫 {48,64,96}，64 在 54/65 最优（hot expert 任务数减半）；
   代价模型 `t_task` 表（t48 32.2→20.7、t64 34.8→20.3）反转确认。
5. **pre_ticks 诊断**：`_C.symm_rwk_ptr/rwk_ticks_off/ticks_count` 读 block0
   时间线（kernel 零改动）。nt=1 分解：dispatch 97µs(28%) / L1 140µs(40%) /
   L2tail 76µs(22%) / prologue+xbar+setup 仅 ~35µs / REDUCE ~0.3µs。

### 已证伪 / 负收益（不再投入）

| 方向 | 结果 |
|---|---|
| kernel 内自适应选窗（counts→select_win→win_mail） | 全 case +34~+85µs：block0 256 次 FG 读串行化全 grid，决策成本 > 收益。已回滚 |
| 小路径阈值 R19_SMALL_MAX=32 | nt=17 数值发散（小路径有 ≤16 硬假设）。已回滚 |
| pull 去重（同窗 4 slice 重复拉） | 假设错误：srcp/dstp 均带 u4_off 列偏移，4 slice 按列分片、无冗余 |
| 4 倍重复清零/GS 广播阈值/自适应窗口之外 | preclear 分片与 GS 广播已落地（见上） |
| REDUCE 融合进 mega 尾（`MXK_FUSE_REDUCE`） | 旧结论 −0.8%；新口径 REDUCE 本身仅 ~0.3µs，问题在尾部同步，融合不解决 |
| coverage（均衡 21 点）整体优化 | 0/21；差距 = 固定 167µs + 4.28µs/token。断崖在 count>16 后 L2tail 尾部同步（+66µs）与 dispatch 随行数线性（+149µs@nt24→54），非 GEMM |
| mybench 作为性能口径 | 间歇 NoBinaryForGpu、单点抖动 20%+；性能决策一律改用 harness + wheel（已验证 NoBinary=0） |

---

## C600UL 优化实录·续（2026-10-14 下午，coverage 方向）

前节 65-case（倾斜负载）64/65 已达标；本节是 GLM5.2 coverage 均衡 21 点
（0/21）的归因与试错记录。结论先行：**coverage 口径无便宜可捡**，
差距 = 固定 167µs + 4.28µs/token（拟合：base≈304+8.34·nt，mega≈471+12.62·nt）。

### 稳态分相（pre_ticks，32 卡，`_C.symm_rwk_ptr/rwk_ticks_off/ticks_count`）

| nt | dispatch | L1 | L2tail | prologue+xbar+setup | REDUCE | chain |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 97 (28%) | 140 (40%) | 76 (22%) | ~35 (10%) | 0.3 | ~350 |
| 16→17 | +5 | +7 | **+66** | ~0 | 0 | +68 |
| 24→54 | +149 | +52 | +110 | ~0 | 0 | +329 |

- 之前"prologue+xbar+setup ~100µs"是误判，实际仅 ~35µs。
- L1 全程平坦：斜率问题不在 GEMM 计算，在"等/搬"。
- nt=16 双峰：热 rank L1~320（大路径多干活），冷 rank L2tail 最高 522µs
  （空等）——不均衡税的定量形态。

### nt=16→17 断崖（+177µs，阈值确认）

- nt=16→0.578ms，nt=17→0.755ms（+177），nt=18→0.771ms（+16）：跳变精确
  发生在 count 跨过 `kM=16` 处（`compute.cuh:1172` 小/大任务分派）。
- 分相：L2tail +66 主导，L1 仅 +7——是大路径下任务完成方差放大的尾部税，
  不是 M-tile 补齐。
- 小路径撑不住 >16（R19_SMALL_MAX=32 实验：nt=17 graph rel=0.926 发散，
  已回滚）。阈值是 load-bearing，不能动。

### win64 在 nt=54/60 的回退（+320µs，三轮复现）

- 计算段四位小数一致（L1/swiglu/L2 不动），REDUCE 排除（pre_ticks 0.3µs；
  MX_SKIP_REDUCE 差 7–17µs 落噪音带）。
- 机制：win64 把 count 54/60 推进 `(48,64]` B-smem 梯档
  （`compute_big.cuh:2202`，`win_cap<=64` 才进）；win48 时同 count 切两窗
  走 MT3 档。是"梯档阈值 vs 窗口"耦合，不是窗口本身错。
- 同一 trade-off 的两面：win64 让倾斜 65-case 里 54 个受益。固定 win64
  仍是正确选择。

### R30_STAGGER=0 对照（无效果）

- nt=17/24/54：+4/+5/+2µs，全在噪音带，rel 全过。
- 结论：列块乱序不是关键路径；PUSH_DEFER 的前提（乱序能 overlap 推送）
  已证伪，不再试。L2tail 方向暂时封存（只剩"让 4 个 slice 算得一样快"，
  即负载均衡本身）。

### 方法论修正（审计结论）

- `stages.tsv` 只是 baseline 的单发 profiling，不能归因 mega。
- `rel=0` 只是 graph-vs-eager 自洽；harness 无 mega-vs-baseline 数值对拍。
- REDUCE 计入 mega 求和（日志 130 次确认双核名）；SKIP 差值在噪音带。
- 两边计时对称（50-in-1/预热10/核时求和/暖L2/同输入），mega 侧残余
  ~1–3% 优势（replay 前 sleep+对齐、输入拷贝核不进求和）。
- win64 有轻微样本内偏差（3 选 1），幅度大，影响小；等新 case 做样本外验证。
- 基线复现 0.913→0.914x，优化复现 1.168→1.176x（mega 列差异 <1%）。

### 分支状态（2026-10-14 17:30）

```
main                        c122068（已复位，等审查合并）
agent/opt/env-baseline      580ff40（环境基线：_FIXED_WIN=48/GS-ON/GROUP_RANKS）
agent/opt/preclear-gs       …→058f769 win64→本节README（7→8提交，单链干净）
```

---

> 合并注（2026-10-15）：以下为 `agent/glm52a/opt` 分支合入的 coverage 调研记录（含 L2TUNE-49 落地），与上一节互补保留。

## coverage 均衡 21 点调研（2026-10-14，`agent/glm52a/opt` 分支）

背景：65 fixed-case（倾斜路由）已到 64/65@1.176x，但 coverage 均衡 21 点
（`--routing-mode coverage`，tokens=1..256，32 rank 负载严格均匀）仍是 0/21
（0.54–0.81x）。本节记录根因定位的机制发现、实验数据与复现方法。所有实验在独立
工作区 `mini_mega/.agents/work/glm52a/wt`（分支 `agent/glm52a/opt`）内执行。

### 1) 现状复现（口径一致）

`rd_cov_glm52a/summary.tsv`（本轮）与 `rd_cov_w64/summary.tsv`（历史）逐点差 ≤2%；
mega 计时=图 50-in-1（graph_reps=50）+ bench_kineto 核时求和（核名
`mega_reduce_kernel`+`mega_kernel`），rankavg，ms：

| tokens | window(逻辑*) | baseline | mega | speedup |
|---:|---:|---:|---:|---:|
| 1 | 8 | 0.3086 | 0.4153 | 0.74 |
| 2 | 8 | 0.3224 | 0.4229 | 0.76 |
| 4 | 8 | 0.3540 | 0.4439 | 0.80 |
| 6 | 8 | 0.3705 | 0.4597 | 0.81 |
| 8 | 8 | 0.3786 | 0.4809 | 0.79 |
| 12 | 16 | 0.4260 | 0.5344 | 0.80 |
| 18 | 24 | 0.5118 | 0.7703 | 0.66 |
| 24 | 24 | 0.5037 | 0.8553 | 0.59 |
| 30 | 32 | 0.5583 | 0.9168 | 0.61 |
| 36 | 48 | 0.6275 | 0.9755 | 0.64 |
| 42 | 48 | 0.6677 | 1.0522 | 0.64 |
| 48 | 48 | 0.7004 | 1.1239 | 0.62 |
| 54 | 64 | 0.7477 | 1.3630 | 0.55 |
| 60 | 64 | 0.7902 | 1.4756 | 0.54 |
| 66 | 80 | 0.8628 | 1.2155 | 0.71 |
| 72 | 80 | 0.9038 | 1.2559 | 0.72 |
| 78 | 80 | 0.9316 | 1.3348 | 0.70 |
| 84 | 48 | 0.9773 | 1.4696 | 0.67 |
| 90 | 48 | 1.0115 | 1.5660 | 0.65 |
| 128 | 64 | 1.2731 | 2.2399 | 0.57 |
| 256 | 64 | 2.5142 | 3.5902 | 0.70 |

\* window 列=harness `compare-window` 打印的 `select_fp8_mega_moe_window()` 返回值，
**不是**计时 launch 的实跑窗口（见 2）。

coverage 分布（harness `compare-routing` 行）：total_pairs=tokens×256、
load_per_rank=8×tokens（min=max，严格均衡）→ 每本地专家行数 = 每 rank tokens。

### 2) 关键发现：wheel `_FIXED_WIN=64` —— 计时 launch 恒用 win_cap=64

- `torch/mxmegakernel/mega/__init__.py:50`：`_FIXED_WIN = 64`；`fp8_mega_moe` /
  `fp8_mega_moe_compute` 内 `win_cap, skip = _FIXED_WIN, 0`（:636/:687，注释残留
  “固定 48”）；C++ 侧 `wc = (win_cap>0) ? win_cap : b->win`（`mega.hpp`
  `make_mega_args`）→ 所有计时 launch 实跑 64。
- harness 的 `compare-window` 行（8/16/…/64/80）只写入 `sym_buffer.expert_window`
  （逻辑窗口：参与 launch 校验 ew∈[8, win_phys] 与 warmup），对计时 launch 无效；
  `DG_MEGA_WIN_AUTO` 在 wheel 内无读取点（仅 harness CLI 帮助文本提及）。
- 交叉验证：mybench `MT_FORCE_WIN=64` 与 harness coverage 逐点吻合（TPR=1:
  415.7 vs 415.3；66: 1257.6 vs 1215.5；128: 2372 vs 2240；256: 3699 vs 3590；
  mybench 系统偏高 1–6%），而 W48/W96 与 harness 数值明显偏离 → 证实计时恒 64。
- 该 64 为 65 fixed-case 实测全局最优（commit 058f769），**保持默认**；下文的
  逐点窗口仅为实验记录/待验证优化，不改默认。

### 3) 实验 A：固定窗口扫描（MT_FORCE_WIN，mybench -R 32 -s 1，us_mean）

| W \ TPR | 54 | 66 | 128 |
|---:|---:|---:|---:|
| 32 | 1390 | 1387 | 2424 |
| 48 | **1137** | 1434 | 2269 |
| 64 | 1446 | **1260** | 2318 |
| 96 | 1505 | 1683 | **2229** |
| 128 | 1505 | 1704 | 2809 |

- 逐点最优≠全局：54 想 win48（−309µs vs 64）、66 想 win64（已最优）、128 想 win96
  （−89µs）；单一全局窗口无法同时最优。
- 非单调：win64 下 harness nt=60=1475.6µs > nt=66=1215.5µs（多 6 token 反快
  260µs；nt=66 起每专家行数 66>64 → 每专家分 2 窗 64+2）。机制未完全归因
  （候选：窗划分→两段流水、staging stride、`R19_BLOCK_KSPLIT_WINMAX=64` 门控组合）。
- 窗口敏感性来源（源码）：win_cap 同时决定 ①每专家每窗行数上限
  Σ_e ceil(count_e/win_cap) ②staging/pool/l1 行 stride ③fragment m 维
  ceil(win_cap/16) ④K-split 门控（≤64 启用）。

### 4) 实验 B：相位诊断（MYBENCH_PHASE=1，block0 pre_ticks 时间线）

方法：`MT_FORCE_WIN=64 MYBENCH_PHASE=1 run.sh mybench -s 1 -tprs …`：常规计时后
reset + 单发 eager launch 读 tick（**eager 口径**，非 graph 50-in-1，绝对值含冷启动
成分，用于比较相位结构；graph lat 列=同 run 常规计时）。µs@≈1.497GHz：

| TPR | graph lat | dispatch(r0) | L1(r0) | L2tail(r0 / max@rank) |
|---:|---:|---:|---:|---:|
| 1 | 416 | 28 | 171 | 92 / 291@23 |
| 8 | 511 | 69 | 170 | 147 / 367@18 |
| 24 | 916 | 182 | 201 | 351 / 687@29 |
| 54 | 1415 | 341 | 232 | 561 / 1096@14 |
| 60 | 1528 | 379 | 247 | 545 / 1147@23 |
| 66 | 1258 | 278 | 214 | 794 / 1073@8 |
| 128 | 2372 | 384 | 208 | 1833 / 2237@9 |
| 256 | 3699 | 209 | 227 | 3219 / 3959@24 |

相位语义（pre_ticks）：t3→t4=首任务 pull+staging+分片旗；t5→t7=L2tail 尾部同步；
route/publish/xbar/setup 合计在热态约占 3–8%。

窗口对照（同法）：W48@54 dispatch 198 / L2tail 677（max 914@22），graph lat 1095
（vs W64@54 1415，**−320µs**）；W48@66 215 / 1007（比 W64 慢）；W96@128 431 /
1524（max 1819），graph lat 2226（vs W64 −146µs）。→ 54 点的窗口收益主要来自首任务
staging 缩小与最慢 rank L2tail 收缩。

### 5) 结论与开放项

- 已闭合：计时恒 win64 机制；窗口敏感性数据（逐点最优 ≈ 54→48、66→64、128→96）；
  nt=60/66 非单调记录；权重路径（仅 GS 组指针变化才 D2D 拷入 rwk，无逐 launch
  300MB 拷贝）与 pull 路径（按列分片、无冗余）复核。
- 逐点窗口（env-gated，不动默认 64）预估可挽回：54 点 −320µs（≈0.55→0.70x）、
  128 点 −90～150µs；仍不足以整体 >1x——21 点的结构性差距在 dispatch 随行数增长
  与 L2tail 尾部同步（28 block 下由最慢 rank/block 决定）。
- 未决：60→66 非单调的精确机制；kernel 内自适应选窗（已证伪，勿重试）；
  设备码变体漂移（构建后 smoke 门禁，见上文“mybench 作为性能口径”条）。
- 下一步候选：(a) env-gated 逐点窗口 + 21 点 harness 复测；(b) dispatch/staging
  与 L2tail 同步的结构性优化；(c) 非单调机制定因。

### 6) 复现命令

```bash
# 环境（SDK 重组后：/opt/maca 与 /opt/maca-20261011 指向
#   ~/opt/maca-sdk-c600u-20261011；mybench 的 mpirun 需要它）
export MACA_PATH=/opt/maca CUDA_HOME=/opt/maca/tools/cu-bridge \
       CUCC_PATH=$CUDA_HOME CUBRIDGE_HOME=$HOME \
       DG_JIT_NVCC_COMPILER=$HOME/cucc_maca_include.sh
export LD_LIBRARY_PATH=$MACA_PATH/ompi/lib:$CUDA_HOME/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/lib
export KERNEL_CFLAGS="--offload-arch=xcore1600"   # 缺省报 ErrorNoBinaryForGpu
# 本机 32 卡（0..31）：test_env.sh 默认面向 16..23 卡，HCA/网卡需覆盖
export MXSHMEM_HCA_LIST=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1
export MCCL_IB_HCA=$MXSHMEM_HCA_LIST
export MCCL_SOCKET_IFNAME=ens59f0np0 NCCL_SOCKET_IFNAME=ens59f0np0

# 1) wheel（wt 内；产出 torch/dist/mxmegakernel-*.whl）
cd /home/fuyx/mini_mega/.agents/work/glm52a/wt
MACA_PATH=/opt/maca PYTHON=/home/fuyx/myenv/bin/python bash scripts/build.sh whl
unzip -o torch/dist/mxmegakernel-*.whl -d /home/fuyx/wheel_glm52a_pb
md5sum /home/fuyx/wheel_glm52a_pb/mxmegakernel/megakernel.so
#   → L2TUNE-49 前：65bd063241300c388e2dd44f79329102
#   → L2TUNE-49 后：e3485a545900cd9cdb09a3aa6ac06d22（check --all 762/762、
#     smoke 11/11；设备码变体漂移时需重编到与前一个可用 md5 一致的变体，
#     或用 run.sh check --smoke 门禁后再部署）

# 2) coverage 21 点（32 卡；持共享 GPU 锁）
#   环境（run_mega_moe_all.sh 缺省 PYTHON=py310-maca 本机不存在，必须显式给；
#   缺 MCCL_IFNAME/HCA 会 rank0 “Bootstrap: no socket interface found”→NCCL init 失败）
export MCCL_SOCKET_IFNAME=ens59f0np0 NCCL_SOCKET_IFNAME=ens59f0np0 \
       MXSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=ens59f0np0 GLOO_SOCKET_IFNAME=ens59f0np0
export PYTHON=/home/fuyx/myenv/bin/python
cd /home/fuyx/MXMOEMega
DG_TEST_PACKAGE_ROOT=/home/fuyx/wheel_glm52a_pb \
PYTHONPATH=/home/fuyx/wheel_glm52a_pb:/home/fuyx/pylibs-20261011 \
OUT_DIR=/home/fuyx/rd_cov_glm52a MASTER_PORT=34553 \
flock -w 3600 /home/fuyx/mini_mega/.agents/work/lock/lock_gpu \
bash scripts/mx_scripts/run_mega_moe_all.sh --num-processes 32 \
  --routing-mode coverage --ep-backend=auto
# 输出 summary.tsv / run_metadata.txt / profile.log（全程约 2 分钟）；
# 65 定档换 --routing-fixed-cases all（约 2.5 分钟）；历史对照 /home/fuyx/rd_cov_w64/

# 3) 相位诊断（wt 内；worktree 里 run.sh 的 GPU_LOCK 会落到私有锁，
#    需外层持共享锁 + GPU_LOCK=0）
cd /home/fuyx/mini_mega/.agents/work/glm52a/wt
flock -w 3600 /home/fuyx/mini_mega/.agents/work/lock/lock_gpu \
env GPU_LOCK=0 MT_FORCE_WIN=64 MYBENCH_PHASE=1 \
bash scripts/run.sh mybench -R 32 -s 1 -tprs 1,8,24,54,60,66,128,256 -quiet

# 4) 固定窗口扫描
for W in 32 48 64 96 128; do
  flock -w 3600 /home/fuyx/mini_mega/.agents/work/lock/lock_gpu \
  env GPU_LOCK=0 MT_FORCE_WIN=$W bash scripts/run.sh mybench -R 32 -s 1 -tprs 54,66,128 -quiet
done
```

### 7) 本轮产物清单

- wheel：`torch/dist/mxmegakernel-0.0.0+mini.mega.2026101417-*.whl`（解包到
  `/home/fuyx/wheel_glm52a_pb`，megakernel.so md5 65bd0632…；L2TUNE-49 后重打为
  e3485a54…，旧部署备份 `wheel_glm52a_pb_prel2fix/`）
- coverage：`/home/fuyx/rd_cov_glm52a/{summary.tsv,run_metadata.txt,profile.log}`
  + 完整日志 `.../glm52a/tmp/cov_pb_run.log`；历史对照 `/home/fuyx/rd_cov_w64/`
- 相位/扫描日志：`.../glm52a/tmp/{phase_s1,phase_w64,phase_w48,phase_w96,winsweep}.log`
- 可用构建备份：`.../glm52a/tmp/backup_good_1714/`（65bd0632 产物 + objs）

### 8) [L2TUNE-49] L2 49..63 慢回退修复（coverage nt=54/60 −93/−99µs；65case 大面积提速）

#### 问题：W64 下 TPR 49..63 断崖

mybench（`-R 32 -s 1`，W64）悬崖探针：48→49 跳 +179µs（1122.9→1302.3），49..63
整段偏离 48↔64 线性弦线 +74..160µs；64 回到趋势线，66（每专家两窗 64+2）反而
更快。coverage 21 点命中 54/60（每专家行数=49..63），恰是 speedup 最差两点
（0.55/0.54）。_FIXED_WIN=64 下凡每专家窗行数落入 49..63 的 case 全部中招
（均衡 TPR=49..63；倾斜 case 中任一专家窗该区间）。

#### 根因

L2 任务分派链的 [OPT-64] 门控 `count % (4*kM) == 0`（count≤64 时仅 64 通过）
把 49..63 排除在 r24 A-smem 路径外 → 掉进 `r19_l2_big<4, R19_L2_PF, 3>` 慢
回退（MT4 M 空转 + 非 r24 管线）。同构问题 L1 侧已在 [L1TUNE-D] 用「掩码 MT4
r24」修掉（见上文「C600UL 优化实录」），L2 从未镜像。

#### 修复（src/compute_big.cuh，r19_big_task L2 链，14+/6-）

去掉 `count%64==0` 门控，函数体内二分支：count%64==0 → 原无掩码
`r24_l2_big<4, R24_L2_KS, 3, false, 4>`；49..63 → `kMask=true, kMTAct=4` 掩码版
（与 L1TUNE-D 完全同构）。掩码路径安全性沿用既有语义：越界 M-tile 不搬运
（r24_issue_a）、m_abs0+m ≥ mt_total 折叠到末 tile、尾部 out/acc_chk 由
`row < cc` 保护、残余组数经 kBatch=1 递归。不动 L1、不动 win_cap、不加开关/
宏（NSLICE=8 等构建同样生效）。

#### 验证数据（v3 构建 e3485a54；前值=修复前同口径）

A. mybench 悬崖探针（W64，`-R 32 -s 1`，us_mean）：

| TPR | 48 | 49 | 50 | 52 | 54 | 56 | 60 | 62 | 64 | 66 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 修复前 | 1122.9 | 1302.3 | 1335.3 | 1374.0 | 1399.5 | 1426.6 | 1499.5 | 1497.8 | 1461.9 | 1253.5 |
| 修复后 | 1124.0 | 1211.1 | 1233.1 | 1272.8 | 1302.8 | 1327.5 | 1399.7 | 1401.2 | 1456.0 | 1255.3 |
| Δµs | +1 | **−91** | **−102** | **−101** | **−97** | **−99** | **−100** | **−97** | −6(噪声) | +2(不受影响) |

B. harness coverage 21 点（32 卡，graph 50-in-1 rankavg，mega_rankavg ms）：

| tokens | 1 | 2 | 4 | 6 | 8 | 12 | 18 | 24 | 30 | 36 | 42 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 修复前 | .4153 | .4229 | .4439 | .4597 | .4809 | .5344 | .7703 | .8553 | .9168 | .9755 | 1.0522 |
| 修复后 | .4155 | .4217 | .4424 | .4607 | .4789 | .5341 | .7735 | .8565 | .9165 | .9761 | 1.0586 |

| tokens | 48 | **54** | **60** | 66 | 72 | 78 | 84 | 90 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 修复前 | 1.1239 | **1.3630** | **1.4756** | 1.2155 | 1.2559 | 1.3348 | 1.4696 | 1.5660 | 2.2399 | 3.5902 |
| 修复后 | 1.1226 | **1.2704** | **1.3767** | 1.2193 | 1.2469 | 1.3325 | 1.4649 | 1.5805 | 2.2432 | 3.5891 |
| Δµs | −1 | **−93** | **−99** | +4 | −9 | −2 | −5 | +15 | +3 | −1 |

54/60 speedup 0.549→0.585、0.536→0.566；其余点 ±15µs 为跑间抖动（baseline 同步
漂移；两端点已复核为噪声口径）。数据：`rd_cov_glm52a`（前）→`rd_cov_l2fix`（后）。

C. 65 定档 harness（`--routing-fixed-cases all`，rd_re_opt 前 → rd_65_l2fix 后）：
mega 逐点 Δ 均值 **−9.75µs**、中位 −9.10（min −35.0 / max +12.5）；38/65 改善
（<−5µs）、8/65 小回退（+5.0..+12.5：863100/850800/844400/806600/838700/
853900/843500/889500）；speedup≥1.0：65/65 → **64/65**（853900 由 1.000 到
0.989 边界擦线），均值 **1.1761→1.1943**。

D. TK top50 mybench A/B（每二进制 3 跑交替取均值）：Δ 均值 **−14.35µs**、中位
−12.37；38 项 <−3µs、9 中性、3 项 >+3µs（804700 +5.5σ8.4、805400 +10.8σ2.4、
809600 +3.0；三者均不在 65 定档集内）。最大改善：804900 −47.7、807100 −37.9、
808800 −37.4、804300 −34.6。

正确性门禁：`check --all` **762/762 PASS**（100.4s）、`check --smoke` 11/11
（6.7s）；harness 内嵌 graph-vs-eager 逐点 rel=0.000e+00。

#### 残余结构差（非路径问题）

修复后 TPR=49 仍高于 48↔64 弦线 ~66µs：49..64 的 L2 任务从 [OPT-33] 的 MT3
（3×16 行 tile）跳到 MT4（ceil(count/16)=4 tile），16 行 MMA tile 量化天然产生
一个台阶；且该档每 rank 仅 8 个单窗任务（66 点起是每专家两窗 16 任务，并行度
翻倍）。属窗口/tile 粒度结构性差，需 M 维细分或任务并行类改动，不在本修复范围。

#### 构建/设备码漂移记录（本轮实证）

同源两次 cucc 编译：v1（d781ff37）`check --smoke` 挂——804900 自旋超时
（11.2s）、L0/L1L2 FAIL；重编 v3（e3485a54）全部门禁通过。**同源不同变体，
部署前必须过 smoke 门禁**；v1 产物留档
`.../glm52a/tmp/megakernel_l2fix_d781ff37.so`。

#### 复现命令

```bash
# 环境同 6)（MACA/KERNEL_CFLAGS + NIC env + PYTHON）；以下均在 wt 内、持共享锁
# 悬崖探针（前=backup_pre_l2fix 的 mybench+.so；后=test/ 当前产物）
cd /home/fuyx/mini_mega/.agents/work/glm52a/wt
flock -w 3600 /home/fuyx/mini_mega/.agents/work/lock/lock_gpu \
env GPU_LOCK=0 MT_FORCE_WIN=64 \
bash scripts/run.sh mybench -R 32 -s 1 -tprs 48,49,50,52,54,56,60,62,64,66 -quiet
# TK top50 A/B（两产物目录各自含 mybench+megakernel.so）
flock -w 3600 /home/fuyx/mini_mega/.agents/work/lock/lock_gpu \
env GPU_LOCK=0 bash scripts/run.sh mybench -R 32 -s 2 -top 50 -quiet
# 正确性门禁
flock -w 3600 /home/fuyx/mini_mega/.agents/work/lock/lock_gpu \
env GPU_LOCK=0 bash scripts/run.sh check --all      # 762 用例 ~100s
# harness：覆盖 21 点 / 65 定档（见 6)）
```

#### 相关待办

- NSLICE=8（g_l2=3）构建下 count 33..48 有同类 L2 空洞（`count<=48` 分支要求
  `g_l2%6==0`，g_l2=3 走不到 [OPT-33]）；49..63 已被本修复顺带覆盖
  （本修复分支无 g_l2 门控）。NSLICE=8 几何未在本轮实测，需单独构建验证。

### 9) [261014_21_59] 排空支配模型 + 17/18 断崖量化（诊断基建 + ISA 实证；生产路径零改动）

#### 结论先行

1. **`__stwt` 在本平台 ≡ 普通缓存写（ISA 位级证据）**：cucc 对 `__stwt`/`__stwb`
   生成的编码与普通写完全相同（`stg_b64 ... devc`，`bc 00 00 30…`）；仅
   `__stcs`/`__stcg` 带 sysc 位（`bc 00 00 70…`）。即项目全部跨 rank 推送
   （行 + 完成标记）实为**惰性缓存写**，靠驱逐/排空才到达对端——统一解释历史现象：
   标记早写晚见（wend ≈ −23k 但 rspan 300–600k）、run 间 rspan 波动 380k→605k、
   原子读探针不可见（atomicOr(…,0) 实验无效果，`coh_ctrl.log`）。
   volatile 读 = `ldx … sysc devc`（一致读），**读侧无陈旧问题，延迟全部来自写侧排空**。
2. **迭代周期受「外发洪峰排空」支配**：kspan+rspan 跨配置近乎恒定（TPR=64 各档
   ~2.8–3.0M cycles）。三个 store 风味实验只把记账在 kspan↔rspan 间搬家：
   | 实验 | TPR64 period | kspan | rspan | wall |
   |---|---:|---:|---:|---:|
   | 基线 __stwt | 2.876M | 2.180M | 605k | 1480.7µs |
   | MARK-STCS（标记→__stcs） | 2.928M | 2.414M | 406k | 1485.7µs |
   | MARK-ATOMIC（标记→atomicExch） | 2.924M | 2.914M(in=194k) | 127k | 1571.1µs |
   | ROW-STCS（行推送→__stcs） | 2.75–2.80M | 2.288M | 466k | 1489.2µs |
   → store 风味不是杠杆；有效外发带宽 D≈2–4GB/s/rank 是墙（用 `atomicExch` 把标记
   挪进请求路径后 rspan 崩塌至 127k，但 kspan 同步 +500k，period 不变）。
3. **64 vs 65 真实结构差 ≈ 末波洪峰**：TPR=64 末波 = el7 整窗 768KB/rank 突发收尾
   （无后续计算遮挡）；65/66 的 el7 大窗在 wave1 完成，末尾 1–2 行 tiny 波吸收排空
   → 快 ~180µs。属调度结构涌现差，非路径缺陷。
4. **17/18 断崖量化（36 行 tile 量化）**：16→17 突增 +184µs，全部落在 mega loop
   内：L2 相位 17..24 全程平坦（~16.3–17.2M ticks，不随行数下降）——17 行必须开第
   2 个 16 行 MMA tile，tile2 近空转仍付全价（固定 ~145µs）；33/49 台阶经
   [OPT-33]/[L1TUNE-D]/L2TUNE-49 已收敛（+9µs/+70µs）。属 16 行 tile 粒度结构性
   差，无廉价通用修复（小档路径仅支持 ≤16 行，win_cap=64 不可动）。
   分档台阶实测（us_mean）：16:627 · 17:809 · 24:910 · 32:987 · 33:997 · 48:1178
   · 49:1248 · 64:1493。

#### 诊断基建（`R24_DIAG_PHASE=1` 构建；生产构建零影响）

- 内核侧 5 时间戳 bank（`local_flags.pre_ticks` 扩展）：bank0 原有；第三 bank
  （+kGrid*16）逐 block 起止 + 任务循环入口/出口；第四 bank（+kGrid*24）单任务
  span min/max/sum/n；第五 bank（+kGrid*32）REDUCE 逐块 start/end/spin/token +
  逐切片自旋 + 逐 (k,s2) 标记到达时刻。**条件化分配**：诊断 bank 仅在
  `#if R24_DIAG_PHASE` 下扩展数组，生产构建 `pre_ticks[kGrid*16]` 与基线布局
  逐字节一致（`sizeof(rwk_data)=1543676672` 已核对）。
- mybench 侧（同样 `#if R24_DIAG_PHASE`）：`MYBENCH_PHASE=1` 输出
  `[span-graph]`（period/kspan/b0span/in/loop/post）、`[spanR-graph]`
  （gap1/rblk0/rspan/gap2/wend）、`[spanT-*]`（单任务分布）、`[spanR2/3/4-*]`
  （reduce 逐块/逐切片/逐 k 标记到达）；`MYBENCH_PHASE_RANK=0` 输出
  `[phase-blk-*]`（逐 block L1/L2/pull/move/wait/pw/loop/tail/tasks）。
- ISA 探针（`tmp/stwt_probe/probe2.cu`，19 行：plain/__stwt/__stcs/__stcg/__stwb
  写 + volatile/__ldcv/__ldcs/__ldcg 读 + atomicExch/Max）：

```bash
# 环境同 6)；wt 内
cd /home/fuyx/mini_mega/.agents/work/glm52a/tmp/stwt_probe
/opt/maca/tools/cu-bridge/bin/cucc -O3 -std=c++17 -I$CUDA_HOME/include \
  -I/opt/maca/include --offload-arch=xcore1600 -c probe2.cu -o probe2.o
/opt/maca/mxgpu_llvm/bin/mxobjdump --print-code --print-instruction-encoding \
  --target-kind=maca-mxc-metax-macahca--xcore1600 --demangle probe2.o
# 判读：k_stwt 与 k_plain 编码相同（bc 00 00 30）；k_stcs/k_stcg 为 bc 00 00 70（sysc）
```

#### 复现命令

```bash
cd /home/fuyx/mini_mega/.agents/work/glm52a/wt
# 诊断构建（5 bank + span 输出）
export MACA_PATH=/opt/maca CUDA_HOME=/opt/maca/tools/cu-bridge   # 分条 export
export DJ_JIT_NVCC_COMPILER=$HOME/cucc_maca_include.sh
export LD_LIBRARY_PATH=/opt/maca/ompi/lib:/opt/maca/tools/cu-bridge/lib:/opt/maca/mxgpu_llvm/lib:/opt/maca/lib
touch src/combine.cuh && rm -f test/megakernel.so test/mybench
KERNEL_CFLAGS="--offload-arch=xcore1600 -DR24_DIAG_PHASE=1" bash scripts/build.sh mybench
# 6 点快筛（对比 kspan/rspan 记账；16..24 断崖）
flock -w 3600 /home/fuyx/mini_mega/.agents/work/lock/lock_gpu \
env GPU_LOCK=0 MT_FORCE_WIN=64 MYBENCH_PHASE=1 MYBENCH_PHASE_RANK=0 \
bash scripts/run.sh mybench -R 32 -s 1 -tprs 64,65,65,64,63,66 -quiet
# 生产构建（无 -D）：布局/行为回归基线
KERNEL_CFLAGS="--offload-arch=xcore1600" bash scripts/build.sh mybench
```

#### 门禁与产物

- 生产重建 `f6beeb10`：21 点 coverage（`cov21_gate.log`）全部落 ±10µs 噪声带
  （TPR256 +21µs=0.6%）；`check --all` **762/762**（112.5s，`check_all_gate.log`）。
  与上一部署产物 e3485a54 的差异 = cucc 变体漂移（同源异码，见 8)），行为经双门禁
  等价。mybench `b95a7289`（打印门控，`MYBENCH_PHASE=1` 下生产构建零 span 垃圾输出）。
- 实验日志：`tmp/{spanw,stcs,matom,rowstcs,cliff,tier}_ctrl.log`、`cov21_gate.log`。
- 未采用/回退：`[EXP-COH]`（reduce 自旋内原子读探针，无效果，已回退）；
  `combine_push_pair`/`combine_mark_rows` 均已回 `__stwt` 基线语义。

