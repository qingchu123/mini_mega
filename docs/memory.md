# mini_mega 缓冲区归属与容量

## 维度约定

| 符号 | 当前口径 |
|---|---|
| `R` | rank 数 |
| `E` | 全局 expert 数，通常 `R*kEPR` |
| `kEPR` | 每 rank 本地 expert 数，8 |
| `kTopk` | 每 token 路由数，8 |
| `H` | `kHidden`，6144 |
| `I` | `kIntermediate`，2048 |
| `kRing` | 每 expert 窗口 ring 数，8 |
| `kGrid` | 一次并发任务 block 数，32 |
| `cap` | `max_tokens`/slot_capacity |
| `win` | 单项窗口容量，`8 <= win <= 128` |

`slot/`win` 当前在多个路径上是等价约束。sym 布局见 `src/host_layout.cuh`（ C++ bench ）与 torch/csrc/apis/layout.hpp。

## 1. 跨 rank 可写区 `FG/HBM`

这些属于对称区 `sym` 或 `ranks_flags`，可通过 FG allocated/IPC 给 peer 访问。

| 缓冲 | 最大维度/地址形式 | 说明/写者→读者 |
|---|---|---|
| `sym.src_index_table` | `[EPR][R][cap]` uint32 | `ROUTE_BUILD` 各 peer 散布写，目标 rank `DISPATCH` 读 |
| `sym.expert_recv_counts` | `[R][ceil((EPR+1)/4)*4]` uint32 | peer `PUBLISH_COUNTS` 写，目标 rank `TASK_SETUP` 校验读 |
| `sym.combine_rows` | `[kTopk][cap][H]` BF16 | expert rank `COMBINE` 写，目标 rank `REDUCE` 读 |
| `ranks_flags.combine_sums` | `[8][kTopk][512][32]` uint32 | `COMBINE` 标记，`REDUCE` 等待 |
| `ranks_flags.xbar_barrier` | `[8][kGrid][kGrid]` uint32 | peer route-ready arrive/wait |
| `sym.barrier_slots` | `[R][R]` uint32 | 历史保留/布局占位，当前主链已迁 `ranks_flags` |

## 2. 本 rank 私有 HBM，本地生产和消费

以下用 ordinary `mcMalloc` 分配（非跨 rank 数据）。

| 缓冲 | 最大维度 | 说明/写者→读者 |
|---|---|---|
| `local_flags.current_time` | `[kGrid][64]` int64 | 每 block/每 phase 单调计数，graph replay 仍执行 |
| `local_flags.bar` | `[8][1032]` uint32 | task/slice/l1/l2/nslice 旗，本 rank 内用 |
| `local_flags.out_total/out_processed/dbg/ck_dbg/pre_ticks/l1_chk/l2_chk` | `grid`、`grid*16` 或常量 | TASK_SETUP/TASK_LOOP 写，host/decode 诊断读 |
| `w1` | `[kEPR][2I][H]` uint8 | host 或 `fill_weights_lane` 写，L1 读 |
| `w2` | `[kEPR][H][I]` uint8 | host 或 `fill_weights_lane` 写，L2 读 |
| `ffn_hidden` | `[kEPR*kRing][win][I]` uint8 | L1 写，L2 读 |
| `l2_out` | `[kGrid][win][H]` uint16 | L2 写，COMBINE 读 |
| `frag_a` | `[kEPR*kRing][(H/32)*ceil(win/16)*512]` uint8 | DISPATCH 写，L1/L2 A fragment 读 |
| `frag2` | `[kEPR*kRing][(I/32)*ceil(win/16)*512]` uint8 | L1 写，L2 读 |
| `y` | `[max_tokens][H]` float | REDUCE 写，host/上层读回 |
| `ynode[n]` | `[max_tokens][H]` float | graph 每个 node 独立输出，上层读回 |
| `d_base_ptrs` | `[R]` uint8* | host 构造，kernel 取 peer sym 地址 |
| `d_rf_ptrs` | `[R]` ranks_flags* | host 构造，kernel 取 peer `rf` 地址 |

## 3. 本 rank 本地使用，但仍在 `sym` / HF 对称区（FG/HBM）

这些逻辑角色是 local only，但为了同一 IPC 共享边界，仍随 `sym` 分配。

| 缓冲 | 最大维度 | 说明/写者→读者 |
|---|---|---|
| `sym.x` | `[max_tokens][H]` uint8/FP8 | 上层写，本 rank dispatch local pull 读；peer 也可能 direct read for dispatch |
| `sym.topk_idx` | `[max_tokens][kTopk]` int64 | 上层/ROUTE 模型读 |
| `sym.topk_weights` | `[max_tokens][kTopk]` float | DISPATCH 写 weight staging，REDUCE 读 |
| `sym.route_slot_counts` | `[R*kEPR]` int32 | ROUTE_BUILD 写，PUBLISH_COUNTS 读 |
| `sym.prefix_tbl` | `[kEPR][R+1]` int32 | TASK_SETUP 写，DISPATCH 读 |
| `sym.expert_tokens` | `[EPR*kRing][win][H]` | dispatch 填，L1 读；`pool_ptr` 可替换该基址 |
| `sym.token_route_weights` | `[max(EPR,kGrid)][win]` float | dispatch 写，COMBINE 读 |
| `sym.token_src_meta` | `[max(EPR,kGrid)][win][3]` int | dispatch 写，COMBINE 读 |
| `sym.ffn_hidden` | sym generic layout history | 当前 `check/Dev` 另行分配到 `d.ffn`；保留偏移兼容 |

## 4. shared memory / registers（每个 block 私有）

| 缓冲 | 最大维度/公式 | 说明 |
|---|---|---|
| `s_recv` / `s_prefix` / `s_cum` | `R*recv_stride * 4`、`kEPR*(R+1)*4`、`kEPR+1` int | TASK_SETUP/DISPATCH 解码临时 |
| `s_ab` | `union { A[2][R24_MT*R24_KS*512], B[max(CM... )*8*1024] }` | A 双缓冲/B 异步 BSM 缓冲折叠复用 |
| task decode scalar context | `R19Sched/TaskDesc` 常量 | leader block/task context |
| `a_smem/b_smem` | `R24_MT`/`R24_KS`/`CM_*` 宏决定 | A/B fragment 流水位；kernel-wide union 复用 |
| thread registers | 上限通常 256 regs | L1/L2/A fragment ring |

动态 shared 总量由 `prod_smem(R,window,block_n)` 估算：
`max(window*block_n*4, R*kEPR*4) + control region`，并按 16B/128B 对齐。

## 5. 重要关系

- FG 只标注跨 rank 会访问的 `sym` peer 区和 `ranks_flags`。
- `local_flags`、`ffn_hidden`、`l2_out`、`frag*`、`y` 等在当前实现是本 rank 私有 HBM，不用于跨 rank 数据发布。
- `win_cap` 影响 `expert_tokens/ffn/l2/staging/frag` 的有效窗口，`slot_capacity/max_tokens` 影响 `src_index_table`、`combine_rows`、输出容量。当前 guard 要求 `0 < win_cap <= slot_capacity`，且多数测试路径使用 `win_cap == slot_capacity` 以避免窗口索引错位。
