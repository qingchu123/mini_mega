# Mega Kernel 缓冲区、复用与竞争分析

## 范围与结论

本文分析当前 `fp8xfp8` 实现：`src/megakernel.cu` 发射的 `mega_kernel`、独立的 `mega_reduce_kernel`，及 Torch 入口 `torch/csrc/apis/mega.hpp`。下文的“轮”指同一组对称区/旗区上的一次 `launch_compute`（通常随后同流执行 `launch_reduce`），不是窗口 `w`；窗口复用另行说明。布局尺寸以 `include/deep_gemm/layout/mega_moe.cuh`、`torch/csrc/apis/layout.hpp` 为准，`cap` 是每源 rank 的 token 容量，`win` 是本轮窗口宽度，`H=6144`、`I=2048`、`EPR=8`、`ring=8`、`grid=32` 是当前内核口径。

**结论**：单轮内按 rank、专家、窗口、slice 或 block 划分写入，并用旗标建立大部分读写顺序；但**下一轮的 ROUTE/PUBLISH 在本轮 xbar 之前运行**，快 rank 可以在慢 rank 仍消费上一轮时写入其路由表/recv 行。若所有 rank 均以同流 `mega → reduce → 下一 mega` 且每轮执行 xbar，下一轮的 DISPATCH/COMBINE 则须等待全部 rank 进入下一轮 xbar，因此数据覆盖风险不能一概而论；`skip_prologue=2` 跳过该跨 rank 门后保障更弱。轮转旗标只隔离旗槽，不能直接隔离未轮转的数据区。此外，源码中归约等待槽与粗化写槽不一致、非融合推送缺少整行发布栅栏等风险仍需重点核实。

## 1. 内存清单（按实际读写，而非布局中所有历史字段）

| 缓冲区 | 位置/容量与生产者 → 消费者 | 轮内/轮间复用 |
| --- | --- | --- |
| `x`、`topk_idx`、`topk_weights` | 每 rank FG 对称区的输入视图：`[cap,H]` FP8、`[cap,topk]` int64/float；调用方写；本 rank ROUTE 读路由，peer DISPATCH 读 `x/weights`，本 rank REDUCE 读路由及权重 | 固定地址跨轮复用，必须保持至所有 peer 拉取、归约结束 |
| `route_slot_counts` | 本 rank `sym` scratch `[R*EPR]` int；block0 ROUTE 在共享内存计数后写本地计数，PUBLISH 读 | 完整前序每轮覆盖；跳过前序时沿用首节点值 |
| `src_index_table` | 目标 rank `sym` scratch `[EPR,R,cap]` uint32；源 rank ROUTE 按 `(目标专家,源 rank,slot)` 远端写，本 rank DISPATCH 读 | 每轮只覆盖有效槽；有效范围由本轮计数/前缀限定；跳过前序时复用旧表 |
| `expert_recv_counts` | 目标 rank `sym` scratch `[R,recv_stride]` uint32，当前 `recv_stride=12`；源 rank PUBLISH 写前 8 列及 checksum，本 rank TASK_SETUP 校验读 | 完整前序每轮原址改写，checksum 使用固定 `publish_epoch=1`，本身不区分轮号；跳过前序时复用 |
| `prefix_tbl` | 本 rank `sym` scratch `[EPR,R+1]` int，或可选独立 `prefix_ptr`；block0 TASK_SETUP 根据 recv 建表，所有 block DISPATCH 读 | 完整前序重建；`skip_prologue` 时保持不变 |
| `expert_tokens` / `pool_ptr` | 默认本 rank `sym` scratch，或单独普通显存；内核按 FP8 `[EPR,ring,win,H]`（以 `pool_stride` 寻址），通用 scratch 布局为它预留 BF16 字节容量；DISPATCH 从 peer 输入拉取各 K 片，L1 读 | `(专家,w%ring)` 复用，`l2_done` 门防同轮跨窗覆盖 |
| `token_route_weights` / `token_src_meta` | 本 rank `sym` scratch 的每 block staging（分别 float 与 3 个 int/行）；DISPATCH 写、COMBINE 读 | 按 `blockIdx*win+row`，同 block 顺序处理下一任务时复用；并非按专家 ring 索引 |
| `ffn_hidden` | 本 rank 普通显存 `[EPR,ring,win,I]` FP8；L1 写、L2 读 | `(专家,w%ring)` 复用；单窗快路径还会借用本专家未用的 ring 槽保存 A/F2 fragment |
| `frag_a` / `frag2` | 可选普通显存，每种 `[EPR*ring]` 槽，单槽大小分别为 `(H/32)*ceil(win/16)*512`、`(I/32)*ceil(win/16)*512` 字节；DISPATCH/L1 写，L1/L2 读 | 多窗口按 `(专家,w%ring)` 复用；未传入时走其它路径 |
| `l2_out` | 本 rank 普通显存 `[grid,win,H]` BF16；L2 写，非融合 COMBINE 读；融合推送从写后的列块推给 peer | **每 block** 一槽，顺序任务复用；不随专家/ring 寻址 |
| `combine_rows` | 目标（源 token 所在）rank FG 对称区 `[topk,cap,H]` BF16；专家 rank COMBINE 分列写，源 rank REDUCE 读 | 不轮转；同一 `(topk,token)` 每轮覆写，有效行按路由决定 |
| `y` | REDUCE 输出 `[tokens,H]` FP32；Torch 层随后转 BF16 写用户输出 | 调用方分配，Torch `SymmBuffer._y_fp32` 跨轮复用；测试可按节点分配 `ynode` |
| `w1/w2`、`peer_bases/rf_peers` | 本 rank 的只读打包权重；两个设备指针表各有 `R` 个 peer 地址 | 跨轮共享；必须保持内容、映射及分配存活，不与异步 kernel 并发销毁/改写 |
| 动态/静态共享内存、寄存器 | ROUTE 的专家计数、TASK_SETUP 的 recv/窗口表、DISPATCH 的前缀/任务号；`kernel.cuh` 的 `s_ab` 联合体将 A 双缓冲和 B 异步缓冲复用；GEMM 的寄存器预取环 | **每 block/每次 kernel 私有**；相位与任务之间重用，不是跨 rank 的持久缓冲 |

`sym` 还按通用布局保留 `barrier_slots`、`combine_sums`、`ffn_out` 等位置；**当前调用链**跨 rank barrier/完成标记已使用 `ranks_flags`，L2 输出使用独立 `l2_out`，不要把布局中的历史字段当作当前生产者。实际偏移见 `torch/csrc/apis/layout.hpp:73-113`，任务索引见 `src/compute.cuh:907-916,964-1005,1055-1134` 与 `src/compute_big.cuh:1079-1213`。

### 旗区及其寿命

| 缓冲区 | 归属与用途 | 复用机制 |
| --- | --- | --- |
| `local_flags.current_time[32][64]` | 每 rank 普通显存，每 block 自己的 `[0]` xbar、`[1]` task_setup 计数 | 每轮各加 1，graph replay 仍实际推进；初值需清零 |
| `local_flags.bar[8][1032]` | 本 rank 普通显存：`xrelease`、`task_ready`、`task_counter`、`l2_done`、`slice_flags`、`l1_prog`、`nslice_eff` 等 | 本轮 `c=current_time%8`；入口预清 `(c+2)%8`，任务开始重置当前 `l2_done`，其余旗按轮号/窗口代际或每轮写入 |
| `local_flags` 诊断区 | `out_total/out_processed`、`dbg/ck_dbg`、`pre_ticks`、`l1_chk/l2_chk` | 固定地址跨轮写/累加；不能把未清零的诊断值当成单轮结果 |
| `ranks_flags.xbar_barrier[8][32][32]` | 每 rank FG 显存，peer 写入目标 rank 的 `[c][src][dst]`，该目标 rank 的 block0 等待 | 同 `bar` 轮转、预清；epoch = 当前 launch 计数 + 1 |
| `ranks_flags.combine_sums[8][8][512][32]` | 每 rank FG 显存，专家 rank 按源 `(topk,token,slice)` 写完成 epoch；源 rank REDUCE 等待 | 同周期 `c` 轮转但**不清零**，读侧要求标记 `>= epoch`；必须保证 epoch 不回绕误判 |

分配：Torch 层 `sym` 用 fine-grained 并通过 IPC 交换，`lf` 用普通显存，`rf` 用 FG；`ffn_hidden/l2_out/frag` 与两份设备指针表用普通显存（`torch/mxmegakernel/mega/__init__.py:213-256`、`torch/csrc/apis/mega.hpp:83-145`）。`rf` 的分配函数在 FG 失败时回退普通 `mcMalloc`，注释标为**仅诊断**；peer 可写性不能据此保证（`torch/csrc/flags_bridge.cu:23-34`）。`pool_ptr` 仅作为可选参数传入，Torch 当前 `make_mega_args` 不设置它；`prefix_ptr` 同理（`torch/csrc/apis/mega.hpp:190-225`）。

## 2. 一轮执行的发布、消费与槽复用

1. `ROUTE_BUILD`：block0 将各源的 `(token,topk)` 编码写入目标 `src_index_table`，本地计数写 `route_slot_counts`。`PUBLISH_COUNTS` 将计数和 checksum 写入各目标的 `expert_recv_counts`。`XBAR_ROUTE_READY` 在 `rf` 中等待所有 rank 的发布，再写本地 `bar[c].xrelease` 放行其它 block（`src/prologue.cuh:105-235`、`src/flags.cuh:267-359`）。
2. `TASK_SETUP`：block0 校验 recv 行，生成 prefix，清本轮 `l2_done`、设置任务计数，发布 `task_ready=epoch`；其它 block 等待后读 prefix。`skip_prologue=1` 保留 xbar 而复用路由/prefix；`=2` 连 xbar 也跳过，仅推进其计数（`src/kernel.cuh:193-236`、`src/prologue.cuh:247-527`）。
3. 任务号按窗口和物理 slice 分配，同窗不同 slice 负责互不重叠的 K/N 部分；pull 后写 `slice_flags`，L1 等齐本窗 slice，发布 `l1_prog`，L2 等齐后计算。`gen=epoch+w/ring` 防同轮同一槽前代的 slice/progress 旗被误认为新代（`src/compute_big.cuh:1087-1094,1213-1297`）。
4. 下一次使用同一专家槽 `w+ring` 的任务先等本槽 `l2_done >= (w/ring)*nslice_eff`；每个物理任务在 COMBINE 后递增该计数。此门保护 pool、ffn 和相同地址的 fragment，**不是跨轮全局栅栏**；`l2_out`、weight/meta 则因同 block 顺序任务而复用（`src/flags.cuh:215-244`、`src/compute.cuh:1055-1134`）。
5. COMBINE 按 `meta` 将 L2 的 BF16 列片写入源 rank 的 `combine_rows`，并写 `rf.combine_sums[c]`；同一 stream 随后的 REDUCE 由本 rank `current_time[0][1]-1` 选轮转片并等待标记，然后读行、写 `y`（`src/kernel.cuh:239-295`、`src/combine.cuh:381-580`、`src/megakernel.cu:190-200`）。

## 3. 竞争风险矩阵

| 范围/触发条件 | 当前保护与判断 | 风险及处理条件 |
| --- | --- | --- |
| 同轮不同 rank 写同一目标 `src_index_table`、recv | 表按源 rank 分行；recv 也按源 rank 分行；xbar + checksum 把发布与读取隔开 | **按设计不冲突**，前提是 peer FG、索引/容量合法，所有 rank 参与同一次 xbar。ROUTE 超过 `cap` 会丢条目，不是缓冲竞争但会丢路由 |
| 同轮不同 rank 写 `combine_rows`/`combine_sums` | 不同 `(源 rank,token,topk,slice 列)` 分工；完成标记按 slice；REDUCE 等标记 | 写范围正确时不应互踩；对不同 rank 的完成先后没有统一完成门。`topk` 重复目标并不会自动导致同位置冲突，因为 `topk` 槽由 `encoded%topk` 决定 |
| 同轮同专家窗口 `w` 与 `w+ring` | `l2_done` 计数门 + `slice_flags/l1_prog` 的窗口代际 | **有协议保护，依赖每个 slice 完成并发布**。等待循环超过 `max_spins` 只记 `dbg` 后继续执行，超时不是安全终止，失败轮的数据不应复用；`NO_GEN_FIX` 等诊断编译选项会改变保障 |
| 跨轮本 rank、同一 stream | 同流 `mega → reduce → 下一 mega` 的发射顺序；`bar/rf` 轮转和设备计数 | **本 rank 顺序发射时通常不重叠**。若同一 `SymmBuffer` 在不同 stream 并发发射，`current_time`、任务计数、诊断区、pool、ffn、`l2_out`、`y` 均无独立实例/跨流依赖，不能并发复用 |
| 跨轮不同 rank（快 rank 先进入下一轮） | 下一轮 ROUTE/PUBLISH 在 xbar **之前** 写对方 sym；若每轮都执行 xbar，DISPATCH/COMBINE 在 xbar **之后**。本 rank 同流 `mega → reduce` 完成后才进入下一轮 xbar | **前序写仍存在竞争窗口**：快 rank 下一轮 ROUTE 可改写慢 rank 上一轮仍在 DISPATCH 读的 `src_index_table`；下一轮 recv 行也可与慢 rank 上一轮 TASK_SETUP 交错，且 checksum 固定为 1。调用方过早改写本 rank `x/topk/weights` 还会与慢 rank 的 peer DISPATCH 读交错。正常同流且每轮 xbar 时，下一轮 COMBINE 只能在所有 rank 的上一轮 REDUCE 已完成并进入下一轮 xbar 后运行，**这一特定覆盖受到保护**；`skip_prologue=2` 取消 xbar 后不再有此保证。须在下一轮 ROUTE/输入更新前增加跨 rank 消费完成门，或把路由表/输入数据按轮隔离 |
| rank 漏轮、乱序、复用/非复用前序混用 | `current_time` 本地推导 `c/epoch`；xbar 用该值等待 | **协议前提不满足**：必须所有 rank 同序参与，特别是 `skip_prologue=2` 没有每节点 xbar。快 rank 若领先到慢 rank 正在使用的 `(c+2)%8` 槽，预清还可能覆盖 peer 正写的 xbar 标记。超时分支继续运行，不能当作恢复机制；重新建档/清零必须在所有用户停止访问后统一进行 |
| `skip_prologue` 跨节点/graph replay | 首节点构建路由与 prefix，随后节点复用；Python opt-in `prologue_reuse` 在 capture 的非首节点设为 2 | **仅路由及输入视图在整个复用区间不变时成立**。跨 graph replay 首节点仍需正常前序；相邻 graph 或不同 rank 节点顺序不一致时不能借用旧 prefix。`skip_prologue=2` 更没有跨 rank 本节点门（`torch/mxmegakernel/mega/__init__.py:378-389`） |
| 缩放/粗化后 REDUCE 等待槽 | COMBINE 为一个物理 slice `s` 写逻辑槽 `[s*merge, (s+1)*merge)`；REDUCE 只等 `[0,nslice_eff)` | **疑似实际提前放行**：如 `kNSlice=4, merge=2, nslice_eff=2`，物理 slice 0 写槽 0、1，slice 1 写 2、3；REDUCE 只等 0、1，就可能读到 slice 1 尚未写完的列。应按逻辑槽全数等待，或改成与物理 slice 一一对应的标记；需用强制 `merge>1` 的跨 rank 测试验证（`src/combine.cuh:398-408,453-461,521-545`） |
| 非融合 COMBINE 的跨 rank 数据可见性 | `combine_push_pair` 中多 lane 写 `combine_rows`，同一行 lane0 紧接着写完成标记；REDUCE 等标记后做 fence | **发布顺序未闭合**：该路径未见标记之前的全 lane fence + block/warp 汇合，不能从当前代码证明 payload 均先于标记可见；融合 `combine_mark_rows` 有 `__threadfence(); __syncthreads()`。若非融合路径会执行，需补足发布协议并验证 peer 可见性（`src/combine.cuh:381-410,422-462`） |
| 计数/标记跨极长 graph 的回绕 | `current_time` 64 位，完成标记与 `gen` 截成 32 位；`combine_sums` 不预清 | **边界风险**：`epoch`/`gen` 回绕、旧值大于新值时 `>=` 可误放行；需定义可支持的最大轮数及重建/清零时机，避免有活动 peer 时清零 |

另外，`src/megakernel.cu:70-80` 明确要求 `slot_capacity == win_cap`，不相等则 `std::exit(2)`；`torch/csrc/apis/mega.hpp:169-185` 却允许 `win_cap < b.cap`，Python 的自适应窗口也会选较小值。更值得注意的是 Python 构造 `SymmBuffer` 时先把 `cap` 向上对齐到 384，而窗口限制为 ≤128（`torch/mxmegakernel/mega/__init__.py:31-34,165-190`），按**当前源码**经此 Torch 入口发射始终不满足等式；不仅是自适应场景。这个入口问题与缓冲竞争是两件事，应先单独修复/核实部署版本，再评估动态窗口复用。测试辅助文件同样注明 `win<slot` 的 REDUCE/graph 问题（`test/mega_common.cuh:79-86`）。

## 4. 最小验证建议

1. 使用不同 rank 的多轮异步 graph，在快 rank 下一轮 ROUTE/COMBINE 前故意延迟慢 rank 的 DISPATCH/REDUCE，并对比逐轮独立缓冲结果；重点记录 `current_time`、xbar/REDUCE 超时和 `combine_rows` 的代际。不要在测试中无限等待，超时即失败。
2. 强制 `merge>1` 且有效路由覆盖所有 topk/slice，分别运行融合与非融合 L2 路径，对 REDUCE 完成标记逐槽检查；超时/数值不符立即终止，不将诊断 `dbg` 当作通过。
3. 对多 stream/跨 rank 复用同一实例建立显式的缓冲区所有权与完成事件；仅当全部 peer 消费完前轮数据才允许覆盖 `sym` 输入/路由/归约行，释放 IPC/FG/权重也遵守同一完成边界。

本文件是静态源码审查，不把上述疑似风险表述为已复现的运行故障。
