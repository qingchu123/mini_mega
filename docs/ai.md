# mini_mega AI 协作实录（关键信息汇总）

时间范围：2026-10-06 ~ 10-07。口径：mini_mega（GLM-5.2 FP8，R=32 / E=256 本地 8 专家 /
topk=8 / H=6144 / I=2048 / tokens=16/rank）。

## 1. 性能口径与工具

- 内核基准：`./bench --tk-info test/70.r32_tpr_perf/0928_tk_info.log --tk-first 50`
  取输出第 6 列 `us/kernel` 平均。口径为**跨 rank 取 min**；因图内 REDUCE 每节点同步
  全体 rank，min ≈ 最慢 rank 的关键路径。
- 新增 `--tk-all` / `--tk-first N` / `--tk-launch N`（单进程扫描 TK_INFO 日志）。
- 真实负载（mxdbg_s.log）：tokens=16/rank、topk=8、NEXTN accept≈0.83、**生产未开
  CUDA graph**；测试脚本解码态为 `--cuda-graph`。

## 2. 已提交的关键改动（main）

| commit | 内容 | 效果 |
|---|---|---|
| `49d1d34` | bench `--tk-first N` | 快速前 N 例口径 |
| `eb8774b` | 逐例自适应 win_cap | 898.5→855.4µs |
| `cce2fdf` | 中等档(33–48) r24 A-smem | 再 −2.3%（→835.8） |
| `2b96a0c` | kCap64：删除不可达 MT8 大档 | 835.8→809.9 |
| `c0020d8` | kCap 下探 48：裁 MT4 兜底 | 810.7→776.1 |
| `bde6343` | 跨 graph 节点前序复用 + tk 专属 B 预取（bench 专用） | 776.8→743 |
| `a6f6043` | **M_CYCLE 4→8**（修 case1 图内竞态） | 正确性修复，性能中性 |
| `298e955` | **补 `__launch_bounds__`** + bench `--graph-check` 等 | 消除 MACA 图内重编译 |

## 3. MACA「图内重编译」问题（重要）

现象：宿主机 `./bench` 每次跑都被 MACA 运行时对图内核**在线重编译**
（`StatRecompileKernel ... Recompile kernel in graph kernel, kernel_name:_Z11mega_kernel...`），
慢且默认编译器报 `invalid target ID: xcore1610`。

触发条件（用户确认）：
1. **kernel 未写 `__launch_bounds__`** ← 本次命中：`src/megakernel.cu` 的
   `mxk_dbg_capture_kernel` 缺 bounds，导致**整个模块**进图时重编译。补
   `__launch_bounds__(64)` 后宿主直跑 **0 次重编译**。
2. **对 FG（finegrained）显存做 atomic**。本仓库 FG 区为 `sym`（路由/combine_rows 等）与
   `rf`（ranks_flags）；`lf`、`y`、`w1/w2` 为普通 `mcMalloc`。已核对：现有 atomic 均打在
   `lf`（bar/task_counter/dbg/l1_chk/l2_chk/pre_ticks）或 shared（`slots`/`smem_ctr`）上，
   **未发现打在 FG 上的 atomic**。

环境注意：
- 宿主机 `/opt/maca` 被改到 `maca-3.8.4`（不支持 xcore1610，12:20 变更）。
  恢复：`sudo ln -sfn /opt/maca-20260926 /opt/maca`。
- 恢复前宿主构建/运行需 `MACA_PATH=/opt/maca-20260926`（并同步 PATH/LD_LIBRARY_PATH）。
- `cl_test` 容器用自带 `/opt/maca -> maca-20260909`，构建/运行一致、缓存命中、不重编译。

## 4. C++ bench 复现 harness 测试（脱离 python）

新增选项（`src/bench.cu`）：
- `--graph-check 1`：图内**每节点输出 vs eager** 逐例对拍，打印
  `graph-check: case=<launch> nodes=<n> rel_max_diff=<d> <PASS|FAIL>`；`>1e-3` 判 FAIL 并
  返回退出码 3。
- `--graph-reps N` = `--nodes N`；`--routing-fixed-cases all|<id>` = `--tk-all` / `--tk-launch`。

复现（宿主，无 python）：
```
export MACA_PATH=/opt/maca-20260926   # 恢复软链后可省略
MACA_VISIBLE_DEVICES=$(seq -s, 0 31) ./bench \
  --tk-info test/70.r32_tpr_perf/0928_tk_info.log \
  --tk-launch 809200 --nodes 50 --graph-check 1
```
注：bench 图默认 `reuse_prologue`（`skip_prologue=2`，节点 1..N 跳过 prologue+xbar），
会引入漂移伪发散；对齐 harness（`prologue_reuse=False`）需 `MX_NO_REUSE=1`。

## 5. case2（809200）图内发散 —— 未解 BUG

现象（M_CYCLE=8 之后）：
- `--tk-launch 809200 --nodes 3..50 --graph-check 1` → `rel_max_diff≈0.19~0.87 FAIL`；
  发散行随机（row=15/row=4…）。g=3 即失败（M_CYCLE 余量充足）。
- **非图特有**：eager 两次自比（且中间加 host `dist.barrier()` 强制串行）仍差
  ≈8700~13040 → 是**单 launch 内生产者/消费者可见性竞态**（同一输入、串行两次结果不同）。

已定位的证据：
- REDUCE 对**从未写入的标记**超时（`part=0 < epoch`），随后照读未写的 `combine_rows`。
- 结构不对称：`combine_sums`（完成标记）按 `[M_CYCLE]` 轮转；`combine_rows`（BF16 数据）
  **单缓冲、不轮转**（layout 注释「每次 launch 全新覆写」）。
- 每 launch 的 xbar 对齐判据为 `(int32_t)(*mine - epoch) >= 0`，**允许快的 rank 领先**，
  故存在跨 launch 漂移（case1 实测 Δ=2：M_CYCLE=4 挂、=5 过）。

已尝试但**未修复**（均回退，未提交）：
- REDUCE 超时行按 `w8[k]=0` 屏蔽 → 超时 56→11，发散仍在。
- `combine_push_pair`/`combine_mark_rows` 加发布序 fence（设备域、system 域）→ 仍在。
- 非融合路径 `combine_push_pair` 行/标记由 `__stwt` 改普通写（`__stwt` 不更新 L2）→ 仍在。
- 固定窗口 `DG_MEGA_WIN_AUTO=0` → 仍在。

修复候选：
- A（推荐）：给 `combine_rows` 加 cycle 维（与 `combine_sums` 一致）——需改
  `include/deep_gemm/layout/mega_moe.cuh`（尺寸/offset）、`src/combine.cuh`（读写行地址
  = `rows_off + cycle*rows_bytes`）、`src/host_layout.cuh`、torch 层 SymBuffer；消费端按
  标记 epoch 选片。
- B：把每 launch 对齐改成严格屏障（drift=0，单缓冲即安全），风险：`>=`→`==` 可能死锁。
- C：`combine_rows` 双缓冲 + 消费端读前握手。

## 6. case1（807500）已修

`M_CYCLE 4→8`（`a6f6043`）：每 launch 入口预清 `(c+2)%M_CYCLE` 槽，安全前提
`Δ < M_CYCLE-2`。极偏斜下 Δ=2，M_CYCLE=4 时 launch L 的清零砸掉仍在飞的 L+2 旗槽 →
发散。8 后 case1 全节点 PASS，性能中性。

## 7. bench 运行耗时分析（graph-check）

分段计时（`[T]` 打印）显示：
- host 参考权重生成：~0.1s。
- 32 rank 分配/初始化：~13~23s（sym/rf finegrained 分配 + IPC + peer access + 设备权重填充）。
- 单例 case loop：~26~49s（**异常大且随机器负载波动**，与节点数/graph-check 非单调）；
  graph-check 仅比节点行数放大拷贝（已改为只比 `tokens` 行，仍偏慢）。
结论：迭代慢主要在「32 卡初始化 + 单例固定开销」；调试建议用 `--nodes 3`（阈值）。

## 8. 其它事实

- 权重尺寸：每本地专家 W1+W2 = H·2I + I·H = 6144×4096 + 2048×6144 = 37,748,736 B ≈ 36MiB；
  ×8 专家 ≈ 302MB(10^6)/288MiB。按 ~1.45TB/s ≈ 208µs ≈ test/76 实测 L1+L2=207µs、
  地板 264µs → 小 token decode 的固定成本是**权重读**。
- 真实负载 tokens=16/rank；`eager` 与 50-in-1 图的差异在 harness（`prologue_reuse=False`，
  每节点跑全 prologue）。
- MXMOEMega harness 位置：`/home/lchen1/data/projs/MXMOEMega/tests/mx_test/test_mega_moe_all.py`；
  用 `DG_TEST_PACKAGE_ROOT=<wheelroot>` 指向解包后的 mini.mega wheel 树。
- wheel 构建：`cd torch && MACA_PATH=/opt/maca python setup.py bdist_wheel`；产物含
  `mxmegakernel/_C*.so` 与 `mxmegakernel/megakernel.so`。
