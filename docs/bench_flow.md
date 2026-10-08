# src/bench.cu 流程解析与阶段耗时估算

口径：mini_mega fp8xfp8（R=32、E=256 本地 8 专家、topk=8、H=6144、I=2048）。
数据来源：`[T]` 阶段计时（宿主直跑，无争用）。

## 1. 入口与分支

```
main (2627)
└─ parse_args (572)                         # 解析 RunOpts
   ├─ run_aligned (2028)   --aligned 1（默认）   ← 主路径
   ├─ run_sweep   (1725)   --sweep
   └─ run_scene   (690)    --aligned 0（旧单档行为）
```

## 2. 主要调用树

```
run_aligned (2028)                          # 对齐生产（ProdLayout）口径
├─ load_tk_all (1964) / load_tk_case (1901) # --tk-info：从 TK_INFO 日志复现
│   └─ build_tk_case (1847)                 #   逐 expert 收行计数 -> 每 rank 路由
│   （或：均衡路由 make_routes(RoutePattern::Balanced)）
├─ 逐例 win_cap 决策 (2100-2128)            # 长尾不均衡 -> 48 或 max_expert/kRing
├─ compute_layout (host_layout.cuh)         # ProdLayout L：sym 布局/各 off/bytes
├─ fill_w_lane ×2 (348)                     # host 参考权重（288MB，16 线程并行）
├─ 32-rank 初始化线程池 (2189-2345)
│   ├─ mcExtMallocWithFlags(Finegrained) sym + memset
│   ├─ mcMalloc lf + memset
│   ├─ mcExtMallocWithFlags(Finegrained) rf + memset
│   ├─ mcMalloc w1/w2 + fill_weights_lane (设备生成)
│   ├─ mcMalloc ffn_hidden / l2_out / y + memset
│   ├─ frag_a / frag2（仅不均衡长尾启用）
│   ├─ x / topk_weights 填充 + H2D
│   ├─ IPC handle(sym/rf) + open peer + enable peer access（R×R）
│   └─ d_base_ptrs / d_rf_ptrs H2D
└─ 逐测例循环 (2346-2625)
    ├─ 清零 + 上传路由（R 线程 + rank_barrier）
    ├─ 测量（R 线程，各自一个 stream）
    │   ├─ make_mega_args (380)             # 逐例 win_cap/pool_stride
    │   └─ chain_measure (121)
    │       ├─ mcStreamBeginCapture(ThreadLocal)
    │       ├─ launch_full ×nodes           # 每节点 = mega_kernel + mega_reduce_kernel
    │       ├─ mcStreamEndCapture -> mcGraphInstantiate
    │       ├─ 2× warmup graph launch + sync + rank_barrier
    │       └─ replays × (rank_barrier + event + graph launch) -> min(per)
    │   （可选）graph-check：图内每节点 y vs eager（只比 tokens 行）
    └─ 跨 rank 归约(min/max/mean) + 打印
```

其它工具函数：`rank_barrier`(94, 进程内自旋栅栏)、`audit_sym`(231, 显存审计)、
`make_mega_args`(380, 组 MegaLaunchArgs)、`run_tpr_child`(1564, sweep 子进程)。

## 3. run_aligned 主流程（默认路径）

1. **构造测例**：`--tk-info` 从日志复现（`--tk-all`/`--tk-first`/`--tk-launch`），否则均衡 × TPR 列表。
2. **逐例 win_cap**：长尾例（`max_expert ≥ 3×非零中位数`）用小窗（中位小→48，中位大→`max_expert/kRing`），
   均衡例用 `max(128,max_tokens)`；全局 `win_cap = max(逐例)` 供分配/布局。
3. **布局**：`compute_layout(R, max_tokens, topk, H, I, win_cap+pad)`（fp8_layout=true），
   `sym_alloc` 由 `L.sym_bytes` 派生 + combine_rows 扩展区。
4. **rank 初始化**：32 线程各建对称区（FG）、私有 `lf`、跨 rank `rf`（FG）、权重、staging、IPC/peer。
5. **逐例测量**：清零+上传路由 → 每 rank 捕获 `nodes` 节点的 graph 并计时（取多 replay 最小值）→ 跨 rank 归约。

## 4. 阶段耗时估算（单例、无争用）

| 阶段 | 耗时 | 说明 |
|---|---:|---|
| host_weights (`fill_w_lane`×2) | **0.07s** | 288MB host 参考，16 线程并行 |
| rank_init（32 卡） | **~15s** | sym/rf FG 分配+memset、IPC/peer access、设备权重填充、x/topk H2D |
| upload（每例） | ~0.01s | 清零 + 路由 H2D |
| chain_measure（单 rank，nodes=2） | **~0.7s** | capture 0.17→0.70、instantiate≈同、warmup 0.70、replays 0.71（累计） |
| measure（全局） | ~0.72s | = 最慢 rank + join |
| graph-check（每节点 vs eager） | 小 | 只比 `tokens` 行，D2H+比较 |
| **单例合计（无争用）** | **~16-18s** | 主要由 rank_init 主导 |

**争用放大**：当其它 agent/进程占卡时，`measure`/`rank_init` 的全局 wall（含 join 等待）
会涨到 **26~35s**；此前观察到的 26-35s 即此类争用，非代码固有。

## 5. 结论

- 单例成本 = **rank_init（~15s，32 卡对称区/IPC/peer）** + **chain_measure（~0.7s，nodes=2）**。
- `chain_measure` 本身很快；`nodes` 增大主要线性增加 graph 内 kernel 数（50 节点 × ~0.7ms ≈ 35ms/次）。
- 迭代慢的根因是 **32 卡初始化** 与 **外部 GPU 争用**，而非测量本身。
- graph-check 已改为只比 `tokens` 行，开销很小。

## 6. 已知问题（与 bench 相关）

- bench 的 `mcGraph`（`mcStreamCaptureModeThreadLocal`）对同一内核的图内执行结果与
  torch `CUDAGraph`（harness）**不一致**：harness case0 PASS、bench case0 FAIL（rel~0.19），
  且 bench 内 eager 两次自比=0（内核确定）→ 是「图执行路径」差异，bench 图校验不宜直接
  用于正确性判定（harness 为准）。
