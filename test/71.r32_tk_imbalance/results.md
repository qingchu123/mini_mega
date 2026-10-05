# test/71.r32_tk_imbalance — 真实 TK_INFO 不均衡路由性能

- 日期：2026-10-05；机器：本机 202（C600UL，32 卡）
- 数据：`test/70.r32_tpr_perf/0928_tk_info.log`（MXMOEMega `DG_TK_INFO_EVERY` 抽样，
  `[TK_INFO] rank=r launch=n : c0..c7`，`ci` = 该 rank 物理专家 i **收到**的 pair 数）
- 二进制：`./bench` + `./megakernel.so`；口径：单进程 `run_aligned`、均衡/不均衡各建
  一张 graph（每节点 = `mega_kernel + REDUCE`），`nodes=20`、`replays=2`（取 min）、
  跨 rank 取 min；`us/kernel` = 单次 launch 全链耗时（不做正确性校验）。
- 约束：单次 `timeout -k 3 10`。脚本：`run.sh`；原始输出：`run.log`。

## 测例构造

`--tk-info` 读取日志中某次 launch 的 32×8 收行数，把「全局专家 e=r*8+le 需
count[r][le] 个 pair」展开成 pool，按 `j % R` round-robin 分给 32 个源 rank
（每源恰好 `tokens*topk` 个，构造后**回跑自检**逐专家等于日志值）。日志中
`launch` 总 pair 恒为 4096 ⇒ tokens=16（与生产 TPR=16 一致）。

| launch | 单专家最大收行 | 非零专家 | 说明 |
|---:|---:|---:|:--|
| 804400 | 176 | 199/256 | 轻不均衡 |
| 804300 | 362 | 199/256 | 中不均衡 |
| 853000 | 512 | 178/256 | 最不均衡（`--tk-info` 缺省自动选此 launch） |

## 结果（graph 链 full-chain us/kernel）

| 路由 | TPR | 单专家最大收行 | us/kernel | 相对均衡 | graph 总(20 节点) |
|:--|---:|---:|---:|---:|---:|
| 均衡（基线） | 16 | 16 | **354.57** | 1.00× | 7.09 ms |
| TK_INFO launch=804400 | 16 | 176 | **1809.95** | 5.10× | 36.20 ms |
| TK_INFO launch=804300 | 16 | 362 | **3458.19** | 9.75× | 69.16 ms |
| TK_INFO launch=853000 | 16 | 512 | **4716.84** | 13.30× | 94.34 ms |

> 结论：真实不均衡把单次全链从 354.6µs 拉到 1.81/3.46/4.72ms（5.1/9.75/13.3×），
> 且基本随「单专家最大收行」线性放大——瓶颈是单个 rank/专家侧的长尾负载
> （均衡 16 行 → 最坏 512 行）。此即不均衡测例下 graph 中的整体性能。

## 复现

```bash
bash test/71.r32_tk_imbalance/run.sh
# 或单跑：
./bench --tk-info test/70.r32_tpr_perf/0928_tk_info.log --tk-launch 853000
./bench --tprs 16          # 均衡基线
```
