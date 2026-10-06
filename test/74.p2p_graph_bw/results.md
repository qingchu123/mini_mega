# test/74.p2p_graph_bw — GPU 间 P2P 单 block 远写/远读带宽（graph 口径）

- 日期：2026-10-06；机器：本机 202（MetaX C600-UL ×32，32 SM/卡）
- 口径：每个 (pair,size) 建一张 **CUDA graph，含 10 个同构节点**；每节点 = **1 个 block × 1024 线程**；
  预热 2、replay 3 取 min，event 夹住，per-node = 总时间/10，带宽 = size / per-node。
- 显存：**FG**（`mcExtMallocWithFlags(..., mcDeviceMallocFinegrained)`）+ `mcIpcGetMemHandle` /
  `mcIpcOpenMemHandle` + `mcDeviceEnablePeerAccess`。
- 粒度：6KB, 12KB, 24KB, 48KB, 96KB, 192KB, 384KB, 768KB。
- 并发：写测试用卡 0–15、读测试用卡 16–31（并行跑，互不抢卡）。

## 构建与运行

```bash
bash build_write.sh && bash run_write.sh      # 远写（卡 0-15）
bash build_read.sh  && bash run_read.sh       # 远读（卡 16-31）
```

## 远写带宽（GB/s，`__stwt`/plain 取快者；0→1 / 0→8 / 0→15 基本一致）

| size | 6K | 12K | 24K | 48K | 96K | 192K | 384K | 768K |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| GB/s | 2.2 | 4.3 | 8.3 | 14.6 | 24.5 | 36.5 | 48.2 | **59.1** |

- 峰值 **≈59.7 GB/s @768KB**；随粒度单调上升，未出现平台。
- `__stwt` 与普通 store 差异 <2%（噪声内）。
- 扣除 ~2.6µs/节点固定开销后，边际带宽 ≈ **72–74 GB/s**（单 block 未打满链路）。
- 距离（0→1/8/15）无显著差异。

## 远读带宽（GB/s，vec128；reader=16）

| size | 6K | 12K | 24K | 48K | 96K | 192K | 384K | 768K |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 16→17 | 1.6 | 3.0 | 4.4 | 6.8 | 8.4 | 9.6 | 10.3 | **10.8** |
| 16→24 | — | — | — | — | — | — | — | 8.4 |
| 16→31 | — | — | — | — | — | — | — | 8.5 |

- 峰值 **≈10.8 GB/s @768KB（相邻 16→17）**；跨节点 ~8.4 GB/s。
- 128-bit 向量化 load 比 32-bit 快 ~3.7×。
- 扣除 ~3–4µs/节点开销后，边际带宽 ≈ **11 GB/s**（单 block 受 MLP/延迟限制）。

## 结论

- **单 block 远写（~59.7 GB/s）≈ 远读（~10.8 GB/s）的 5.5×**（768KB）；小粒度差距更大
  （6KB：2.26 vs 1.55）。这印证了「把通信改成远写」的动机：本平台远写带宽显著高于远读。
- 固定开销：每 graph 节点 ~2.6–4µs（空 block 基线 ~2.6µs）。
- 校验：远写 24/24 PASS、远读全 PASS；peer 全互通（can/open/enable 15/15）。

## 复现

```bash
bash test/74.p2p_graph_bw/build_write.sh && bash test/74.p2p_graph_bw/run_write.sh
bash test/74.p2p_graph_bw/build_read.sh  && bash test/74.p2p_graph_bw/run_read.sh
```
