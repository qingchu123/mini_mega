# mini_mega torch 层（pybind + Python 包）

时间标签 `[261006_12_40]`（未提交 git）。

在 `torch/` 下新增 torch/pybind 层，产出可安装 wheel `mxmegakernel`，把
MegaMoE 的 Python/C++ API 面对齐 sglang megamoe 后端（C600UL fp8xfp8 路径），
并**真实绑定** mini_mega 的 kernel 接口层（`src/megakernel.h` / `megakernel.so`）。
**已在 10.0.82.71 用 GLM-5.2-FP8（tp32/dp32/megamoe）跑通服务 + 出 token。**

## 1. API 契约（sglang megamoe 实际调用）

来源：71 上 `/home/m01182/sglang/python/sglang/srt/layers/moe/mega_moe.py`
（fp8xfp8 分支）与 `.../quantization/fp8.py` 的 `build_mega_moe_experts_weights`
调用点。fp8xfp8 路径使用：

- `get_symm_buffer_for_mega_moe(group, E, cap, topk, H, I, num_shared_experts=0,
  mma_type="fp8xfp8", activation="swiglu", num_tokens=...)`
- `buf.x / buf.topk_idx / buf.topk_weights`
- `transform_weights_for_mega_moe(w13_raw, w2_raw)`
- `fp8_mega_moe(y, l1, l2, buf, activation="swiglu", activation_clamp=...)`
- `mxmegakernel.mega._interleave_weights / _transpose_sf_for_utccp`（fallback import）

## 2. 文件与实现程度

```
torch/
├── setup.py                     # CUDAExtension -> mxmegakernel._C；链接 megakernel.so
├── mxmegakernel/
│   ├── __init__.py              # 顶层导出
│   ├── mega/__init__.py         # 主 API（本文件核心）
│   ├── utils/{__init__,math,layout,dist}.py
│   └── testing/{__init__,numeric,utils,bench}.py
├── csrc/
│   ├── python_api.cpp           # PYBIND11_MODULE(_C)
│   ├── kernel_bridge.cpp        # 真实绑定 fp8mega::launch_empty / fill_weights_lane
│   ├── apis/{layout,mega}.hpp   # 对称区布局 + MegaLaunchArgs 组装 + 发射
│   ├── memory.cpp/.hpp          # FG/IPC/peer access 平台层
│   └── flags_bridge.cu          # local_flags/ranks_flags 分配
├── tests/smoke.py               # 单卡冒烟
├── tests/test_fp8_mega_moe.py   # 单/多 rank 数值自检
└── sglang_patches/              # 71 上 sglang 侧必需的三个补丁（见 §5）
```

| API | 状态 | 说明 |
|---|---|---|
| `get_token_alignment_for_mega_moe` | 已实现 | `kLCMCandidateBlockM`=384 |
| `select_fp8_mega_moe_window` | 已实现 | FP8 AUTO 表 |
| `get_symm_buffer_size_for_mega_moe` | 已实现 | `compute_symm_layout(...).total_bytes` |
| `get_block_m_for_mega_moe` | 已实现 | `kLCMCandidateBlockM` |
| `transform_weights_for_mega_moe` | 已实现 | gate/up 配对 + 16B/lane swizzle（与源项目逐字节等价） |
| `SymmBuffer` / `get_symm_buffer_for_mega_moe` | 已实现 | 多 rank FG + IPC + 建档预热 + 输入视图 |
| `fp8_mega_moe` | 已实现 | `fp8mega::launch_full`，当前 stream 发射 |
| `mega_launch_empty` / `fill_weights_lane` | 已实现 | 直接调用 mini_mega kernel |
| IPC（`get_ipc_handle`/`allocate_finegrained`/`enable_peer_access`…） | 已实现 | mcr 原生 API |
| `bf16_mega_moe` / `bf16_fp8_mega_moe` / `fp8_fp4_mega_moe` | not_implemented | 非 fp8xfp8 路线 |
| `transform_sf_into_required_layout` / `read_route_tables` | not_implemented | fp8xfp4 scale 路径 |

## 3. 关键实现点（对齐源项目 `01.fp8xfp8_moe_kernel`）

1. **权重打包**：`_interleave_weights(gran=256)` + `_swizzle_lane` 与源项目
   `_pair_gate_up_w1_for_prj_mma` / `_pack_fp8_weights_for_prj_mma` 逐字节一致。
2. **预热（capture 安全）**：建档时（`SymmBuffer.__init__`）用均衡路由覆盖全部
   专家的 dummy 输入空跑 2 次，走热跨 rank FG 懒映射；首次真实 launch 若不在
   CUDA graph 捕获内再补 2 次真实输入预热。缺此预热时首档数值错（本地 R=16
   实测 9/16 错 → 预热后全过）。
3. **发射**：`apis/mega.hpp::make_mega_args` 逐字段对齐 `src/bench.cu` 的
   `make_mega_args`（offset/local_flags/ranks_flags/peer_bases/w1/w2），
   `fp8mega::launch_full(a, current_stream)`。

## 4. 构建 wheel

```bash
cd torch && MACA_PATH=/opt/maca python setup.py bdist_wheel
# 产物：torch/dist/mxmegakernel-0.0.0+mini.mega.<stamp>-cp310-cp310-linux_x86_64.whl
```

wheel 内置 `megakernel.so`（`DT_NEEDED=megakernel.so` + rpath `$ORIGIN`）。

## 5. 71 跑通 GLM-5.2-FP8（sglang megamoe）

环境：`ssh 10.0.82.71`，容器 `lyx_test_new5_all`，sglang 走
`PYTHONPATH=/home/m01182/sglang/python`（镜像自带 editable 安装指向不存在的
`/sw_home/...`，需显式 PYTHONPATH）。

服务脚本：`/home/lchen1/decode_megamoe_71.sh`（由本机 megamoe 脚本改造：
`--model=/data3/GLM-5.2-FP8 --tp-size=32 --dp-size=32 --moe-a2a-backend=megamoe
--quantization=fp8 --enable-dp-attention --enable-dp-lm-head`，
`SGLANG_MEGA_MOE_MMA_TYPE=fp8xfp8`，关 EPLB / 关 MTP / 关 PD-disagg）。

**三个 sglang 侧必需补丁**（见 `torch/sglang_patches/`，均已备份原文件）：

1. `patch_mega_moe.py`：`build_mega_moe_experts_weights` 里把原始 block-quantized
   权重**原位接管**（`experts.w13_weight.data = l1_mega`），否则原始 + 变换后两份
   MoE 权重常驻 → KV cache 无空间。
2. `patch_mega_scale.py`（**关键正确性修复**）：原 sglang fp8xfp8 分支把 `x` 与
   `W1` 同时按 per-tensor scale **下缩放**，再把 `act_scale*w13_scale*w2_scale`
   折回输出。但 L1 后接 **SwiGLU 非线性**，缩放不能这样折回，导致输出乱码。
   按源项目 `model_convert.py` 的契约改为：`x/W1` 直接 raw cast（不做下缩放），
   仅 `W2` 用 2 的幂上缩放并把 `topk_weights` 同比例缩小（REDUCE 精确抵消），
   输出不再缩放。
3. `patch_warmup.py`：均衡路由预热（已并入 §3 的 wheel 源码，patch 仅用于现场）。

结果证据：
```
[max_total_num_tokens=148160, available_gpu_mem=8.77 GB]
The server is fired up and ready to roll!
/health = 200
The capital city of France is -> " Paris. It is also the largest city in France."
1+1= -> "2 is a statement. ..."
Hello, my name is -> " Sarah. I am from England. ..."
```

压测（自写并发 `/generate`，脚本 `sglang_patches/` 同级）：
```
requests=48 concurrency=8 elapsed=13.41s
prompt_tokens=1824 output_tokens=1536
output_throughput=114.51 tok/s  total_throughput=250.49 tok/s
```

A/B 对照：同一服务改用 `--moe-a2a-backend=deepep` 也输出连贯文本；未修
`patch_mega_scale.py` 前 megamoe 输出乱码。据此定位乱码根因是 sglang 的缩放，
而非 mini_mega kernel（kernel 在 R=32 真实几何下对随机权重全部 PASS）。
