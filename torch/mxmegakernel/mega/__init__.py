"""MegaMoE Python API 层 —— mini_mega 版。

对外签名对齐 MXMOEMega / sglang megamoe 后端（C600UL fp8xfp8 路径）所需的
API 集：

  get_symm_buffer_for_mega_moe / SymmBuffer  对称区 + 输入视图（多 rank FG/IPC）
  transform_weights_for_mega_moe             权重离线打包（gate/up 配对 + lane swizzle）
  fp8_mega_moe                               融合 kernel 发射（经 fp8mega::launch_full）
  select_fp8_mega_moe_window                 窗口选型

实现程度：
  * 已实现：fp8xfp8 全链（对称区 fine-grained 分配 + IPC + MegaLaunchArgs 组装 +
    真实发射）、权重打包、窗口选型、输入视图。
  * not_implemented：bf16/fp8xfp4 等其它精度路线。
"""

import os
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist

from .. import _C

# ---------------------------------------------------------------- 冻结配置
FROZEN_HIDDEN = 6144
FROZEN_INTERMEDIATE_HIDDEN = 2048
FROZEN_NUM_TOPK = 8
FROZEN_NUM_EXPERTS_PER_RANK = 8

EXPERT_WINDOW_MIN = 8
EXPERT_WINDOW_MAX = 128
EXPERT_WINDOW_MULTIPLE = 8

_SUPPORTED_MMA_TYPES = ("fp8xfp8",)

# L1 gate/up 配对粒度 = kernel 的 l1_block_n（见 make_mega_args）。
_L1_BLOCK_N = 256

# kernel 常量（与 src/common.cuh 对齐）
_KERNEL_RING = 8        # kRing：每专家窗口槽数
_KERNEL_M_TILE = 16     # kM：M tile 行数（win_cap 对齐粒度）
_KERNEL_GRID = 32       # kGrid：grid 块数（缺省 merge 上限）

# 物理窗口容量缺省：与调用方逻辑窗口解耦，按最大值分配（AUTO 表最大 80、
# 长尾自适应最大 64）；可用 MXK_WIN_PHYS 覆盖（不超过 EXPERT_WINDOW_MAX）。
_WIN_PHYS_DEFAULT = min(EXPERT_WINDOW_MAX, max(
    EXPERT_WINDOW_MIN,
    int(os.environ.get("MXK_WIN_PHYS", EXPERT_WINDOW_MAX) or EXPERT_WINDOW_MAX)))


def _resolve_win_phys(expert_window: int) -> int:
    """逻辑窗口 → 物理容量：至少 _WIN_PHYS_DEFAULT，且不小于逻辑值。"""
    return max(int(expert_window), _WIN_PHYS_DEFAULT)


# ------------------------------------------------------------- 窗口选型（AUTO）
_FP8_AUTO_WINDOW_TABLE_FP8XFP8 = (
    (8, 8), (16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (80, 80),
    (96, 48), (128, 64), (160, 80), (float("inf"), 64))


def _estimate_fp8_mega_moe_rows(num_tokens: int, num_topk: int,
                                num_ranks: int, num_experts: int) -> int:
    return ((int(num_tokens) * int(num_topk) * int(num_ranks))
            + int(num_experts) - 1) // int(num_experts)


def select_fp8_mega_moe_window(num_tokens: int, num_topk: int, num_ranks: int,
                               num_experts: int,
                               mma_type: str = "fp8xfp8") -> int:
    """选择一次 launch 的专家窗口宽度；全 rank 必须同值。"""
    if mma_type not in ("fp8xfp8", "fp8xfp4"):
        raise ValueError(f"unsupported FP8 MegaMoE mma_type: {mma_type!r}")
    for name, value in (("num_tokens", num_tokens), ("num_topk", num_topk),
                        ("num_ranks", num_ranks), ("num_experts", num_experts)):
        if (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if num_topk == 0 or num_ranks == 0 or num_experts == 0:
        raise ValueError("num_topk, num_ranks and num_experts must be positive")
    tail = float(os.environ.get("DG_MEGA_AUTO_TAIL", "1.0") or 1.0)
    hyst = float(os.environ.get("DG_MEGA_AUTO_HYST", "0.0") or 0.0)
    est_tail = _estimate_fp8_mega_moe_rows(
        num_tokens, num_topk, num_ranks, num_experts) * tail
    for count_hi, cap in _FP8_AUTO_WINDOW_TABLE_FP8XFP8:
        if count_hi == float("inf") or est_tail <= count_hi - hyst:
            return cap
    return _FP8_AUTO_WINDOW_TABLE_FP8XFP8[-1][1]


# ---------------------------------------------------------------- 对齐/尺寸
def get_token_alignment_for_mega_moe() -> int:
    return int(_C.get_token_alignment_for_mega_moe())


def get_block_m_for_mega_moe(num_ranks: int, num_experts: int,
                             num_max_tokens_per_rank: int, num_tokens: int,
                             num_topk: int,
                             mma_type: str = "fp8xfp8") -> int:
    return int(_C.get_block_m_for_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk,
        mma_type))


def get_symm_buffer_size_for_mega_moe(
        num_ranks: int, num_experts: int, num_max_tokens_per_rank: int,
        num_topk: int, hidden: int, intermediate_hidden: int,
        num_shared_experts: int = 0, mma_type: str = "fp8xfp8",
        num_tokens: int = 0, expert_window: int = 0) -> int:
    win = expert_window or select_fp8_mega_moe_window(
        num_tokens or num_max_tokens_per_rank, num_topk, num_ranks, num_experts,
        mma_type)
    # 与 SymmBuffer 的实际分配一致：按物理容量计算
    return int(_C.get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden,
        intermediate_hidden, num_shared_experts, mma_type, num_tokens,
        _resolve_win_phys(win)))


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


# ---------------------------------------------------------------- 视图工具
def _view(buf: torch.Tensor, dtype: torch.dtype, shape: Tuple[int, ...],
          offset_bytes: int) -> torch.Tensor:
    """从 1-D uint8 buffer 的字节偏移处切出一个 dtype 视图。"""
    n = 1
    for s in shape:
        n *= int(s)
    itemsize = torch._utils._element_size(dtype)
    flat = buf.as_strided((n * itemsize,), (1,), offset_bytes)
    return flat.view(dtype).view(*shape)


class SymmBuffer:
    """对称区（fine-grained 显存 + 跨 rank IPC）。

    world_size==1 时退化为单块 FG 显存；>1 时用 all_gather_object 交换 IPC
    handle，经 _C.open_ipc_handle 建立本进程内的全 rank 基址表。
    """

    def __init__(self, group, num_experts: int, num_max_tokens_per_rank: int,
                 num_topk: int, hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0, mma_type: str = "fp8xfp8",
                 activation: str = "swiglu", grid_num_tokens: Optional[int] = None,
                 expert_window: Optional[int] = None):
        if activation != "swiglu":
            raise ValueError(f"Only `swiglu` activation is supported, got `{activation}`")
        if mma_type not in _SUPPORTED_MMA_TYPES:
            raise NotImplementedError(
                f"mini_mega torch 层仅实现 {_SUPPORTED_MMA_TYPES}，收到 mma_type={mma_type!r}")
        if num_shared_experts != 0:
            raise NotImplementedError("mini_mega torch 层不支持 shared experts")

        if group is None:
            world_size, rank = 1, 0
        else:
            world_size = int(group.size())
            rank = int(group.rank())

        device = torch.cuda.current_device()
        self.device = device

        cap = _align(int(num_max_tokens_per_rank),
                     _C.get_token_alignment_for_mega_moe())
        if expert_window is None:
            expert_window = select_fp8_mega_moe_window(
                num_max_tokens_per_rank, num_topk, world_size, num_experts,
                mma_type)
        if expert_window < EXPERT_WINDOW_MIN or expert_window > EXPERT_WINDOW_MAX \
                or expert_window % EXPERT_WINDOW_MULTIPLE != 0:
            raise ValueError(
                f"expert_window 必须是 [8,128] 且 8 的倍数，收到 {expert_window}")

        self.group = group
        self.num_experts = int(num_experts)
        self.num_max_tokens_per_rank = cap
        self.num_topk = int(num_topk)
        self.hidden = int(hidden)
        self.intermediate_hidden = int(intermediate_hidden)
        self.num_shared_experts = int(num_shared_experts)
        self.mma_type = mma_type
        self.activation = activation
        self.world_size = world_size
        self.rank = rank
        self.expert_window = int(expert_window)   # 逻辑窗口（调用方/AUTO）
        # 物理容量与逻辑窗口解耦：按最大窗口分配，逐 launch 可在
        # [8, win_phys] 内自适应（长尾不均衡时用更大的 win_cap）。
        self.win_phys = _resolve_win_phys(self.expert_window)
        self.grid_num_tokens = int(grid_num_tokens if grid_num_tokens is not None
                                   else num_max_tokens_per_rank)
        self.launch_count = 0
        # 逐 launch 配置缓存 (win_cap, merge_grid, use_frag)：eager 阶段按真实
        # 路由计算，capture 阶段复用（graph 内无法做 host 同步）。
        self._launch_cfg = None
        self._capture_node = 0
        self._adapt_tick = 0
        # [R38] 前序复用 opt-in：仅当调用方保证 graph 内各节点路由不变时开启。
        self.prologue_reuse = False

        info = _C.get_symm_layout_for_mega_moe(
            world_size, num_experts, cap, num_topk, hidden, intermediate_hidden,
            num_shared_experts, mma_type, self.win_phys)
        (self.view_bytes, self.scratch_offset, self.scratch_bytes, self.sym_bytes,
         total_bytes, x_off, x_sf_off, topk_idx_off, topk_weights_off,
         route_slot_counts_off, src_index_table_off, expert_recv_counts_off,
         prefix_tbl_off, expert_tokens_off, token_route_weights_off,
         token_src_meta_off, barrier_slots_off, combine_rows_off,
         self.src_index_table_bytes, self.expert_recv_counts_bytes,
         self.prefix_tbl_bytes, self.recv_stride) = (int(v) for v in info)

        # ---- 分配 fine-grained 对称区（跨 rank 写必须 FG）----
        self._sym_ptr = int(_C.allocate_finegrained(total_bytes, device))
        self.buffer = _C.wrap_device_ptr(self._sym_ptr, total_bytes, device)
        self.buffer.zero_()
        self._lf_ptr = int(_C.alloc_local_flags(device))
        self._rf_ptr = int(_C.alloc_ranks_flags_fg(device))
        self._owns = True

        # ---- IPC：交换 sym / rf handle，建立全 rank 基址表 ----
        self._remote_sym = []
        self._remote_rf = []
        if world_size == 1:
            base_ptrs = [self._sym_ptr]
            rf_ptrs = [self._rf_ptr]
        else:
            sym_handle = _C.get_ipc_handle(self._sym_ptr)
            rf_handle = _C.get_ipc_handle(self._rf_ptr)
            sym_handles = [sym_handle] * world_size
            rf_handles = [rf_handle] * world_size
            dist.all_gather_object(sym_handles, sym_handle, group=group)
            dist.all_gather_object(rf_handles, rf_handle, group=group)
            base_ptrs = []
            rf_ptrs = []
            for i in range(world_size):
                if i == rank:
                    base_ptrs.append(self._sym_ptr)
                    rf_ptrs.append(self._rf_ptr)
                else:
                    p = int(_C.open_ipc_handle(sym_handles[i]))
                    q = int(_C.open_ipc_handle(rf_handles[i]))
                    self._remote_sym.append(p)
                    self._remote_rf.append(q)
                    base_ptrs.append(p)
                    rf_ptrs.append(q)
                    _C.enable_peer_access(device, i)
            if group is not None:
                group.barrier()
            torch.cuda.synchronize()

        # ---- C++ 侧组装（设备基址表 + 计算链 staging）----
        self._impl = _C.create_symm_buffer(
            self._sym_ptr, self._lf_ptr, self._rf_ptr, base_ptrs, rf_ptrs, rank,
            world_size, num_experts, cap, num_topk, hidden, intermediate_hidden,
            self.win_phys, device)

        # ---- 输入视图 ----
        self.x = _view(self.buffer, torch.float8_e4m3fn, (cap, hidden), x_off)
        self.x_sf = None
        self.topk_idx = _view(self.buffer, torch.int64, (cap, num_topk),
                              topk_idx_off)
        self.topk_weights = _view(self.buffer, torch.float32, (cap, num_topk),
                                  topk_weights_off)
        self.l1_acts = _view(self.buffer, torch.float8_e4m3fn,
                             (self.win_phys, hidden), expert_tokens_off)
        self.l2_acts = _view(self.buffer, torch.float8_e4m3fn,
                             (self.win_phys, intermediate_hidden),
                             token_route_weights_off)
        self.shared_l1_acts = self.x
        self.shared_l1_acts_sf = None
        self.shared_l2_acts = None
        self.shared_l2_acts_sf = None

        # REDUCE 输出（FP32），launch 后转 BF16 拷进用户 y
        self._y_fp32 = torch.empty((cap, hidden), dtype=torch.float32,
                                   device=device)

        import types
        self.handle = types.SimpleNamespace(
            buffer_ptrs=base_ptrs, rank=rank, world_size=world_size)

        # 建档期（capture 之前）预热：第 1 次发射触发跨 rank IPC 懒映射，
        # 首档存在残余竞态（202 R=32 实测 1/3 首档偶发错）。sglang 首次
        # fp8_mega_moe 落在 CUDA graph 捕获内，必须在建档时把映射走稳，
        # 否则捕获流上阻塞/结果错。预热不触碰用户数据（建档后用户才写
        # x/topk），结果丢弃。
        self._warmed = False
        try:
            self._warmup_at_build()
        except Exception as _e:  # noqa: BLE001
            print(f"[mxmegakernel] warmup at build skipped: {_e}", flush=True)

    def _warmup_at_build(self):
        if self._warmed:
            return
        self._warmed = True
        dev = self.device
        epr = self.num_experts // self.world_size
        l1 = torch.zeros((epr, 2 * self.intermediate_hidden, self.hidden),
                         dtype=torch.float8_e4m3fn, device=dev)
        l2 = torch.zeros((epr, self.hidden, self.intermediate_hidden),
                         dtype=torch.float8_e4m3fn, device=dev)
        tl1, tl2 = transform_weights_for_mega_moe(l1, l2)
        # 均衡路由覆盖全部专家：跨 rank FG 页是懒映射，首档只碰少数专家会留下
        # 未映射页（capture 内首档偶发错）。用较多 token 把全部专家 pool 走一遍。
        T = max(1, min(int(self.num_max_tokens_per_rank), 256))
        idx = (torch.arange(T, device=dev).unsqueeze(1)
               + torch.arange(self.num_topk, device=dev).unsqueeze(0))
        idx = (idx % int(self.num_experts)).to(self.topk_idx.dtype)
        self.topk_idx[:T].copy_(idx)
        self.topk_weights[:T].fill_(1.0 / max(1, self.num_topk))
        for _ in range(2):
            # 建档预热不参与自适应：用逻辑窗口 + 缺省 merge + 无 frag，
            # 避免合成均衡路由污染 _launch_cfg。
            _C.fp8_mega_moe(self._y_fp32, tl1.view(torch.uint8),
                            tl2.view(torch.uint8), self._impl, T, 1e30,
                            int(self.expert_window), _KERNEL_GRID, False, 0)
            torch.cuda.synchronize()
        del l1, l2, tl1, tl2
        torch.cuda.empty_cache()

    # ---------------------------------------------------------- 自适应发射配置
    def _select_adaptive_cfg(self, num_tokens: int):
        """按真实路由选择本次 launch 的 (win_cap, merge_grid, use_frag)。

        与 C++ bench 的 tk 长尾判据逐式一致：
          imbalanced = max_expert >= 3 * max(1, median(nonzero counts))
          imbalanced: med>=32 → win=ceil(max/RING) else min(48, max)，
                      再按 M tile 对齐；merge 不粗化；启用 A/f2 fragment。
          否则：保持 AUTO 逻辑窗口 + 缺省粗化 + 不用 frag。
        只在非 capture 路径做 host 同步；capture 用缓存值（无则回退缺省）。
        """
        base = (int(self.expert_window), _KERNEL_GRID, False)
        if num_tokens <= 0 or self.group is None or self.world_size <= 1:
            return base
        if os.environ.get("MXK_ADAPTIVE", "1") == "0":
            return base
        every = max(1, int(os.environ.get("MXK_ADAPT_EVERY_N", "1") or 1))
        self._adapt_tick += 1
        if every > 1 and (self._adapt_tick - 1) % every != 0 and self._launch_cfg is not None:
            return self._launch_cfg
        counts = None
        try:
            e = int(self.num_experts)
            idx = self.topk_idx[:num_tokens].reshape(-1)
            idx = idx[(idx >= 0) & (idx < e)]   # 过滤 padding/-1 等非法 id
            if hasattr(torch, "bincount"):
                counts = torch.bincount(idx, minlength=e).to(torch.int64)[:e]
            else:  # 兜底：bincount 不可用时 scatter_add
                counts = torch.zeros(e, dtype=torch.int64, device=idx.device)
                counts.scatter_add_(0, idx,
                                    torch.ones_like(idx, dtype=torch.int64))
            dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=self.group)
            max_expert = int(counts.max().item())
            nz = counts[counts > 0]
            med = int(nz.median().item()) if nz.numel() > 0 else 1
            if max_expert >= 3 * max(1, med):
                if med >= 32:
                    tgt = (max_expert + _KERNEL_RING - 1) // _KERNEL_RING
                else:
                    tgt = min(48, max_expert)
                tgt = ((tgt + _KERNEL_M_TILE - 1) // _KERNEL_M_TILE) * _KERNEL_M_TILE
                win = max(EXPERT_WINDOW_MIN,
                          min(int(self.win_phys), max(_KERNEL_M_TILE, tgt)))
                # [BUGFIX 261007] merge_grid=0x3FFFFFFF（不粗化）会跨波，多波下
                #   wait-all/旗子陈旧读 → 结果错误（harness case 1 rel=0.229）。
                #   回退缺省单波粗化（0）保留 win_cap/frag 自适应。
                cfg = (win, 0, True)
                self._launch_cfg = cfg
                return cfg
        except Exception:  # noqa: BLE001
            return base
        self._launch_cfg = base
        return base

    def launch_config(self, num_tokens: int):
        """当前应使用的 (win_cap, merge_grid, use_frag, skip_prologue)。"""
        capturing = _is_stream_capturing()
        if capturing:
            cfg = self._launch_cfg or (int(self.expert_window), _KERNEL_GRID,
                                       False)
            skip = 2 if (self.prologue_reuse and self._capture_node > 0) else 0
            self._capture_node += 1
            return cfg[0], cfg[1], cfg[2], skip
        self._capture_node = 0
        cfg = self._select_adaptive_cfg(int(num_tokens))
        return cfg[0], cfg[1], cfg[2], 0

    def destroy(self):
        try:
            if getattr(self, "_impl", None) is not None:
                _C.destroy_symm_buffer(self._impl)
                self._impl = None
        except Exception as e:  # noqa: BLE001
            print(f"[mxmegakernel] destroy_symm_buffer failed: {e}")
        for p in getattr(self, "_remote_sym", []) or []:
            try:
                _C.close_ipc_handle(p)
            except Exception:  # noqa: BLE001
                pass
        for p in getattr(self, "_remote_rf", []) or []:
            try:
                _C.close_ipc_handle(p)
            except Exception:  # noqa: BLE001
                pass
        if getattr(self, "_owns", False):
            for attr in ("_sym_ptr", "_lf_ptr", "_rf_ptr"):
                p = getattr(self, attr, 0)
                if p:
                    try:
                        _C.free_device(p)
                    except Exception:  # noqa: BLE001
                        pass
                    setattr(self, attr, 0)
        self._remote_sym = []
        self._remote_rf = []
        self.buffer = None
        self.handle = None
        self.group = None


def get_symm_buffer_for_mega_moe(
        group, num_experts: int, num_max_tokens_per_rank: int, num_topk: int,
        hidden: int, intermediate_hidden: int, num_shared_experts: int = 0,
        use_fp8_dispatch: Optional[bool] = None, mma_type: str = "fp8xfp8",
        activation: str = "swiglu", num_tokens: Optional[int] = None,
        expert_window: Optional[int] = None) -> SymmBuffer:
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split("x")[0] == "fp8")
    return SymmBuffer(group, num_experts, num_max_tokens_per_rank, num_topk,
                      hidden, intermediate_hidden, num_shared_experts,
                      mma_type=mma_type, activation=activation,
                      grid_num_tokens=num_tokens, expert_window=expert_window)


# ---------------------------------------------------------------- 权重打包
def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    assert t.dim() in (2, 3)
    squeeze = t.dim() == 2
    if squeeze:
        t = t.unsqueeze(0)
    is_fp8 = t.dtype == torch.float8_e4m3fn
    work = t.view(torch.uint8) if is_fp8 else t
    g, n, *rest = work.shape
    half = n // 2
    gate = work[:, :half].reshape(g, half // gran, gran, *rest)
    up = work[:, half:].reshape(g, half // gran, gran, *rest)
    result = torch.empty_like(work).copy_(
        torch.stack([gate, up], dim=2).reshape(g, n, *rest))
    if is_fp8:
        result = result.view(torch.float8_e4m3fn)
    return result.squeeze(0) if squeeze else result


def _swizzle_lane(t: torch.Tensor) -> torch.Tensor:
    """把 [E, N, K]（N%32==0, K%32==0）打成 mini_mega kernel 的 16B/lane
    packed tile 布局：tile=[n/32][k/32] 1024B，lane L=kg*16+(n%16)，lane 内
    低 8B=逻辑行 n、K 段 [(L/16)*8,+8)，高 8B=逻辑行 n+16 的同段。
    （与 fp8mega::fill_weights_lane 逐字节同式。）
    """
    if t.dim() not in (2, 3):
        raise TypeError("lane swizzle requires 2D/3D tensor")
    squeeze = t.dim() == 2
    grouped = t.unsqueeze(0) if squeeze else t
    ng, n, k = grouped.shape
    if n % 32 != 0 or k % 32 != 0:
        raise ValueError(f"lane swizzle requires N%32==0 and K%32==0; got N={n}, K={k}")
    work = grouped.contiguous().view(torch.uint8) if grouped.dtype == torch.float8_e4m3fn \
        else grouped.contiguous()
    # dims: [E][np=N/32][rh=2][rl=16][kt=K/32][kg=4][b=8]
    packed = (work.reshape(ng, n // 32, 2, 16, k // 32, 4, 8)
              .permute(0, 1, 4, 5, 3, 2, 6).contiguous()
              .reshape(ng, n, k))
    if grouped.dtype == torch.float8_e4m3fn:
        packed = packed.view(torch.float8_e4m3fn)
    return packed.squeeze(0) if squeeze else packed


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    if sf.dtype != torch.int or sf.dim() not in (2, 3):
        raise TypeError("transpose_sf_for_utccp requires int 2D/3D tensor")
    squeeze = sf.dim() == 2
    if squeeze:
        sf = sf.unsqueeze(0)
    ng, mn, packed_k = sf.shape
    assert mn % 128 == 0
    result = (sf.reshape(ng, -1, 4, 32, packed_k)
              .transpose(2, 3).reshape(ng, mn, packed_k))
    result = torch.empty_like(sf).copy_(result)
    return result.squeeze(0) if squeeze else result


_interleave_l1_weights = _interleave_weights


def transform_weights_for_mega_moe(
        l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        activation: str = "swiglu"):
    """权重离线打包（仅 fp8xfp8 / raw E4M3）。

    W1: gate/up 在 l1_block_n(=256) 粒度配对 → 16B/lane packed tile。
    W2: 直接 16B/lane packed tile（无 gate/up 配对）。
    """
    assert activation == "swiglu", f"Only `swiglu` supported, got `{activation}`"
    if isinstance(l1_weights, tuple) or isinstance(l2_weights, tuple):
        raise NotImplementedError(
            "mini_mega torch 层仅实现 fp8xfp8 的权重打包（raw E4M3）")
    for name, t in (("l1_weights", l1_weights), ("l2_weights", l2_weights)):
        if not isinstance(t, torch.Tensor) or t.dtype != torch.float8_e4m3fn:
            raise TypeError(f"{name} must be a float8_e4m3fn tensor")
    # W1：先按 l1_block_n 做 gate/up 配对，再 lane 打包
    l1_t = _swizzle_lane(_interleave_weights(l1_weights, gran=_L1_BLOCK_N))
    # W2：直接 lane 打包
    l2_t = _swizzle_lane(l2_weights)
    return l1_t, l2_t


# ---------------------------------------------------------------- kernel 发射
def _is_stream_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def fp8_mega_moe(y: torch.Tensor, l1_weights: torch.Tensor,
                 l2_weights: torch.Tensor, sym_buffer: SymmBuffer,
                 cumulative_local_expert_recv_stats=None,
                 activation: str = "swiglu", activation_clamp=None,
                 expert_window: Optional[int] = None) -> None:
    if sym_buffer.mma_type != "fp8xfp8":
        raise ValueError(
            f"fp8_mega_moe requires an fp8xfp8 SymmBuffer, got {sym_buffer.mma_type!r}")
    if activation != "swiglu":
        raise ValueError(f"fp8xfp8 requires activation='swiglu', got {activation!r}")
    num_tokens = int(y.shape[0])
    if num_tokens > sym_buffer.num_max_tokens_per_rank:
        raise ValueError(
            f"num_tokens={num_tokens} exceeds capacity "
            f"{sym_buffer.num_max_tokens_per_rank}")
    clamp = 1e30 if activation_clamp is None else float(activation_clamp)
    # 调用方显式 expert_window 作为逻辑窗口兜底（自适应/缓存优先级更高，
    # 目的是调用方零改动也能走长尾快路径；可用 MXK_ADAPTIVE=0 关自适应）。
    if expert_window is not None:
        ew = int(expert_window)
        if ew < EXPERT_WINDOW_MIN or ew > sym_buffer.win_phys:
            raise ValueError(
                f"expert_window={ew} 超出 [8, {sym_buffer.win_phys}]")
        sym_buffer.expert_window = ew
    # 逐 launch 配置：eager 按路由自适应，capture 复用缓存。
    win_cap, merge_grid, use_frag, skip = sym_buffer.launch_config(num_tokens)
    # 首次 launch 兜底预热（建档预热失败时）：捕获中禁止，避免捕获流阻塞。
    if not getattr(sym_buffer, "_mega_moe_warmed", False):
        sym_buffer._mega_moe_warmed = True
        if not _is_stream_capturing():
            for _ in range(2):
                _C.fp8_mega_moe(
                    sym_buffer._y_fp32, l1_weights.view(torch.uint8),
                    l2_weights.view(torch.uint8), sym_buffer._impl, num_tokens,
                    clamp, int(sym_buffer.expert_window), _KERNEL_GRID, False, 0)
                torch.cuda.synchronize()
    _C.fp8_mega_moe(sym_buffer._y_fp32, l1_weights.view(torch.uint8),
                    l2_weights.view(torch.uint8), sym_buffer._impl, num_tokens,
                    clamp, win_cap, merge_grid, use_frag, skip)
    sym_buffer.launch_count += 1
    if num_tokens > 0:
        y.copy_(sym_buffer._y_fp32[:num_tokens].to(torch.bfloat16))


def bf16_mega_moe(*args, **kwargs):
    raise NotImplementedError("mini_mega 仅实现 fp8xfp8")


def bf16_fp8_mega_moe(*args, **kwargs):
    raise NotImplementedError("mini_mega 仅实现 fp8xfp8")


def fp8_fp4_mega_moe(*args, **kwargs):
    raise NotImplementedError("mini_mega 仅实现 fp8xfp8")


def read_phase_ticks(*args, **kwargs):
    raise NotImplementedError("read_phase_ticks 未实现")


def read_dbg_report(*args, **kwargs):
    raise NotImplementedError("read_dbg_report 未实现")


def read_block_marks(*args, **kwargs):
    raise NotImplementedError("read_block_marks 未实现")


__all__ = [
    "FROZEN_HIDDEN", "FROZEN_INTERMEDIATE_HIDDEN", "FROZEN_NUM_TOPK",
    "FROZEN_NUM_EXPERTS_PER_RANK",
    "SymmBuffer", "get_symm_buffer_for_mega_moe",
    "get_symm_buffer_size_for_mega_moe", "select_fp8_mega_moe_window",
    "get_token_alignment_for_mega_moe", "get_block_m_for_mega_moe",
    "transform_weights_for_mega_moe",
    "fp8_mega_moe", "bf16_mega_moe", "bf16_fp8_mega_moe", "fp8_fp4_mega_moe",
    "read_phase_ticks", "read_dbg_report", "read_block_marks",
]
