"""mini_mega torch 层单/多 rank 数值冒烟测试（fp8xfp8）。

用法（单卡）：
    cd torch && PYTHONPATH=$PWD python tests/test_fp8_mega_moe.py
多卡（NCCL，R=2）：
    torchrun --nproc_per_node=2 tests/test_fp8_mega_moe.py --ranks 2
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mxmegakernel as mxk  # noqa: E402

H = 6144
I = 2048
EPR = 8


def _silu(x):
    return x / (1.0 + torch.exp(-x))


def reference(x, topk_idx, topk_weights, w1_all, w2_all, num_ranks, num_experts):
    """w1_all [R,EPR,2I,H], w2_all [R,EPR,H,I]（原始，未 transform）。"""
    T = x.shape[0]
    y = torch.zeros((T, H), dtype=torch.float32, device=x.device)
    xf = x.float()
    for k in range(topk_idx.shape[1]):
        for e in topk_idx[:, k].unique().tolist():
            if not (0 <= e < num_experts):
                continue
            owner, le = divmod(int(e), EPR)
            mask = topk_idx[:, k] == e
            rows = xf[mask]
            w1e = w1_all[owner][le].float()
            w2e = w2_all[owner][le].float()
            gate = rows @ w1e[:I].T
            up = rows @ w1e[I:].T
            h = _silu(gate) * up
            h_fp8 = h.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
            out = (h_fp8 @ w2e.T).bfloat16().float()
            y[mask] += out * topk_weights[mask, k].float()[:, None]
    return y


def run(local_rank, num_local_ranks, args):
    torch.cuda.set_device(local_rank)
    rank = local_rank
    world = num_local_ranks
    if world > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world)

    num_experts = EPR * world
    topk = 8
    cap = args.cap
    T = args.tokens
    win = args.window

    group = dist.group.WORLD if world > 1 else None

    buf = mxk.get_symm_buffer_for_mega_moe(
        group, num_experts, cap, topk, H, I,
        num_shared_experts=0, mma_type="fp8xfp8", activation="swiglu",
        num_tokens=T, expert_window=win)

    torch.manual_seed(1234 + rank)
    # 每 rank 只持有本 rank 的 EPR 个专家权重（与 sglang 的 local shard 一致）
    w1 = (torch.rand((EPR, 2 * I, H), device="cuda") - 0.5).mul(0.125).to(torch.float8_e4m3fn)
    w2 = (torch.rand((EPR, H, I), device="cuda") - 0.5).mul(0.125).to(torch.float8_e4m3fn)
    x = (torch.rand((T, H), device="cuda") - 0.5).mul(0.25).to(torch.float8_e4m3fn)
    # 参考需要全 rank 权重
    w1_all = [torch.empty_like(w1) for _ in range(world)]
    w2_all = [torch.empty_like(w2) for _ in range(world)]
    if world > 1:
        dist.all_gather(w1_all, w1)
        dist.all_gather(w2_all, w2)
    else:
        w1_all, w2_all = [w1], [w2]

    # 路由：每 token 8 个互异专家
    if args.cross_only:
        other = 1 - rank
        pool = torch.arange(other * EPR, (other + 1) * EPR, device="cuda")
        topk_idx = pool[torch.stack(
            [torch.randperm(EPR, device="cuda") for _ in range(T)])].to(torch.int64)
    elif args.local_only:
        pool = torch.arange(rank * EPR, (rank + 1) * EPR, device="cuda")
        topk_idx = pool[torch.stack(
            [torch.randperm(EPR, device="cuda") for _ in range(T)])].to(torch.int64)
    else:
        topk_idx = torch.stack(
            [torch.randperm(num_experts, device="cuda")[:topk] for _ in range(T)]).to(torch.int64)
    topk_weights = torch.rand((T, topk), device="cuda", dtype=torch.float32)

    buf.x[:T].copy_(x)
    buf.topk_idx[:T].copy_(topk_idx)
    buf.topk_weights[:T].copy_(topk_weights)
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()

    l1_t, l2_t = mxk.transform_weights_for_mega_moe(w1, w2)
    y = torch.empty((T, H), dtype=torch.bfloat16, device="cuda")
    mxk.fp8_mega_moe(y, l1_t, l2_t, buf, activation="swiglu")

    torch.cuda.synchronize()

    y_ref = reference(x, topk_idx, topk_weights, w1_all, w2_all, world, num_experts)
    y_ref_bf = y_ref.bfloat16().float()
    print(f"[rank {rank}] ref_finite={torch.isfinite(y_ref_bf).all().item()} "
          f"y_finite={torch.isfinite(y.float()).all().item()} "
          f"ref_max={y_ref_bf.abs().max().item():.4f} "
          f"y_max={y.float().abs().max().item():.4f}", flush=True)
    diff = (y.float() - y_ref_bf).abs()
    rel = diff.max().item() / (y_ref_bf.abs().max().item() + 1e-6)
    print(f"[rank {rank}] max_abs={diff.max().item():.4f} "
          f"rel={rel:.4f} y_max={y_ref_bf.abs().max().item():.4f}", flush=True)
    ok = rel < args.tol
    print(f"[rank {rank}] {'PASS' if ok else 'FAIL'} (tol={args.tol})", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ranks", type=int, default=1)
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--cap", type=int, default=16)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--tol", type=float, default=0.06)
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--cross-only", action="store_true")
    args = p.parse_args()
    if args.ranks > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", args.ranks))
        return run(local_rank, world, args)
    return run(0, 1, args)


if __name__ == "__main__":
    raise SystemExit(main())
