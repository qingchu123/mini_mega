# [临时] torch 侧对拍 TK 用例 launch=804300（check 已验证 win=128 可通过）
# 输入：x / topk 权重 与 check fill_inputs 一致；路由读取 check 导出的 route_launch=804300_r{rank}.bin
# 权重：fill_weights_lane seed 17/91（与 mega_common 一致，不做 transform）
# 布局：tokens=16（TK 用例 tokens），expert_window=128；对拍文件 y_launch=804300_r{rank}.bin（FP32）
import os, sys
import numpy as np
import torch
import torch.distributed as dist
from mxmegakernel import _C
from mxmegakernel.mega import SymmBuffer, fp8_mega_moe

DUMP = sys.argv[1]
LABEL = "launch=804300"
TOK = 16; H = 6144; I = 2048; TOPK = 8; EPR = 8; R = 32; E = R * EPR; WIN = 128

local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dist.init_process_group("nccl")
rank = dist.get_rank(); world = dist.get_world_size()
assert world == R, world

buf = SymmBuffer(group=dist.group.WORLD, num_experts=E, num_max_tokens_per_rank=TOK,
                 num_topk=TOPK, hidden=H, intermediate_hidden=I, expert_window=WIN)

w1 = torch.empty((EPR, 2 * I, H), dtype=torch.uint8, device="cuda")
w2 = torch.empty((EPR, H, I), dtype=torch.uint8, device="cuda")
_C.fill_weights_lane(w1.view(-1), EPR * 2 * I, H, 17)
_C.fill_weights_lane(w2.view(-1), EPR * H, I, 91)

# x：与 check fill_inputs 相同：字节 (rank*37 + t*7 + c*3) & 0x3f（前 TOK 行）
t = torch.arange(TOK, device="cuda").view(-1, 1)
c = torch.arange(H, device="cuda").view(1, -1)
xb = ((rank * 37 + t * 7 + c * 3) & 0x3f).to(torch.uint8)
buf.x.view(torch.uint8)[:TOK].copy_(xb)

# topk 权重：与 check 相同 wv[i] = 1 + rank*0.25 + (i%7)*0.01，i = token*TOPK + k
i = torch.arange(TOK * TOPK, device="cuda").view(TOK, TOPK).to(torch.float32)
buf.topk_weights[:TOK].copy_(1.0 + rank * 0.25 + (i % 7) * 0.01)

# 路由：读取 check 导出的同一用例路由（int64，[TOK][TOPK] 行主序）
route = np.fromfile(os.path.join(DUMP, f"route_{LABEL}_r{rank}.bin"), dtype=np.int64).reshape(TOK, TOPK)
buf.topk_idx[:TOK].copy_(torch.from_numpy(route.copy()).to("cuda"))

y = torch.zeros((TOK, H), dtype=torch.bfloat16, device="cuda")  # 接口要求 BF16 输出
fp8_mega_moe(y, w1, w2, buf, None)  # 发射 win 取 launch_config（=expert_window=128）
got = buf._y_fp32[:TOK].clone().cpu().numpy()
torch.cuda.synchronize(); dist.barrier()

ref = np.fromfile(os.path.join(DUMP, f"y_{LABEL}_r{rank}.bin"), dtype=np.float32).reshape(TOK, H)
diff = np.abs(got - ref)
mx = float(diff.max()); bad = int((diff > 1e-3 * np.maximum(1.0, np.abs(ref))).sum())
print(f"[rank {rank}] max_abs={mx:.3e} bad={bad} ref_absmax={float(np.abs(ref).max()):.3e}", flush=True)
dist.barrier()
if rank == 0:
    print("COMPARE_TK done", flush=True)
buf.destroy()
dist.destroy_process_group()
