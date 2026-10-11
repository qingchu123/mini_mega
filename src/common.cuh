// ============================================================================
// common.cuh — fp8xfp8 MoE Mega Kernel 公共定义（场景常量 / 工具）
//
// 项目：zzzz/projs_deepseek/projs/01.fp8xfp8_moe_kernel
// 基准：TPR=16 均衡（EPR=8 / topk=8 / H=6144 / I=2048，R=2/8/32）
// ============================================================================
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------- 场景常量 --
constexpr int kMaxTPR = 512;              // 每 rank 最大发送tokens数目
constexpr int kEPR = 8;              // 每 rank 专家数
constexpr int kTopk = 8;             // 每 token 路由专家数
constexpr int kHidden = 6144;        // H（K_IN / K_OUT）
constexpr int kIntermediate = 2048;  // I（中间维度）
constexpr int kGrid = 32;            // grid 块数（= SM 数）
constexpr int kRanks = 32;           
#ifndef KNSLICE_OVERRIDE
#define KNSLICE_OVERRIDE 4
#endif
constexpr int kNSlice = KNSLICE_OVERRIDE;  // 每窗口任务切片数（K 片 / N 组）；NS=4：L1 每片 4 组、L2 每片 6 组
#ifndef KRING_OVERRIDE
#define KRING_OVERRIDE 8
#endif
constexpr int kRing = KRING_OVERRIDE;  // 专家窗口槽数（ring 复用）
constexpr int kMaxWin = 8;  // 专家窗口槽数（ring 复用）
constexpr int kM = 16;               // M tile 行数
// L1 N tile 宽（生产 block_n；原 MegaArgs.l1_block_n 的固定值）
constexpr int kL1BlockN = 256;
// recv 行宽（word）：与 layout 的 ((EPR+1)+3)&~3 一致（原 MegaArgs.recv_stride）
constexpr int kRecvStride = (kEPR + 1 + 3) & ~3;

// [OPT-GEMM-ATTR] 重型 GEMM 档函数属性：__noinline__ 让每个档独立寄存器/栈
//   分配，调用者（任务体/内核）只承担控制流与调用约定，避免最坏路径的 acc/
//   寄存器预算传染到其它档。可用 -DR19_GEMM_ATTR=__forceinline__ 回退实验。
#ifndef R19_GEMM_ATTR
#define R19_GEMM_ATTR __noinline__
#endif

// 自旋上限（异常保护）：约 1s（4M 周期 @2GHz）
constexpr long long kSpinLimit = 4000000LL;
// PUBLISH 校验槽 epoch（阶段 1 用常量；生产为 epoch_base+1）
constexpr unsigned kPublishEpoch = 1u;

// =============================================================================
// 内部执行次数统计（MX_DBG_COUNT）
//
// 双重开关：
//   编译期：MX_DBG_COUNT（默认 0）。为 0 时下方宏展开为空语句 do{}while(0)，
//           编译期即消失，不产生任何 atomicAdd / 全局内存读写 / 分支，
//           性能基线与改造前逐字一致。
//   运行期：MegaArgs::dbg_count（host 侧读 env MXKERNEL_COUNT 填入）。
//           device 侧在 mega_kernel 入口做一次 `dbg_count ? exec_ctr : nullptr`
//           归一，把运行期开关折叠成单指针判空（每相位 1 次，非热路径）。
//
// 计数缓冲：独立缓冲（unsigned long long* exec_ctr，[kExecCtrSlots]），
//           不复用既有 dbg[8]，避免踩现有超时语义。
// graph：atomicAdd 到 device 缓冲在 capture/replay 下合法；但**清零必须在
//        capture 之前、在捕获流之外完成**（不能在 capture 流里做依赖 host
//        同步的清零）。
// =============================================================================
#ifndef MX_DBG_COUNT
#define MX_DBG_COUNT 0
#endif

// 计数槽总数（尾部预留：kExecCtrSlots - 1 个空槽，便于后续扩展不改结构体）
constexpr int kExecCtrSlots = 48;

// 计数槽位表（每项语义见中文注释；跨版本只增不改，保证历史日志可比）
enum ExecCtrSlot : int {
    kCtrKernelLaunch = 0,    // mega_kernel 启动次数（block0 tid0 计 1 => 等于 launch 次数）
    kCtrKernelBlocks = 1,    // mega_kernel 参与 block 总数（应 = gridDim.x × 启动数）
    kCtrPhasePrologue = 2,   // PROLOGUE 相位参与 block 数（协议上恒 = 1）
    kCtrPhaseRouteBuild = 3, // ROUTE_BUILD 执行次数
    kCtrPhasePublish = 4,    // PUBLISH_COUNTS 执行次数
    kCtrPhaseXbar = 5,       // XBAR_ROUTE_READY 参与 block 数（应 = gridDim.x）
    kCtrPhaseTaskSetup = 6,  // TASK_SETUP 参与 block 数（应 = gridDim.x）
    kCtrTaskSetupLeader = 7, // TASK_SETUP 以 leader(block0) 身份执行的次数
    kCtrTaskSetupFail = 8,   // TASK_SETUP checksum 超时返回 -1 的次数
    kCtrLoopEnter = 9,       // 进入任务循环的 block 数
    kCtrLoopExit = 10,       // 任务循环正常结束（break）的 block 数
    kCtrTasks = 11,          // 实际处理的任务总数（Σ 各 block processed）
    kCtrSlices = 12,         // 实际处理的逻辑 slice 总数（Σ 任务 × nslice_eff）
    kCtrTasksPerBlockMax = 13, // 单 block 处理任务数最大值（atomicMax）
    kCtrTasksPerBlockMin = 14, // 单 block 处理任务数最小值（初值 ~0ull，atomicMin）
    kCtrL1Exec = 15,         // L1 阶段执行次数（count>0 的任务）
    kCtrL2Exec = 16,         // L2 阶段执行次数（count>0 的任务）
    kCtrCombineExec = 17,    // COMBINE(tail) 实际执行推送的次数
    kCtrCombineSkip = 18,    // COMBINE 被跳过（combine 关闭 / rows_off<0）的次数
    kCtrL1ProgWait = 19,     // l1_prog 门等待次数
    kCtrL1ProgSpin = 20,     // l1_prog 门自旋总轮数
    kCtrL1ProgTimeout = 21,  // l1_prog 门等待超时次数
    kCtrL2DoneWait = 22,     // l2_done ring 门等待次数
    kCtrL2DoneSpin = 23,     // l2_done 门自旋总轮数
    kCtrL2DoneTimeout = 24,  // l2_done 门等待超时次数
    kCtrSliceGateWait = 25,  // L1 K 循环分片门等待次数
    kCtrSliceGateSpin = 26,  // 分片门自旋总轮数
    kCtrSliceGateTimeout = 27, // 分片门等待超时次数
    kCtrXbarWait = 28,       // xbar release 等待次数
    kCtrXbarSpin = 29,       // xbar 等待自旋总轮数
    kCtrXbarTimeout = 30,    // xbar 等待超时次数
    kCtrTaskReadyWait = 31,  // task_ready 等待次数
    kCtrTaskReadySpin = 32,  // task_ready 自旋总轮数
    kCtrTaskReadyTimeout = 33, // task_ready 等待超时次数
    kCtrChecksumRow = 34,    // checksum 轮询的行数
    kCtrChecksumSpin = 35,   // checksum 自旋总轮数
    kCtrChecksumTimeout = 36, // checksum 超时次数
    kCtrWaitAllSlice = 37,   // 任务内 wait-all（分片就绪旗）等待轮次
    kCtrWaitAllL1Prog = 38,  // 任务内 wait-all（l1_prog 旗）等待轮次
    kCtrL2DonePublish = 39,  // 任务尾 l2_done 递增发布次数
    kCtrReduceLaunch = 40,   // REDUCE kernel 启动次数
    kCtrReduceBlock = 41,    // REDUCE 参与 block 数
    kCtrReduceWait = 42,     // REDUCE 等待 combine 标记的轮次
    kCtrReduceSpin = 43,     // REDUCE 等待自旋总轮数
    kCtrReduceTimeout = 44,  // REDUCE 等待超时次数
    kCtrRoutePair = 45,      // ROUTE_BUILD 处理的 (token,topk) pair 总数
    kCtrRouteDrop = 46,      // ROUTE_BUILD 因超容量/非法 ID 丢弃的 pair 数
};

// 运行期开关归一（device 侧；每 kernel 入口调一次，非热路径）
__device__ __forceinline__ unsigned long long* mx_ctr_open(
        int dbg_count, unsigned long long* exec_ctr) {
#if MX_DBG_COUNT
    return (dbg_count != 0) ? exec_ctr : nullptr;
#else
    (void)dbg_count; (void)exec_ctr;
    return nullptr;
#endif
}

#if MX_DBG_COUNT
// 开启态：原子累加 / 原子取极值
#define MX_CTR_ADD(ctr, slot, n)                                     \
    do {                                                             \
        if ((ctr) != nullptr)                                        \
            atomicAdd((ctr) + (slot), (unsigned long long)(n));      \
    } while (0)
#define MX_CTR_MAX(ctr, slot, v)                                     \
    do {                                                             \
        if ((ctr) != nullptr)                                        \
            atomicMax((ctr) + (slot), (unsigned long long)(v));      \
    } while (0)
#define MX_CTR_MIN(ctr, slot, v)                                     \
    do {                                                             \
        if ((ctr) != nullptr)                                        \
            atomicMin((ctr) + (slot), (unsigned long long)(v));      \
    } while (0)
#else
// 关闭态（默认）：宏展开为空语句，参数不被求值 => 编译期彻底消除
#define MX_CTR_ADD(ctr, slot, n) \
    do {                        \
    } while (0)
#define MX_CTR_MAX(ctr, slot, v) \
    do {                         \
    } while (0)
#define MX_CTR_MIN(ctr, slot, v) \
    do {                         \
    } while (0)
#endif

// 槽位中文名（host 打印用；索引与 ExecCtrSlot 一一对应）
static constexpr const char* kExecCtrNames[kExecCtrSlots] = {
    "kernel启动次数",        "kernel参与block数",     "PROLOGUE参与block数",
    "ROUTE_BUILD执行次数",   "PUBLISH_COUNTS执行次数", "XBAR_ROUTE_READY参与block数",
    "TASK_SETUP参与block数", "TASK_SETUP_leader次数", "TASK_SETUP_checksum失败数",
    "任务循环进入block数",   "任务循环结束block数",   "处理任务总数",
    "处理逻辑slice总数",     "单block任务数max",     "单block任务数min",
    "L1执行次数",            "L2执行次数",           "COMBINE执行次数",
    "COMBINE跳过次数",       "l1_prog等待次数",       "l1_prog自旋轮数",
    "l1_prog超时次数",       "l2_done门等待次数",     "l2_done门自旋轮数",
    "l2_done门超时次数",     "分片门等待次数",        "分片门自旋轮数",
    "分片门超时次数",        "xbar等待次数",          "xbar自旋轮数",
    "xbar超时次数",          "task_ready等待次数",    "task_ready自旋轮数",
    "task_ready超时次数",    "checksum轮询行数",      "checksum自旋轮数",
    "checksum超时次数",      "waitall分片旗轮次",     "waitall_l1prog旗轮次",
    "l2_done发布次数",       "REDUCE启动次数",        "REDUCE参与block数",
    "REDUCE等待轮次",        "REDUCE自旋轮数",        "REDUCE超时次数",
    "ROUTE处理pair数",       "ROUTE丢弃pair数",
};

// ---------------------------------------------------------------- 运行选项 --
struct SceneOpts {
    int ranks = 8;           // rank 数（2/8/32）
    int max_tokens = 16;     // 专家 token 容量（TPR）
    int win_cap = -1;        // 窗口容量；-1 = 与 max_tokens 相同
    int win_pad = 0;         // pool 槽地址 padding（默认 0：>0 会越出生产 pool 区，见骨架 §7.6）
    int tokens = 16;         // 每 rank 实际 token 数（<= max_tokens）
    int window = 8;          // 生产 smem 计算用窗口
    int block_n = 256;       // N tile（512/256/128）
    int nodes = 2000;        // graph 链节点数
    bool audit = true;       // 显存审计
    bool perf_only = false;  // 仅计时（跳过正确性，<=20s 规则）
};
