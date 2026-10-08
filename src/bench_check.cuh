// ============================================================================
// bench_check.cuh — host 侧正确性校验
//   M1：前序三件套（make_routes / validate_route / validate_publish /
//       validate_task_setup；移植自骨架，见各函数头注释）
//   M2/M3：数值正确性参考（E4M3 编解码 / lane 序 tile 读取 / L1 / L2 /
//       REDUCE 的 CPU 参考 + 容差比较；见文件末尾"数值正确性参考"段）
//
// 路由模型与校验逻辑移植自骨架（只读参考，不 include）：
//   make_routes            empty_kernel_bench.cu:2636-2674
//   validate_route         :2679-2753
//   validate_publish       :2759-2801
//   validate_task_setup    :2808-2870
// ============================================================================
#pragma once

#include <mcr/mc_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "common.cuh"
#include "host_layout.cuh"

enum class RoutePattern { Balanced, Hot, MixedInvalid };

static const char* pattern_name(RoutePattern p) {
    switch (p) {
        case RoutePattern::Balanced: return "balanced";
        case RoutePattern::Hot: return "hot";
        case RoutePattern::MixedInvalid: return "mixed-invalid";
    }
    return "?";
}

// 三形态路由（TPR=16 基准：balanced 每专家恰好 16 对，与 R 无关）
static std::vector<int64_t> make_routes(RoutePattern pat, int rank, int R,
                                        int tokens, int salt = 0) {
    const int E = R * kEPR;
    const int n = tokens * kTopk;
    std::vector<int64_t> out(n, 0);
    uint64_t s = 0x9E3779B97F4A7C15ull ^ (uint64_t)(rank + 1) * 0x1234567ull;
    s ^= (uint64_t)salt * 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < n; ++i) {
        switch (pat) {
            case RoutePattern::Balanced:
                // 偏移用 rank*kEPR（步长 8）：128 宽块均匀覆盖 E，每专家恰 16 个 src
                out[i] = (i + rank * kEPR + salt) % E;
                break;
            case RoutePattern::Hot:
                out[i] = (i % 4 != 3)
                             ? 0
                             : (1 + (i / 4 + rank * 3 + salt) % (E - 1));
                break;
            case RoutePattern::MixedInvalid:
                s = s * 6364136223846793005ull + 1442695040888963407ull;
                out[i] = ((int)((s >> 33) % 100) < 10)
                             ? -1
                             : (int)((s >> 20) % (uint64_t)E);
                break;
        }
    }
    return out;
}

// ROUTE 校验：src_index_table 各 (le,src) 行 + 本地 route_slot_counts
static bool validate_route(const ProdLayout& L, int R, int tokens, int rank,
                           const std::vector<std::vector<int64_t>>& routes,
                           const uint8_t* sym, std::string* detail,
                           const ranks_data* rd = nullptr) {
    const int E = R * kEPR;
    const int cap = rd != nullptr ? kMaxTPR : (int)L.slot_capacity;
    std::vector<uint32_t> tbl((size_t)kEPR * R * cap, 0);
    std::vector<int> rsc(E, -1);
    CHECK(mcSetDevice(rank));
    if (rd != nullptr)
        CHECK(mcMemcpy(tbl.data(), &rd->src_index_table[0][0][0][0],
                       tbl.size() * sizeof(uint32_t), mcMemcpyDeviceToHost));
    else
        CHECK(mcMemcpy(tbl.data(), sym + L.src_index_table_off,
                       tbl.size() * sizeof(uint32_t), mcMemcpyDeviceToHost));
    CHECK(mcMemcpy(rsc.data(), sym + L.route_slot_counts_off,
                   rsc.size() * sizeof(int), mcMemcpyDeviceToHost));
    std::vector<int> sent(E, 0);
    for (int64_t v : routes[rank])
        if ((uint64_t)v < (uint64_t)E) ++sent[(int)v];
    for (int e = 0; e < E; ++e) {
        const int want = std::min(sent[e], cap);
        if (rsc[e] != want) {
            if (detail)
                *detail = "route_slot_counts[" + std::to_string(e) + "]=" +
                          std::to_string(rsc[e]) + " want " + std::to_string(want);
            return false;
        }
    }
    for (int src = 0; src < R; ++src) {
        std::vector<std::vector<int64_t>> expect(kEPR);
        for (int i = 0; i < (int)routes[src].size(); ++i) {
            const int64_t v = routes[src][i];
            if ((uint64_t)v >= (uint64_t)E) continue;
            if ((int)(v / kEPR) != rank) continue;
            expect[(int)(v % kEPR)].push_back(i);
        }
        for (int le = 0; le < kEPR; ++le) {
            const uint32_t* row = tbl.data() + ((size_t)le * R + src) * cap;
            const int ne = std::min((int)expect[le].size(), cap);
            std::vector<uint32_t> seen;
            for (int k = 0; k < ne; ++k) {
                const uint32_t v = row[k];
                if (std::find(seen.begin(), seen.end(), v) != seen.end()) {
                    if (detail)
                        *detail = "row 重复值 le=" + std::to_string(le) +
                                  " src=" + std::to_string(src);
                    return false;
                }
                seen.push_back(v);
                bool found = false;
                for (int64_t e : expect[le])
                    if ((int64_t)v == e) { found = true; break; }
                if (!found) {
                    if (detail)
                        *detail = "row 非期望值 le=" + std::to_string(le) +
                                  " src=" + std::to_string(src) + " v=" +
                                  std::to_string(v);
                    return false;
                }
            }
            for (int k = ne; k < cap; ++k) {
                if (row[k] != 0) {
                    if (detail)
                        *detail = "row 越界写入 le=" + std::to_string(le) +
                                  " src=" + std::to_string(src);
                    return false;
                }
            }
        }
    }
    return true;
}

// PUBLISH 校验：expert_recv_counts 行 + 行尾校验槽 ~(Σ+epoch)
static bool validate_publish(const ProdLayout& L, int R, int tokens, int rank,
                             const std::vector<std::vector<int64_t>>& routes,
                             const uint8_t* sym, unsigned epoch,
                             std::string* detail, int recv_stride,
                             long long recv_off, const ranks_data* rd = nullptr) {
    const int E = R * kEPR;
    const int cap = (int)L.slot_capacity;
    if (recv_off < 0) recv_off = L.expert_recv_counts_off;
    std::vector<uint32_t> recv((size_t)R * recv_stride, 0);
    CHECK(mcSetDevice(rank));
    if (rd != nullptr)
        CHECK(mcMemcpy(recv.data(), &rd->expert_recv_counts[0][0][0],
                       recv.size() * sizeof(uint32_t), mcMemcpyDeviceToHost));
    else
        CHECK(mcMemcpy(recv.data(), sym + recv_off,
                       recv.size() * sizeof(uint32_t), mcMemcpyDeviceToHost));
    for (int src = 0; src < R; ++src) {
        std::vector<int> sent(E, 0);
        for (int64_t v : routes[src])
            if ((uint64_t)v < (uint64_t)E) ++sent[(int)v];
        uint32_t sum = 0;
        for (int le = 0; le < kEPR; ++le) {
            const int want = std::min(sent[rank * kEPR + le], cap);
            if (recv[(size_t)src * recv_stride + le] != (uint32_t)want) {
                if (detail)
                    *detail = "recv[src=" + std::to_string(src) + "][le=" +
                              std::to_string(le) + "]=" +
                              std::to_string(recv[(size_t)src * recv_stride + le]) +
                              " want " + std::to_string(want);
                return false;
            }
            sum += (uint32_t)want;
        }
        const uint32_t csum = recv[(size_t)src * recv_stride + kEPR];
        if (static_cast<uint32_t>(sum + epoch) != ~csum) {
            if (detail)
                *detail = "checksum[src=" + std::to_string(src) + "] 不满足 Σ+epoch==~csum";
            return false;
        }
    }
    return true;
}

// TASK_SETUP 校验：prefix_tbl + 每 block 窗口数 + task_ready 推进
static bool validate_task_setup(
        const ProdLayout& L, int R, int rank,
        const std::vector<std::vector<int64_t>>& routes, const uint8_t* sym,
        const uint32_t* bar_flags, const int* ts_out_cpu, std::string* detail) {
    const int cap = (int)L.win_capacity;
    const int E = R * kEPR;
    std::vector<int> cnt(kEPR, 0);
    for (int src = 0; src < R; ++src) {
        std::vector<int> sent(kEPR, 0);
        for (int64_t v : routes[src])
            if ((uint64_t)v < (uint64_t)E && (int)(v / kEPR) == rank)
                sent[(int)(v % kEPR)] += 1;
        for (int le = 0; le < kEPR; ++le) cnt[le] += std::min(sent[le], cap);
    }
    int total = 0;
    for (int le = 0; le < kEPR; ++le) total += (cnt[le] + cap - 1) / cap;
    std::vector<int> prefix((size_t)kEPR * (R + 1), -1);
    CHECK(mcSetDevice(rank));
    CHECK(mcMemcpy(prefix.data(), sym + L.prefix_tbl_off,
                   prefix.size() * sizeof(int), mcMemcpyDeviceToHost));
    for (int le = 0; le < kEPR; ++le) {
        int c = 0;
        for (int r = 0; r < R; ++r) {
            if (prefix[le * (R + 1) + r] != c) {
                if (detail)
                    *detail = "prefix[" + std::to_string(le) + "][" +
                              std::to_string(r) + "]=" +
                              std::to_string(prefix[le * (R + 1) + r]) +
                              " want " + std::to_string(c);
                return false;
            }
            int inc = 0;
            for (int64_t v : routes[r])
                if ((uint64_t)v == (uint64_t)(rank * kEPR + le)) ++inc;
            c += std::min(inc, cap);
        }
        if (prefix[le * (R + 1) + R] != cnt[le]) {
            if (detail)
                *detail = "prefix[" + std::to_string(le) + "] 行尾=" +
                          std::to_string(prefix[le * (R + 1) + R]) + " want " +
                          std::to_string(cnt[le]);
            return false;
        }
    }
    for (int b = 0; b < kGrid; ++b) {
        if (ts_out_cpu[b] != total) {
            if (detail)
                *detail = "block" + std::to_string(b) + " total=" +
                          std::to_string(ts_out_cpu[b]) + " want " +
                          std::to_string(total);
            return false;
        }
    }
    return true;
}

// ############################################################################
// # M2-M3 数值正确性参考（host 纯函数）—— 本段自包含，可整段抽出单独编译    #
// ############################################################################
//
// 设计口径：参考与设备共用同一份 FP8 输入字节，把误差源隔离在被测相位本身。
//   * GEMM 一律 FP32 顺序累加；硬件 MMA 求和次序/指令级舍入与参考不同，
//     相对差约 1e-6 量级，远小于各相位判定阈值。
//   * L1 的 FP8 量化由硬件决定：参考输出"量化前的 FP32 期望"，比对时把设备
//     字节解码回 FP32 再用容差判定（E4M3 半 ulp 相对上界 2^-4 ≈ 6.25%）。
//   * 性能不重要（校验用），清晰正确优先。
//
// 维度约定（scene：H=kHidden=6144、I=kIntermediate=2048、EPR=8）：
//   * x / pool 槽          : [count][H] FP8(E4M3) 行主序（DISPATCH 已汇聚）
//   * W1 [EPR][2I][H]      : lane 序 packed tile（与 fill_w_lane 同字节布局）
//   * ffn_hidden / L1 输出 : [count][I] FP8(E4M3) 行主序
//   * W2 [EPR][H][I]       : lane 序 packed tile
//   * l2_out               : [grid][win_cap][H] BF16 行主序
//   * combine_rows         : [topk][combine_cap][H] BF16 行主序
//   * y                    : [tokens][H] FP32
//
// 16B/lane 配对 tile（src/compute.cuh 契约，W_e 为逻辑 n_rows×k_cols 矩阵）：
//   supertile 基址 = W_e + ((row/32)*kIters + (k/32))*1024，kIters = k_cols/32
//   supertile 内 64 lane × 16B：lane L 的低 8B 覆盖逻辑行 (row/32)*32 + L%16、
//   高 8B 覆盖逻辑行 +16（同一 K 段 [(L/16)*8, +8)）；
//   即 L = (row%16) + ((k%32)/8)*16，字节偏移 = L*16 + ((row/16)&1)*8 + k%8。
//
// L1 的 gate/up 配对（src/compute.cuh 契约，配对块宽固定 256）：
//   隐藏列 h 的 gate 逻辑行 = (h/256)*512 + h%256，up 行 = gate 行 + 256。
//
// ------------------------------------------------------------ 集成说明 ----
// 主控（bench.cu）在"单 launch + mcDeviceSynchronize"之后 dump 设备数据：
//   (a) pool 槽（L1 的 A）[count][H]：
//         src = r.sym + L.expert_tokens_off
//               + (size_t)(e*kRing + w%kRing) * pool_stride * H
//       pool_stride = win_cap + win_pad；行 stride = H；
//       count = min(该专家总行数 - w*win_cap, win_cap)（可用设备 dump 的 count）
//   (b) ffn_hidden 槽（L1 输出 / L2 的 A）[count][I]：
//         src = r.ffn_hidden + (size_t)(e*kRing + w%kRing) * win_cap * I
//   (c) l2_out 块槽（L2 输出 BF16）[win_cap][H]：
//         src = r.l2_out + (size_t)block * win_cap * H
//   (d) combine_rows [kTopk][max_tokens][H] BF16 + combine_sums（M3 扩展区）
//   (e) topk_idx / topk_weights：sym 视图 L.topk_idx_off / L.topk_weights_off
//   W1/W2 必须用生成设备权重时的同一份 host vector（fill_w_lane 产物），
//   不要在校验时重新生成（即使种子相同也无必要，直接复用最可靠）。
//   传入设备 pool 槽（a）时，L1 校验只覆盖 L1+SwiGLU+量化（隔离 DISPATCH）；
//   若要做 DISPATCH+L1 端到端，把 ref_l1 的 x 换成"host 按 src_index_table
//   重建的期望 pool 行"，此时 L1 通过即同时覆盖取行映射。
//
// 建议调用顺序与判定（阈值可按实测收紧；量化以"解码后比 FP32 期望"为准）：
//   1) validate_route / validate_publish / validate_task_setup（已有）
//   2) L1：std::vector<float> ffn_ref((size_t)count * I);
//          ref_l1(pool_host, w1_host.data(), e, count, H, I, ffn_ref.data());
//          CheckStats s1 = compare_fp8_rows(ffn_dev, ffn_ref.data(), count,
//                                           I, I, 0, I, 7e-2, 2e-3);
//        （7e-2 ≈ E4M3 半 ulp 相对上界 2^-4；2e-3 > 次正规半 ulp 2^-10）
//   3) L2：std::vector<float> out_ref((size_t)count * H);
//          ref_l2(ffn_dev, w2_host.data(), e, count, I, H, out_ref.data());
//          const int c0 = slice * (H / kNSlice);          // 本 block 只写本片
//          CheckStats s2 = compare_bf16_rows(l2_block, out_ref.data(), count,
//                                            H, H, c0, c0 + H / kNSlice,
//                                            1e-2, 1e-3);
//        （1e-2 覆盖 BF16 输出舍入 2^-9 与两侧 FP32 累加序差）
//   4) COMBINE 抽查：l2_block 有效行与目标 rank 的
//          combine_rows[(k*combine_cap + src_token)*H] 逐字节相等（原样搬运）
//   5) REDUCE：ref_reduce(combine_rows_rank, topk_idx_host, weights_host,
//                         token, kTopk, R*kEPR, max_tokens, H, y_ref.data());
//          CheckStats s3 = compare_f32(y_dev, y_ref.data(), H, 1e-5, 1e-6);
//        （两侧同一 BF16 输入，只有累加序差；RNE 权重解码一致）
//
// 场景注意：
//   * balanced TPR=16 每专家恰好 1 个窗口，槽内容即最终态；hot/mixed 多窗口时
//     ring 槽会被后续窗口覆盖，只能核验"每专家最后一个窗口"。
//   * 当前 fill_w_lane / x 填充只用 0x00..0x3F（正有限 E4M3）；本段参考仍按
//     IEEE 语义处理负数与 NaN：0x7F/0xFF = NaN，0x80 = -0.0。若设备/参考任一侧
//     出现 NaN，比较函数把"单侧 NaN"判为 bad，两侧 NaN 视为一致。
//   * 设备 E4M3 转换的精确舍入/饱和字节不作保证；主判据用解码+容差。若要字节级
//     对照：先用 ref_quantize_fp8() 生成期望字节，允许 ±1 ulp（不推荐作主判据）。

// ------------------------------------------------------------ E4M3 编解码 --
// E4M3（OCP FP8）：b7=符号、b6..b3=指数（偏置 7）、b2..b0=尾数。
//   exp=0    : ±(m/8)·2^-6（次正规，含 ±0；即 ±m·2^-9）
//   exp=1..15: ±(1+m/8)·2^(exp-7)（最大有限值 448 = 0x7E）
//   0x7F/0xFF: NaN（E4M3 无 inf）
namespace bench_check_num {
struct E4M3DecodeTable {
    float v[256];
    constexpr E4M3DecodeTable() : v{} {
        for (int i = 0; i < 256; ++i) {
            const int sign = (i >> 7) & 1;
            const int exp = (i >> 3) & 0xF;
            const int man = i & 7;
            if (exp == 0xF && man == 7) {          // 0x7F / 0xFF
                v[i] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            // 尾数部分；再乘 2^p（p 为整数，乘除 2 精确，constexpr 可算）
            float a = (exp == 0) ? (float)man / 8.0f
                                 : 1.0f + (float)man / 8.0f;
            int p = (exp == 0) ? -6 : exp - 7;
            while (p > 0) { a *= 2.0f; --p; }
            while (p < 0) { a *= 0.5f; ++p; }
            v[i] = sign ? -a : a;
        }
    }
};
}  // namespace bench_check_num

inline constexpr bench_check_num::E4M3DecodeTable kE4M3Decode{};

// 1 字节 E4M3 -> float（查表；含次正规；0x7F/0xFF = NaN）
static inline float fp8_e4m3_decode(uint8_t b) { return kE4M3Decode.v[b]; }

// float -> E4M3 字节（round-to-nearest-even，饱和到 ±448，NaN -> 0x7F）。
// 仅用于"期望字节"对照；设备转换的边界/饱和字节不保证与此完全一致。
static inline uint8_t fp8_e4m3_quantize(float v) {
    if (std::isnan(v)) return 0x7Fu;
    const uint32_t sign = std::signbit(v) ? 0x80u : 0u;
    const float a = std::fabs(v);
    if (a >= 448.0f) return (uint8_t)(sign | 0x7Eu);       // 饱和（含 +inf）
    if (a < 1.0f / 64.0f) {                                // < 2^-6：次正规
        const int q = (int)std::nearbyint(a * 512.0f);     // 步长 2^-9，RNE
        return (uint8_t)(sign | (uint32_t)q);
    }
    int e2 = 0;
    const float mf = std::frexp(a, &e2);                   // a = mf·2^e2
    const float mant = mf * 2.0f;                          // [1,2)
    const int eu = e2 - 1;                                 // a = mant·2^eu
    int q = (int)std::nearbyint(mant * 8.0f);              // [8,16]
    int enc_e = eu + 7;
    if (q == 16) { q = 8; enc_e += 1; }                    // 进位
    if (enc_e > 15) return (uint8_t)(sign | 0x7Eu);        // 进位后溢出：饱和
    return (uint8_t)(sign | ((uint32_t)enc_e << 3) | (uint32_t)(q - 8));
}

// -------------------------------------------------------------- BF16 解码 --
// BF16 = FP32 高 16 位；解码即左移 16 位（与 combine.cuh 设备端同构）。
static inline float bf16_to_f32(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float v;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}

// float -> BF16（RNE；仅供期望字节/值对照，设备 cm_pack_f2bf16 期望同舍入）
static inline uint16_t f32_to_bf16_rne(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    if ((u & 0x7F800000u) == 0x7F800000u)   // inf/NaN：保留高位
        return (uint16_t)(u >> 16);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;                      // RNE（就近偶数）
    return (uint16_t)(u >> 16);
}

// -------------------------------------------------------- lane 序 tile 读取 --
// W_e 基址（该 expert 的逻辑 [n_rows][k_cols] 16B/lane 配对 packed tile），
// 返回逻辑 (row, k) 处的 FP8 字节。只解码 host 生成的字节布局，与设备实现独立。
static inline uint8_t w_lane_get(const uint8_t* w_e, int row, int k,
                                 int k_cols) {
    const int k_iters = k_cols / 32;
    const uint8_t* tile =
        w_e + ((size_t)(row >> 5) * (size_t)k_iters + (size_t)(k >> 5)) * 1024;
    const int lane = (row & 15) + (((k & 31) >> 3) << 4);
    const int sub = (row >> 4) & 1;
    return tile[(size_t)lane * 16 + (size_t)sub * 8 + (size_t)(k & 7)];
}

// ------------------------------------------------------------------ L1 参考 --
// x        : [count][H] FP8 行主序（建议直接 dump 设备 pool 槽，隔离 DISPATCH）
// w1       : 整张 [EPR][2I][H] lane 序 tensor 基址（与设备同一份 host 字节）
// expert   : 本地专家号 [0, EPR)
// 输出     : ffn_expect [count][I] FP32 = silu(gate)·up（FP8 量化前期望）
static inline void ref_l1(const uint8_t* x, const uint8_t* w1, int expert,
                          int count, int H, int I, float* ffn_expect) {
    const uint8_t* w_e = w1 + (size_t)expert * (size_t)(2 * I) * (size_t)H;
    std::vector<float> xd((size_t)H);
    for (int r = 0; r < count; ++r) {
        const uint8_t* xr = x + (size_t)r * H;
        for (int k = 0; k < H; ++k) xd[(size_t)k] = fp8_e4m3_decode(xr[k]);
        float* fr = ffn_expect + (size_t)r * I;
        for (int h = 0; h < I; ++h) {
            const int grow = (h / 256) * 512 + (h % 256);  // gate 逻辑行
            const int urow = grow + 256;                   // up 逻辑行
            float g = 0.0f, u = 0.0f;
            for (int k = 0; k < H; ++k) {
                const float xv = xd[(size_t)k];
                g += xv * fp8_e4m3_decode(w_lane_get(w_e, grow, k, H));
                u += xv * fp8_e4m3_decode(w_lane_get(w_e, urow, k, H));
            }
            fr[h] = (g / (1.0f + std::exp(-g))) * u;       // silu(g)·u
        }
    }
}

// ------------------------------------------------------------------ L2 参考 --
// ffn     : [count][I] FP8 行主序（建议 dump 设备 ffn_hidden 槽）
// w2      : 整张 [EPR][H][I] lane 序 tensor 基址
// 输出    : out_expect [count][H] FP32（BF16 量化前期望）
static inline void ref_l2(const uint8_t* ffn, const uint8_t* w2, int expert,
                          int count, int I, int H, float* out_expect) {
    const uint8_t* w_e = w2 + (size_t)expert * (size_t)H * (size_t)I;
    std::vector<float> ad((size_t)I);
    for (int r = 0; r < count; ++r) {
        const uint8_t* ar = ffn + (size_t)r * I;
        for (int k = 0; k < I; ++k) ad[(size_t)k] = fp8_e4m3_decode(ar[k]);
        float* orow = out_expect + (size_t)r * H;
        for (int n = 0; n < H; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < I; ++k)
                acc += ad[(size_t)k] * fp8_e4m3_decode(w_lane_get(w_e, n, k, I));
            orow[n] = acc;
        }
    }
}

// --------------------------------------------------------------- REDUCE 参考 --
// combine_rows : 本 rank [topk][combine_cap][H] BF16 行主序（设备 dump）
// topk_idx     : 本 rank 输入 [tokens][topk]（设备 topk_idx 视图 host 拷贝）
// weights      : 本 rank 输入 [tokens][topk] FP32
// E            : 总专家数（= R*kEPR，设备有效性判据 0 <= e < E）
// 输出         : y_expect [H] FP32（token 越界时清零返回）
static inline void ref_reduce(const uint8_t* combine_rows,
                              const int64_t* topk_idx, const float* weights,
                              int token, int topk, int E, int combine_cap,
                              int H, float* y_expect) {
    for (int c = 0; c < H; ++c) y_expect[c] = 0.0f;
    if (token < 0 || token >= combine_cap) return;
    const uint16_t* rows_h = reinterpret_cast<const uint16_t*>(combine_rows);
    for (int k = 0; k < topk; ++k) {
        const int64_t e = topk_idx[(size_t)token * topk + k];
        if (e < 0 || e >= (int64_t)E) continue;            // 无效 k：同设备跳过
        const float wv = weights[(size_t)token * topk + k];
        const uint16_t* row = rows_h + ((size_t)k * combine_cap + token) * H;
        for (int c = 0; c < H; ++c)
            y_expect[c] += bf16_to_f32(row[c]) * wv;
    }
}

// 参考期望批量量化为 E4M3 字节（长度 n 的一维缓冲；如 n = count*I）
static inline void ref_quantize_fp8(const float* expect, long long n,
                                    uint8_t* out) {
    for (long long i = 0; i < n; ++i) out[i] = fp8_e4m3_quantize(expect[i]);
}

// ------------------------------------------------------------ 容差比较工具 --
// 判定：|got - want| <= max(abs_tol, rel_tol·|want|)；NaN 双侧视为一致、单侧
// 判 bad；统计最大绝对/相对差与最差下标（最长绝对差处），供日志定位。
struct CheckStats {
    long long n = 0;          // 参与比较元素数
    long long bad = 0;        // 超差元素数（含单侧 NaN）
    long long nan = 0;        // 双侧 NaN（视为通过）
    double max_abs = 0.0;     // 最大绝对误差
    double max_rel = 0.0;     // 最大相对误差（分母 max(|want|, 1e-30)）
    long long worst = -1;     // 最大绝对误差下标（行内展平）
    float worst_got = 0.0f;   // 最差处实测值
    float worst_want = 0.0f;  // 最差处期望值
    bool pass() const { return bad == 0; }
};

static inline void check_update(CheckStats& s, float got, float want,
                                long long idx, double rel_tol, double abs_tol) {
    ++s.n;
    const bool gn = std::isnan(got), wn = std::isnan(want);
    if (gn || wn) {
        if (gn && wn) { ++s.nan; return; }
        ++s.bad;
        if (s.worst < 0) {
            s.worst = idx;
            s.worst_got = got;
            s.worst_want = want;
        }
        return;
    }
    const double d = std::fabs((double)got - (double)want);
    const double rel =
        d / std::max(1e-30, std::fabs((double)want));
    if (d > s.max_abs) {
        s.max_abs = d;
        s.worst = idx;
        s.worst_got = got;
        s.worst_want = want;
    }
    if (rel > s.max_rel) s.max_rel = rel;
    if (d > std::max(abs_tol, rel_tol * std::fabs((double)want))) ++s.bad;
}

enum class RefNumKind { kF32, kFp8, kBf16 };

// 行主序缓冲比较核心：[rows][stride] 取每行列区间 [c0, c1)。
// got_stride / want_stride 均以各自元素类型计（F32/FP8/BF16）。
static inline CheckStats compare_rows_impl(RefNumKind kind, const void* got,
                                           const float* want, int rows,
                                           int got_stride, int want_stride,
                                           int c0, int c1, double rel_tol,
                                           double abs_tol) {
    CheckStats s;
    for (int r = 0; r < rows; ++r) {
        const float* wr = want + (size_t)r * want_stride;
        for (int c = c0; c < c1; ++c) {
            const long long idx = (long long)r * (c1 - c0) + (c - c0);
            float g;
            if (kind == RefNumKind::kF32)
                g = reinterpret_cast<const float*>(got)[(size_t)r * got_stride + c];
            else if (kind == RefNumKind::kFp8)
                g = fp8_e4m3_decode(
                    reinterpret_cast<const uint8_t*>(got)[(size_t)r * got_stride + c]);
            else
                g = bf16_to_f32(
                    reinterpret_cast<const uint16_t*>(got)[(size_t)r * got_stride + c]);
            check_update(s, g, wr[c], idx, rel_tol, abs_tol);
        }
    }
    return s;
}

// 各类型便捷包装（c0/c1 为行内列区间；整块比较用 0..cols）
static inline CheckStats compare_f32_rows(const float* got, const float* want,
                                          int rows, int got_stride,
                                          int want_stride, int c0, int c1,
                                          double rel_tol, double abs_tol) {
    return compare_rows_impl(RefNumKind::kF32, got, want, rows, got_stride,
                             want_stride, c0, c1, rel_tol, abs_tol);
}
static inline CheckStats compare_fp8_rows(const uint8_t* got, const float* want,
                                          int rows, int got_stride,
                                          int want_stride, int c0, int c1,
                                          double rel_tol, double abs_tol) {
    return compare_rows_impl(RefNumKind::kFp8, got, want, rows, got_stride,
                             want_stride, c0, c1, rel_tol, abs_tol);
}
static inline CheckStats compare_bf16_rows(const uint16_t* got,
                                           const float* want, int rows,
                                           int got_stride, int want_stride,
                                           int c0, int c1, double rel_tol,
                                           double abs_tol) {
    return compare_rows_impl(RefNumKind::kBf16, got, want, rows, got_stride,
                             want_stride, c0, c1, rel_tol, abs_tol);
}

// 一维连续缓冲便捷包装（rows=1、整区间）
static inline CheckStats compare_f32(const float* got, const float* want, int n,
                                     double rel_tol, double abs_tol) {
    return compare_f32_rows(got, want, 1, n, n, 0, n, rel_tol, abs_tol);
}
static inline CheckStats compare_fp8(const uint8_t* got, const float* want,
                                     int n, double rel_tol, double abs_tol) {
    return compare_fp8_rows(got, want, 1, n, n, 0, n, rel_tol, abs_tol);
}
static inline CheckStats compare_bf16(const uint16_t* got, const float* want,
                                      int n, double rel_tol, double abs_tol) {
    return compare_bf16_rows(got, want, 1, n, n, 0, n, rel_tol, abs_tol);
}

// 单行日志（主控可直接打印判定）
static inline void print_check_stats(const char* tag, const CheckStats& s) {
    std::printf("[数值校验] %-28s n=%-7lld bad=%-6lld max_abs=%.3e "
                "max_rel=%.3e %s\n",
                tag, s.n, s.bad, s.max_abs, s.max_rel,
                s.pass() ? "PASS" : "FAIL");
    if (!s.pass())
        std::printf("           最差 idx=%lld got=%.6g want=%.6g\n", s.worst,
                    (double)s.worst_got, (double)s.worst_want);
}
