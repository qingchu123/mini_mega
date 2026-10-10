// 80.l1_func_bench — 按真实分派逐 M 调对应 L1 body（真实函数），计时。
#include <mcr/mc_runtime.h>
#include <cstdio>
#include <cstdint>
#include "common.cuh"
#include "flags_defs.cuh"
#include "compute.cuh"
#include "compute_big.cuh"

constexpr int kI=2048, kH=6144, kN=4096, kBN=256;
constexpr int kGL1=(kN/kBN)/kNSlice;   // 4
constexpr int kR23_BATCH=4;

// 真实 L1 分派（g_l1=4, frag 可用）：
//  count<=16 : paired<2,2,1,4>
//  <=32      : paired<2,2,2,4>
//  <=48      : r24<3,16,B4,true,3>
//  <=64      : r24<4,8,B4,false,4>
//  >64       : r24<8,8,B2,false,8>
template<bool FRAG>
__device__ __forceinline__ void dispatch_l1(uint8_t* smem,const uint8_t* pool,const uint8_t* w1,
                                            uint8_t* ffn,const uint8_t* frag,int count,int mt_total,
                                            int lane,int warp,int nwarps){
    const uint8_t* fb = FRAG?frag:nullptr;
    if (count<=16)
        l1_gemm_swiglu_body_paired<2,2,1,4,FRAG>(pool,w1,0,count,kH,kN,kBN,lane,warp,nwarps,
            nullptr,0,kGL1,ffn,0,kI,nullptr,fb,mt_total,nullptr);
    else if (count<=32)
        l1_gemm_swiglu_body_paired<2,2,2,4,FRAG>(pool,w1,0,count,kH,kN,kBN,lane,warp,nwarps,
            nullptr,0,kGL1,ffn,0,kI,nullptr,fb,mt_total,nullptr);
    else if (count<=48 && FRAG)
        r24_l1_big<3,16,4,true,3,2>(smem,fb,mt_total,w1,0,0,count,kH,kN,kBN,lane,warp,nwarps,
            nullptr,0,kGL1,ffn,nullptr,0,kI);
    else if (count<=48)
        l1_gemm_swiglu_body_paired<2,2,3,4,false>(pool,w1,0,count,kH,kN,kBN,lane,warp,nwarps,
            nullptr,0,kGL1,ffn,0,kI,nullptr,nullptr,mt_total,nullptr);
    else if (count<=64 && FRAG)
        r24_l1_big<4,8,4,false,4,2>(smem,fb,mt_total,w1,0,0,count,kH,kN,kBN,lane,warp,nwarps,
            nullptr,0,kGL1,ffn,nullptr,0,kI);
    else if (count>64 && FRAG)
        r24_l1_big<8,8,2,false,8,2>(smem,fb,mt_total,w1,0,0,count,kH,kN,kBN,lane,warp,nwarps,
            nullptr,0,kGL1,ffn,nullptr,0,kI);
    else
        r19_l1_big<4,2,4>(pool,fb,mt_total,w1,0,count,kH,kN,kBN,lane,warp,nwarps,
            nullptr,0,kGL1,ffn,nullptr,0,kI);
}

// [EXP] r24 kBatch=1 两遍：gate/up 各跑一次完整 K 循环（A 两次 bsm+lds），
//   gate acc 暂存 smem。acc 仅 kMT*kNT*4=128（MT16 不再溢出）。
//   量化"两次 K 循环"相对 kBatch=2（一次 K 循环共享 A）的额外开销。
template<int kMT,int kKS,int kMTAct,int kBPF>
__device__ __forceinline__ void r24_l1_big_b1(
    uint8_t* s_mem,float* s_gate,const uint8_t* fbase,int mt_total,
    const uint8_t* w1,int expert,int c0_begin,int c0_end,int k_hidden,
    int n_total,int block_n,int lane,int warp,int nwarps,
    int n_group_begin,int n_group_count,uint8_t* ffn_out,uint8_t* ffn2,
    int out_col_base,int k_out){
    constexpr int kKStep=32,kNT=2,kMask=true;
    constexpr int kBufBytes=kMT*kKS*512;
    const int k_iters=k_hidden/kKStep;
    const int nstage=k_iters/kKS;
    const int n_per_warp=block_n/nwarps;
    const uint8_t* w_e=w1+(size_t)expert*n_total*k_hidden;
    float* gbuf=s_gate+warp*(kMT*kNT*4);
    for(int c0=c0_begin;c0<c0_end;c0+=kMT*kM){
        const int cc=(c0_end-c0<kMT*kM)?(c0_end-c0):kMT*kM;
        const int m_abs0=c0>>4;
        uint8_t* f_c=(ffn_out!=nullptr)?ffn_out+(size_t)c0*k_out:nullptr;
        for(int g0=n_group_begin;g0+1<n_group_begin+n_group_count;g0+=2){
            const int nbase_g=g0*block_n+warp*n_per_warp;
            const int nbase_u=(g0+1)*block_n+warp*n_per_warp;
            R19Acc acc[kMT][kNT];
            R19Frag2 b_pair[kBPF];
            #pragma unroll
            for(int pass=0;pass<2;++pass){
                const int nbase=pass?nbase_u:nbase_g;
                #pragma unroll
                for(int m=0;m<kMT;++m){ if(kMask&&m>=kMTAct) continue;
                    #pragma unroll
                    for(int t=0;t<kNT;++t) acc[m][t]=R19Acc{0.f,0.f,0.f,0.f}; }
                r24_issue_a<kMT,kKS,kMask,kMTAct>(s_mem,fbase,0,mt_total,m_abs0);
                #pragma unroll
                for(int sp=0;sp<kBPF;++sp){ if(sp>=k_iters) break;
                    b_pair[sp]=R19_LDB(w_e,nbase,sp*kKStep,k_hidden); }
                for(int st=0;st<nstage;++st){
                    R24_WAIT_BSM(); __syncthreads();
                    if(st+1<nstage) r24_issue_a<kMT,kKS,kMask,kMTAct>(
                        s_mem+((st+1)&1)*kBufBytes,fbase,(st+1)*kKS,mt_total,m_abs0);
                    const uint8_t* sb=s_mem+(st&1)*kBufBytes;
                    #pragma unroll
                    for(int j=0;j<kKS;++j){
                        const int g=st*kKS+j; const int sp=g&(kBPF-1);
                        R19Frag a_pipe[kMT];
                        #pragma unroll
                        for(int m=0;m<kMT;++m){ if(kMask&&m>=kMTAct) continue;
                            a_pipe[m]=r24_lds_a<kMT>(sb,j,m); }
                        #pragma unroll
                        for(int m=0;m<kMT;++m){ if(kMask&&m>=kMTAct) continue;
                            const R19Frag2 bp=b_pair[sp];
                            acc[m][0]=__builtin_mxc_mma_16x16x32f8(a_pipe[m],R19Frag{bp[0],bp[1]},acc[m][0]);
                            acc[m][1]=__builtin_mxc_mma_16x16x32f8(a_pipe[m],R19Frag{bp[2],bp[3]},acc[m][1]); }
                        const int nk=g+kBPF;
                        if(nk<k_iters) b_pair[sp]=R19_LDB(w_e,nbase,nk*kKStep,k_hidden);
                    }
                }
                if(pass==0){
                    #pragma unroll
                    for(int m=0;m<kMT;++m){ if(kMask&&m>=kMTAct) continue;
                        #pragma unroll
                        for(int t=0;t<kNT;++t)
                            #pragma unroll
                            for(int i=0;i<4;++i) gbuf[(m*kNT+t)*4+i]=acc[m][t][i]; }
                    __syncthreads();
                }
            }
            #pragma unroll
            for(int m=0;m<kMT;++m){ if(kMask&&m>=kMTAct) continue;
                #pragma unroll
                for(int t=0;t<kNT;++t){
                    R19Acc sw;
                    #pragma unroll
                    for(int i=0;i<4;++i){ const float gv=gbuf[(m*kNT+t)*4+i];
                        const float uv=acc[m][t][i]; sw[i]=gv/(1.0f+__expf(-gv))*uv; }
                    const uint32_t packed=__builtin_mxc_cvt_pk4_f32tof8_cfg(sw,_MACA_FROUND_TO_NEAREST_EVEN,_MACA_DENORMAL_FLUSH_NONE,_MACA_SRC_NO_MODIFIER,false,true);
                    const int col=out_col_base+((g0-n_group_begin)>>1)*block_n+warp*n_per_warp+t*16+(lane&15);
                    uint8_t* f2p=(ffn2!=nullptr)?ffn2+((size_t)(col>>5)*mt_total+(m_abs0+m))*512+(lane>>4)*4*32+(col&31):nullptr;
                    #pragma unroll
                    for(int i=0;i<4;++i){ const int row=m*kM+(lane>>4)*4+i;
                        if(row<cc){ const uint8_t byte=(uint8_t)(packed>>(8*i));
                            if(f_c!=nullptr) f_c[(size_t)row*k_out+col]=byte;
                            if(f2p!=nullptr) f2p[i*32]=byte; } }
                }
            }
            __syncthreads();
        }
    }
}
template<int kMT,int kKS,int kMTAct,int kBPF>
__global__ void __launch_bounds__(512)
k_r24_b1(const uint8_t* fbase,const uint8_t* w1,int count,int mt_total,
         uint8_t* ffn){
    extern __shared__ __align__(16) uint8_t dyn[];
    const int lane=threadIdx.x&63,warp=threadIdx.x>>6;
    constexpr int kAbuf=2*kMT*kKS*512;
    float* sgate=reinterpret_cast<float*>(dyn+kAbuf);
    r24_l1_big_b1<kMT,kKS,kMTAct,kBPF>(dyn,sgate,fbase,mt_total,w1,0,
        0,count,kH,kN,kBN,lane,warp,8,0,kGL1,ffn,nullptr,0,kI);
}
__global__ void __launch_bounds__(512) k_disp(uint8_t* smem,const uint8_t* pool,const uint8_t* w1,
                                              uint8_t* ffn,const uint8_t* frag,int count,int mt_total,bool use_frag){
    extern __shared__ __align__(16) uint8_t dyn[];
    const int lane=threadIdx.x&63, warp=threadIdx.x>>6;
    if(use_frag) dispatch_l1<true>(dyn,pool,w1,ffn,frag,count,mt_total,lane,warp,8);
    else     dispatch_l1<false>(dyn,pool,w1,ffn,frag,count,mt_total,lane,warp,8);
    __syncthreads(); (void)smem;
}
template<class F> static double timeit(F launch,int REP){
    mcStream_t st=nullptr; mcStreamCreate(&st);
    launch(st); mcStreamSynchronize(st);
    mcEvent_t e0,e1; mcEventCreate(&e0); mcEventCreate(&e1); mcEventRecord(e0,st);
    for(int r=0;r<REP;++r) launch(st);
    mcEventRecord(e1,st); mcStreamSynchronize(st);
    float ms=0; mcEventElapsedTime(&ms,e0,e1);
    mcEventDestroy(e0);mcEventDestroy(e1);mcStreamDestroy(st);
    return (double)ms*1000.0/REP;
}
int main(){
    mcSetDevice(0);
    const int Mx=512, mt=(Mx+15)/16;
    const size_t poolB=(size_t)Mx*kH, w1B=(size_t)kEPR*kN*kH, ffnB=(size_t)Mx*kI, fragB=(size_t)(kH/32)*mt*512;
    uint8_t *pool=nullptr,*w1=nullptr,*ffn=nullptr,*frag=nullptr;
    mcMalloc((void**)&pool,poolB); mcMalloc((void**)&w1,w1B); mcMalloc((void**)&ffn,ffnB); mcMalloc((void**)&frag,fragB);
    mcMemset(pool,0x3c,poolB); mcMemset(w1,0x3c,w1B); mcMemset(frag,0x3c,fragB);
    const char* vn[9]={"paired<1>","paired<2>","r24<3,16>","r24<3,16>","r24<4,8>","r24<4,8>","r24<8,8>","r24<8,8>","r24<8,8>"};
    const int Ms[]={64,128,256,384,512};
    printf("%-6s %-8s %-14s %-10s %-10s\n","M","mt","variant","frag","rowmajor");
    for(int M:Ms){int m=(M+15)/16;
      auto Lf=[&](mcStream_t st){ k_disp<<<32,512,65536,st>>>(nullptr,pool,w1,ffn,frag,M,mt,true);};
      auto Lb=[&](mcStream_t st){ if(M<256){k_r24_b1<16,4,8,2><<<32,512,69632,st>>>(frag,w1,M,mt,ffn);}
                                  else {k_r24_b1<16,4,16,2><<<32,512,69632,st>>>(frag,w1,M,mt,ffn);} };
      auto Lr=[&](mcStream_t st){ k_disp<<<32,512,65536,st>>>(nullptr,pool,w1,ffn,frag,M,mt,false);};
      int idx = M<=16?0 : M<=32?1 : M<=48?2 : M<=64?4 : 6;
      printf("%-6d %-8d %-14s %-10.2f %-10.2f  b1(MT16,twopass)=%.2f\n",M,m,vn[idx],timeit(Lf,50),timeit(Lr,50),timeit(Lb,50));}
    return 0;
}
