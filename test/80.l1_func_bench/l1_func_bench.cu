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
    const int Mx=128, mt=(Mx+15)/16;
    const size_t poolB=(size_t)Mx*kH, w1B=(size_t)kEPR*kN*kH, ffnB=(size_t)Mx*kI, fragB=(size_t)(kH/32)*mt*512;
    uint8_t *pool=nullptr,*w1=nullptr,*ffn=nullptr,*frag=nullptr;
    mcMalloc((void**)&pool,poolB); mcMalloc((void**)&w1,w1B); mcMalloc((void**)&ffn,ffnB); mcMalloc((void**)&frag,fragB);
    mcMemset(pool,0x3c,poolB); mcMemset(w1,0x3c,w1B); mcMemset(frag,0x3c,fragB);
    const char* vn[9]={"paired<1>","paired<2>","r24<3,16>","r24<3,16>","r24<4,8>","r24<4,8>","r24<8,8>","r24<8,8>","r24<8,8>"};
    const int Ms[]={16,17,32,33,48,64,65,96,128};
    printf("%-6s %-8s %-14s %-10s %-10s\n","M","mt","variant","frag","rowmajor");
    for(int M:Ms){int m=(M+15)/16;
      auto Lf=[&](mcStream_t st){ k_disp<<<32,512,65536,st>>>(nullptr,pool,w1,ffn,frag,M,mt,true);};
      auto Lr=[&](mcStream_t st){ k_disp<<<32,512,65536,st>>>(nullptr,pool,w1,ffn,frag,M,mt,false);};
      int idx = M<=16?0 : M<=32?1 : M<=48?2 : M<=64?4 : 6;
      printf("%-6d %-8d %-14s %-10.2f %-10.2f\n",M,m,vn[idx],timeit(Lf,50),timeit(Lr,50));}
    return 0;
}
