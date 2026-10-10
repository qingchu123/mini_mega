// L1 形状 = 忠实 l1_gemm_swiglu_body_paired 的 装载:MMA 比例
// 单 warp 单 k-step: load 1 A(F1) + kBatch B(F2) ; MMA = 2*kBatch (kMT=1, kNT=2/F2)
#include <mcr/mc_runtime.h>
#include <cstdio>
#include <cstdint>
using F1=__NATIVE_VECTOR__(2,uint32_t); using F2=__NATIVE_VECTOR__(4,uint32_t);
using ACC=__NATIVE_VECTOR__(4,float);
constexpr int kK=6144,kKStep=32,kKI=kK/kKStep;      // 192
constexpr int kARows=16*8;                          // A: 128 行
constexpr int kNExp=64;                             // B: 64 个"专家"权重区

__device__ __forceinline__ F1 ldA(const uint8_t* gA,int lane,int m,int kk){
    const int row=m*16+(lane&15);
    return *reinterpret_cast<const F1*>(gA+(size_t)row*kK+kk*kKStep+((lane>>4)&3)*8);
}
__device__ __forceinline__ F2 ldB(const uint8_t* base,int lane,int q,int kk){
    const uint8_t* tile=base+((size_t)q*kKI+kk)*1024;   // 每 q 一个 32 列 supertile
    return *reinterpret_cast<const F2*>(tile+(size_t)lane*16);
}
template<int PF,int kBatch>
__global__ void __launch_bounds__(512) kern(int m_tiles,const uint8_t* gA,const uint8_t* gB,float* sink){
    const int lane=threadIdx.x&63, warp=threadIdx.x>>6;
    const size_t bExp=(size_t)kBatch*kKI*1024;          // 一个专家区
    ACC acc[kBatch][2];
    #pragma unroll
    for(int q=0;q<kBatch;++q){acc[q][0][0]=acc[q][0][1]=acc[q][0][2]=acc[q][0][3]=0.f;acc[q][1]=acc[q][0];}
    for(int m=0;m<m_tiles;++m){
      const uint8_t* B=gB+((size_t)(blockIdx.x%kNExp))*bExp;  // 每 block 一个专家区
      F1 ap[PF]; F2 bn[PF][kBatch];
      #pragma unroll
      for(int s=0;s<PF;++s){ ap[s]=ldA(gA,lane,m,s);
        #pragma unroll
        for(int q=0;q<kBatch;++q) bn[s][q]=ldB(B,lane,q,s); }
      for(int kk=0;kk<kKI;++kk){
        const int sp=kk%PF; F1 a=ap[sp];
        #pragma unroll
        for(int q=0;q<kBatch;++q){
          F2 bp=bn[sp][q]; F1 b0={bp[0],bp[1]}, b1={bp[2],bp[3]};
          acc[q][0]=__builtin_mxc_mma_16x16x32f8(a,b0,acc[q][0]);
          acc[q][1]=__builtin_mxc_mma_16x16x32f8(a,b1,acc[q][1]);
        }
        const int nk=kk+PF;
        if(nk<kKI){ ap[sp]=ldA(gA,lane,m,nk);
          #pragma unroll
          for(int q=0;q<kBatch;++q) bn[sp][q]=ldB(B,lane,q,nk); }
      }
    }
    float s=0.f;
    #pragma unroll
    for(int q=0;q<kBatch;++q) s+=acc[q][0][0]+acc[q][1][0];
    if(s==12345.0f)*sink=s;
}
template<int PF,int kBatch> static double bench(int M,int REP,const uint8_t* gA,const uint8_t* gB){
    const int mt=(M+15)/16; float* sink=nullptr; mcMalloc((void**)&sink,4);
    static uint8_t* flush=nullptr; const size_t FB=128ull<<20;
    if(!flush) mcMalloc((void**)&flush,FB);
    mcStream_t st=nullptr; mcStreamCreate(&st);
    kern<PF,kBatch><<<32,512,0,st>>>(mt,gA,gB,sink); mcStreamSynchronize(st);
    double tot=0;
    for(int r=0;r<REP;++r){
        mcMemset(flush,0x5a,FB);
        mcEvent_t e0,e1; mcEventCreate(&e0); mcEventCreate(&e1); mcEventRecord(e0,st);
        kern<PF,kBatch><<<32,512,0,st>>>(mt,gA,gB,sink);
        mcEventRecord(e1,st); mcStreamSynchronize(st);
        float ms=0; mcEventElapsedTime(&ms,e0,e1); tot+=ms*1000.0;
        mcEventDestroy(e0);mcEventDestroy(e1);
    }
    mcStreamDestroy(st);mcFree(sink); return tot/REP;
}
int main(){
    mcSetDevice(0); const int kB=4;
    uint8_t *gA=nullptr,*gB=nullptr;
    const size_t aB=(size_t)kARows*kK, bB=(size_t)kB*kKI*1024*kNExp;
    if(mcMalloc((void**)&gA,aB)!=mcSuccess||mcMalloc((void**)&gB,bB)!=mcSuccess){printf("alloc fail\n");return 1;}
    mcMemset(gA,0x3c,aB); mcMemset(gB,0x3c,bB);
    printf("kBatch=%d A=%.0fKB B=%.1fMB  装载:MMA = 1+%d : %d\n",kB,aB/1024.0,bB/1048576.0,kB,2*kB);
    const int Ms[]={16,32,33,64,65,128};
    printf("%-6s %-8s | %-10s %-10s %-10s %-10s\n","M","m_tiles","PF1","PF2","PF4","PF8");
    for(int M:Ms){int mt=(M+15)/16;
      printf("%-6d %-8d | %-10.2f %-10.2f %-10.2f %-10.2f\n",M,mt,
        bench<1,kB>(M,50,gA,gB),bench<2,kB>(M,50,gA,gB),
        bench<4,kB>(M,50,gA,gB),bench<8,kB>(M,50,gA,gB));}
    mcFree(gA);mcFree(gB); return 0;
}
