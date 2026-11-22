// mma_peak.cu: what the tensor pipe of this GPU (GB10, sm_121) sustains from registers alone, per mma shape and operand type, with fp32 accumulate.
//   nvcc -O3 -arch=sm_121a -DHAVE_MX -o mma_peak mma_peak.cu && ./mma_peak       (without -DHAVE_MX: no block-scaled line; -arch=sm_121 also builds)
// Each warp runs 8 independent accumulator chains of one mma shape for many iterations; a grid of SMs x 4 blocks of 4 warps. TFLOP/s = flops of the
// mma x mmas issued / time. This is the ceiling the engine's kernels are measured against (the A-era figure for bf16 m16n8k16 was about 107), and the
// answer to "does FP8 double it" before any FP8 kernel is written (Segment B2). Lab tool: no engine code, nothing is served.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

enum Kind { BF16, F16, E4M3, E4M3_MX };

template <Kind K>
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t sfa, uint32_t sfb) {
    if constexpr (K == BF16) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (K == F16) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (K == E4M3) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
#if defined(__CUDA_ARCH_FEAT_SM121_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL)
        // block-scaled: one ue8m0 scale per 32 k for A (row) and for B (column), applied inside the mma
        const uint16_t bid = 0, tid = 0;
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "h"(bid), "h"(tid), "r"(sfb), "h"(bid), "h"(tid));
#endif
    }
}

template <Kind K>
__global__ void __launch_bounds__(128) peak(float* out, int iters) {
    uint32_t a[4], b[2];
    for (int i = 0; i < 4; ++i) a[i] = 0x3c003c00u ^ (threadIdx.x * 2654435761u + i);
    for (int i = 0; i < 2; ++i) b[i] = 0x3c003c00u ^ (threadIdx.x * 40503u + i);
    const uint32_t sfa = 0x7f7f7f7fu, sfb = 0x7f7f7f7fu;
    float d[8][4] = {};
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < 8; ++j) mma<K>(d[j], a, b, sfa, sfb);
    }
    float s = 0;
    for (int j = 0; j < 8; ++j) s += d[j][0] + d[j][1] + d[j][2] + d[j][3];
    if (s == 1234.5f) out[0] = s;   // keeps the chains alive
}

template <Kind K>
int run(const char* name, double flops_per_mma, int sms) {
    float* out;
    CK(cudaMalloc(&out, 16));
    const int iters = 4000, blocks = sms * 4;
    peak<K><<<blocks, 128>>>(out, 100);          // warm
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    float best = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        cudaEventRecord(a);
        peak<K><<<blocks, 128>>>(out, iters);
        cudaEventRecord(b);
        CK(cudaEventSynchronize(b));
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    const double mmas = (double)blocks * 4 /*warps*/ * iters * 8;
    printf("%-34s %8.2f ms   %7.1f TFLOP/s\n", name, best, mmas * flops_per_mma / (best * 1e-3) / 1e12);
    cudaFree(out);
    return 0;
}

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    int clk = 0;
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    printf("%s sm_%d%d, %d SMs, clock %d MHz\n", p.name, p.major, p.minor, p.multiProcessorCount, clk / 1000);
    run<BF16>("bf16 m16n8k16", 2.0 * 16 * 8 * 16, p.multiProcessorCount);
    run<F16>("fp16 m16n8k16", 2.0 * 16 * 8 * 16, p.multiProcessorCount);
    run<E4M3>("e4m3 m16n8k32", 2.0 * 16 * 8 * 32, p.multiProcessorCount);
#ifdef HAVE_MX
    run<E4M3_MX>("e4m3 m16n8k32 block-scaled (ue8m0)", 2.0 * 16 * 8 * 32, p.multiProcessorCount);
#else
    printf("(block-scaled mma: build with -arch=sm_121a -DHAVE_MX)\n");
#endif
    return 0;
}
