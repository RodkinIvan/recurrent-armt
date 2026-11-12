// armt.cu -- ARMT (Associative Recurrent Memory Transformer) on a Gated DeltaNet-2 backbone, in one CUDA file.
//
// Forward (inference, with a carried fixed-size state) and forward+backward (training) of
// recurrent_armt.RecurrentARMTForCausalLM over a GDN-2 base model (recurrent_armt/gdn2.py, fla GatedDeltaNet2).
//
// Build:   nvcc -O3 -arch=sm_89 -shared -Xcompiler -fPIC armt.cu -lcublas -o libarmt_bf16.so            (bf16 activations)
//          nvcc -O3 -arch=sm_89 -shared -Xcompiler -fPIC -DARMT_FP32 armt.cu -lcublas -o libarmt_fp32.so (fp32 everything)
//
// Layout of one (layer, segment) step. The segment's real tokens and the memory tokens form ONE sequence of
// T1 = Tr + Tm rows per sample: every token-wise op (norms, projections, MLP, memory read) runs once over it; the
// recurrent parts (short conv, GDN-2 scan) run over the concatenation, which is exactly "memory tokens continue
// from the state after the last real token"; the state carried to the next segment is snapshotted after the real
// tokens (read-out branch) or after the memory tokens (mem_in_stream).
//
// GDN-2 scan, chunked (chunk CH=64; per head, S in R^{K x V}):
//   S_t = (I - k_t (b_t*k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t*v_t)^T ,  o_t = S_t^T q_t
// per chunk: gc = cumsum(g), L_ij = sum_d (b k)_id k_jd e^{gc_id-gc_jd} (i>j), A = (I+L)^-1,
//   u = A (w*v), Wm = A (b*k*e^{gc}), M_ij = sum_d q_id k_jd e^{gc_id-gc_jd} (i>=j), Qg = q e^{gc}, Kt = k e^{gl-gc}
//   vn = u - Wm S0 ; o = Qg S0 + M vn ; S1 = e^{gl} S0 + Kt^T vn
// Intra-chunk pieces are custom kernels (exact pairwise decays: never overflow), the per-chunk matmuls are
// cuBLAS strided-batched GEMMs, and only the S recurrence is sequential (state_fwd / state_bwd kernels).
// Chunk buffers are zero-padded: padded tokens (k=v=b=w=0, g=0) are exact no-ops of the recurrence.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <vector>
#include <string>
#include <stdexcept>

#ifdef ARMT_FP32
typedef float floatX;
#define CUDA_X CUDA_R_32F
#define EXPF expf
#else
typedef __nv_bfloat16 floatX;
#define CUDA_X CUDA_R_16BF
#define EXPF __expf
#endif

// SCAN32: the GDN-2 scan in fp32 (CUDA-core kernels, fp32 intermediates). Always in the fp32 build; in the bf16 build with
// -DARMT_SCAN32 (bf16 matmuls and activations, fp32 scan: slower, closer to fp32 numerics).
#if defined(ARMT_FP32) || defined(ARMT_SCAN32)
#define SCAN32
typedef float floatS;
#define CUDA_S CUDA_R_32F
#else
typedef floatX floatS;        // storage of the GDN-2 scan intermediates (bf16 in the bf16 build, like fla)
#define CUDA_S CUDA_X
#endif
#ifndef ARMT_FP32
#include <mma.h>
using namespace nvcuda;
#endif
// errors throw; the C API catches them and returns -1 (message: armt_last_error()), so Python raises instead of exiting
static void armt_fail(const std::string& msg) { throw std::runtime_error(msg); }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) armt_fail(std::string("CUDA error ") + cudaGetErrorString(e_) + " at armt.cu:" + std::to_string(__LINE__)); } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) armt_fail("cuBLAS error " + std::to_string((int)s_) + " at armt.cu:" + std::to_string(__LINE__)); } while (0)
#define CH 64
#define MAXL 256

static __device__ __forceinline__ float toF(float x) { return x; }
static __device__ __forceinline__ float toF(__nv_bfloat16 x) { return __bfloat162float(x); }
static __device__ __forceinline__ void stX(float* p, float v) { *p = v; }
static __device__ __forceinline__ void stX(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
// vector loads / stores of N consecutive elements (N = 2 or 4; aligned)
template <int N> static __device__ __forceinline__ void ldv(const float* p, float* o) {
    if constexpr (N == 4) { float4 v = *(const float4*)p; o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w; }
    else { float2 v = *(const float2*)p; o[0] = v.x; o[1] = v.y; }
}
template <int N> static __device__ __forceinline__ void ldv(const __nv_bfloat16* p, float* o) {
    if constexpr (N == 4) {
        uint2 v = *(const uint2*)p;
        __nv_bfloat162 a = *(__nv_bfloat162*)&v.x, b = *(__nv_bfloat162*)&v.y;
        o[0] = __low2float(a); o[1] = __high2float(a); o[2] = __low2float(b); o[3] = __high2float(b);
    } else { __nv_bfloat162 a = *(const __nv_bfloat162*)p; o[0] = __low2float(a); o[1] = __high2float(a); }
}
template <int N> static __device__ __forceinline__ void stv(float* p, const float* x) {
    if constexpr (N == 4) *(float4*)p = make_float4(x[0], x[1], x[2], x[3]);
    else *(float2*)p = make_float2(x[0], x[1]);
}
template <int N> static __device__ __forceinline__ void stv(__nv_bfloat16* p, const float* x) {
    if constexpr (N == 4) {
        uint2 v;
        *(__nv_bfloat162*)&v.x = __floats2bfloat162_rn(x[0], x[1]);
        *(__nv_bfloat162*)&v.y = __floats2bfloat162_rn(x[2], x[3]);
        *(uint2*)p = v;
    } else *(__nv_bfloat162*)p = __floats2bfloat162_rn(x[0], x[1]);
}
static __device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + EXPF(-x)); }
static __device__ __forceinline__ float silu(float x) { return x * sigm(x); }
static __device__ __forceinline__ float dsilu(float x) { float s = sigm(x); return s * (1.f + x * (1.f - s)); }
static __device__ __forceinline__ float softplus(float x) { return x > 20.f ? x : log1pf(EXPF(x)); }
static __device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
// sum over the block (blockDim.x multiple of 32, <= 1024); every thread gets the result
static __device__ float block_sum(float v) {
    __shared__ float sh[33];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    if (w == 0) {
        float x = lane < nw ? sh[lane] : 0.f;
        x = warp_sum(x);
        if (lane == 0) sh[32] = x;
    }
    __syncthreads();
    return sh[32];
}
// flattened chunk position of token t: real tokens fill chunks [0, nr), memory tokens start chunk nr
static __device__ __forceinline__ int tok_of_pos(int pos, int Tr, int Tm, int nr) {
    if (pos < nr * CH) return pos < Tr ? pos : -1;
    int r = pos - nr * CH;
    return r < Tm ? Tr + r : -1;
}
static __device__ __forceinline__ int pos_of_tok(int t, int Tr, int nr) { return t < Tr ? t : nr * CH + (t - Tr); }

// =============================================================================================== config / params
typedef struct {
    int vocab, d_model, n_layers, n_heads, head_dim, d_ff, conv_size, gate_swish;
    float norm_eps;
    int num_mem, d_mem, mem_heads, segment_size, mem_in_stream, gating, tie_emb, allow_tf32;
    int wrap[MAXL];
} ArmtConfig;

// canonical parameter order (the Python side passes pointers in this order; grads use a flat buffer in this order)
enum { P_EMB, P_NORMF, P_LMHEAD, P_MEMTOK, P_GLOBAL };
enum { L_N1, L_Q, L_K, L_V, L_B, L_W, L_F1, L_G1, L_QC, L_KC, L_VC, L_F2, L_ALOG, L_DTB, L_G2, L_G2B, L_ON, L_O, L_N2,
       L_W1, L_W2, L_W3, L_MQ, L_MK, L_MV, L_MB, L_MBB, L_COUNT };

typedef struct {
    ArmtConfig c;
    int D, L, H, KD, KDt, VDt, Hv, F, W, NP, Cc, dm, Hm, dk, nmb, M, V;
    int has_mem[MAXL];
    long n_params;
    long numel[P_GLOBAL + MAXL * L_COUNT];
    long offset[P_GLOBAL + MAXL * L_COUNT];
    long total;
} Dims;

static void make_dims(Dims* d, const ArmtConfig* c) {
    memset(d, 0, sizeof(*d));
    d->c = *c;
    d->D = c->d_model; d->L = c->n_layers; d->H = c->n_heads; d->KD = c->head_dim; d->F = c->d_ff; d->W = c->conv_size;
    d->KDt = d->H * d->KD; d->VDt = d->KDt; d->Hv = d->KD;
    d->NP = 2 * d->KDt + d->VDt + d->KDt + d->VDt + 2 * d->Hv;
    d->Cc = 2 * d->KDt + d->VDt;
    d->M = c->num_mem; d->dm = c->d_mem; d->Hm = c->mem_heads > 0 ? c->mem_heads : 1; d->dk = 6 * c->d_mem;
    d->nmb = c->gating ? d->D : d->Hm; d->V = c->vocab;
    long* n = d->numel;
    n[P_EMB] = (long)d->V * d->D; n[P_NORMF] = d->D; n[P_LMHEAD] = c->tie_emb ? 0 : (long)d->V * d->D; n[P_MEMTOK] = (long)d->M * d->D;
    for (int i = 0; i < d->L; i++) {
        long* p = n + P_GLOBAL + i * L_COUNT;
        int D = d->D;
        d->has_mem[i] = d->M > 0 && c->wrap[i];
        p[L_N1] = D; p[L_Q] = (long)d->KDt * D; p[L_K] = (long)d->KDt * D; p[L_V] = (long)d->VDt * D; p[L_B] = (long)d->KDt * D;
        p[L_W] = (long)d->VDt * D; p[L_F1] = (long)d->Hv * D; p[L_G1] = (long)d->Hv * D;
        p[L_QC] = (long)d->KDt * d->W; p[L_KC] = (long)d->KDt * d->W; p[L_VC] = (long)d->VDt * d->W;
        p[L_F2] = (long)d->KDt * d->Hv; p[L_ALOG] = d->H; p[L_DTB] = d->KDt; p[L_G2] = (long)d->VDt * d->Hv; p[L_G2B] = d->VDt;
        p[L_ON] = d->KD; p[L_O] = (long)D * d->VDt; p[L_N2] = D;
        p[L_W1] = (long)d->F * D; p[L_W2] = (long)d->F * D; p[L_W3] = (long)D * d->F;
        if (d->has_mem[i]) {
            p[L_MQ] = (long)d->dm * D; p[L_MK] = (long)d->dm * D; p[L_MV] = (long)D * D; p[L_MB] = (long)d->nmb * D; p[L_MBB] = d->nmb;
        }
    }
    d->n_params = P_GLOBAL + (long)d->L * L_COUNT;
    long off = 0;
    for (long i = 0; i < d->n_params; i++) { d->offset[i] = off; off += n[i]; }
    d->total = off;
}
static inline int PI(int layer, int j) { return P_GLOBAL + layer * L_COUNT + j; }

// =============================================================================================== context
// Device memory comes from the allocator set with armt_set_allocator (the Python side plugs in PyTorch's caching
// allocator, so the library's memory is visible to torch.cuda memory stats and reuses its cache); cudaMalloc otherwise.
typedef void* (*armt_alloc_fn)(size_t bytes, void* stream);
typedef void (*armt_free_fn)(void* ptr, void* stream);
static armt_alloc_fn g_alloc = NULL;
static armt_free_fn g_free = NULL;
static cudaStream_t g_cur_stream = 0;
typedef struct { void* p; size_t n; } Buf;
static void buf_free(Buf* b) {
    if (b->p) { if (g_free) g_free(b->p, (void*)g_cur_stream); else CK(cudaFree(b->p)); }
    b->p = NULL; b->n = 0;
}
static void* ensure(Buf* b, size_t n) {
    if (n == 0) n = 16;
    if (b->n < n) {
        buf_free(b);
        if (g_alloc) { b->p = g_alloc(n, (void*)g_cur_stream); if (!b->p) armt_fail("out of memory (" + std::to_string(n >> 20) + " MB)"); }
        else CK(cudaMalloc(&b->p, n));
        b->n = n;
    }
    return b->p;
}

typedef struct {   // per-layer recurrent + memory state, as device pointers (may be NULL = zero state)
    float* S;      // [B, H, KD, KD]
    float* cs;     // [B, Cc, W]
    float* Wm;     // [B, Hm, dk/Hm, D/Hm]
    float* z;      // [B, Hm, dk/Hm]
} LState;

enum {
    // forward scratch
    B_MQ, B_PHIQ, B_DENQ, B_NUMQ, B_XR, B_RSTD1, B_XMID, B_RSTD2, B_H1, B_P, B_FB, B_GG, B_ON, B_RSTDO, B_AO, B_H2, B_A, B_MM, B_MO,
    B_CQ, B_CK, B_CB, B_CG, B_GC, B_Y, B_QG, B_KT, B_WM, B_CV, B_CW, B_CWV, B_U, B_VN, B_O, B_AM, B_MMAT, B_GL, B_S0,
    B_XM, B_MK, B_PHIK, B_DENK, B_NRMK, B_NMV, B_MBP, B_PREVN, B_E, B_COEF, B_CSTMP, B_SDISC, B_WDISC, B_ZDISC, B_CSDISC,
    // backward scratch
    B_DXX, B_DMM, B_DA, B_DH2, B_DON, B_DGG, B_DP, B_DF, B_DH1, B_DC, B_DNUMQ, B_DPHIQ, B_DMQ, B_DDENQ,
    B_DO, B_DSE, B_DVN, B_DQG, B_DKT, B_DWM, B_DY, B_DMMAT, B_DAM, B_T1, B_DL, B_DWV, B_DQ, B_DK, B_DB, B_DG,
    B_DE, B_DPHIK, B_DPREVN, B_DNMV, B_DMBP, B_DDENK, B_DMK, B_DXM, B_DCSTMP,
    // final / loss
    B_HF, B_RSTDF, B_LOGITS, B_TGT, B_DHF,
    // inference ping-pong / training stores
    B_X0, B_X1, B_XS, B_DXS, B_ST_S, B_ST_CS, B_ST_W, B_ST_Z, B_DST_S, B_DST_CS, B_DST_W, B_DST_Z, B_ZERO,
    B_PACK, B_PACKTAB, B_WCAT, B_AMX, B_PREDVN, B_IST_S, B_IST_CS, B_IST_W, B_IST_Z, B_XOT, B_S0LO, B_O32,
    B_COUNT
};

typedef struct {
    Dims d;
    cublasHandle_t blas;
    cudaStream_t stream;
    int sms;
    Buf buf[B_COUNT];
    const float** params;     // host array of device pointers (canonical order)
    float* grad;              // flat fp32 grad buffer (device) of the current call, or NULL
    uint8_t* gmask;           // per parameter: 1 = needs a gradient (host, n_params)
    char* htab;               // pinned host staging of the pack descriptor table
    cudaEvent_t pack_done;    // the last copy out of htab finished
    // packed floatX weights
    floatX* w_in[MAXL]; floatX* w_f2[MAXL]; floatX* w_g2[MAXL]; floatX* w_o[MAXL]; floatX* w_12[MAXL]; floatX* w_3[MAXL];
    floatX* w_lm;
    // training record
    int tr_B, tr_T, tr_nseg, tr_T1max, tr_valid;
    int *tr_Tr, *tr_Tm, *tr_start, tr_cap;   // per piece of the recorded training forward (grown on demand)
    const uint8_t* tr_reset;  // [nseg, B] device, or NULL
    float* tr_grad;           // flat grad buffer of the recorded training forward
} Ctx;

#define BF(c, id, T, n) ((T*)ensure(&(c)->buf[id], (size_t)(n) * sizeof(T)))

static float* PP(Ctx* c, int idx) { return (float*)c->params[idx]; }
static float* GP(Ctx* c, int idx) {   // grad pointer of a parameter (tied lm_head -> embedding); NULL if it needs none
    if (!c->grad) return NULL;
    if (idx == P_LMHEAD && c->d.c.tie_emb) idx = P_EMB;
    if (c->gmask && !c->gmask[idx]) return NULL;
    return c->grad + c->d.offset[idx];
}
// grad of a packed group of consecutive parameters [i0, i1] (one GEMM): its base if any member needs a gradient
static float* GPG(Ctx* c, int i0, int i1) {
    if (!c->grad) return NULL;
    bool any = !c->gmask;
    for (int i = i0; i <= i1 && !any; i++) any = c->gmask[i];
    return any ? c->grad + c->d.offset[i0] : NULL;
}

// =============================================================================================== GEMM (row-major)
// C[M,N] (ldc) = alpha * op(A)[M,K] * op(B)[K,N] + beta * C ; op(X) = X^T if tX. Strided batched if batch > 1.
static void mm(Ctx* c, int tA, int tB, int M, int N, int K, float alpha,
               const void* A, cudaDataType ta, int lda, long long sA,
               const void* B, cudaDataType tb, int ldb, long long sB, float beta,
               void* C, cudaDataType tc, int ldc, long long sC, int batch) {
    if (M <= 0 || N <= 0 || batch <= 0) return;
    if (K <= 0) { if (beta == 0.f && batch == 1 && ldc == N) CK(cudaMemsetAsync(C, 0, (size_t)M * N * (tc == CUDA_R_32F ? 4 : 2), c->stream)); return; }
    cublasComputeType_t ct = CUBLAS_COMPUTE_32F;
    if (ta == CUDA_R_32F && tb == CUDA_R_32F && c->d.c.allow_tf32) ct = CUBLAS_COMPUTE_32F_FAST_TF32;
    cublasOperation_t oa = tB ? CUBLAS_OP_T : CUBLAS_OP_N, ob = tA ? CUBLAS_OP_T : CUBLAS_OP_N;
    if (batch == 1)
        CB(cublasGemmEx(c->blas, oa, ob, N, M, K, &alpha, B, tb, ldb, A, ta, lda, &beta, C, tc, ldc, ct, CUBLAS_GEMM_DEFAULT));
    else
        CB(cublasGemmStridedBatchedEx(c->blas, oa, ob, N, M, K, &alpha, B, tb, ldb, sB, A, ta, lda, sA, &beta, C, tc, ldc, sC,
                                      batch, ct, CUBLAS_GEMM_DEFAULT));
}
#define F32 CUDA_R_32F
// y[N,O] = x[N,I] W[O,I]^T   (floatX)
static void lin(Ctx* c, floatX* y, int ldy, const floatX* x, int ldx, const floatX* W, int N, int O, int I, float beta) {
    mm(c, 0, 1, N, O, I, 1.f, x, CUDA_X, ldx, 0, W, CUDA_X, I, 0, beta, y, CUDA_X, ldy, 0, 1);
}
// dx[N,I] (+)= dy[N,O] W[O,I]
static void lin_dx(Ctx* c, floatX* dx, int lddx, const floatX* dy, int lddy, const floatX* W, int N, int O, int I, float beta) {
    mm(c, 0, 0, N, I, O, 1.f, dy, CUDA_X, lddy, 0, W, CUDA_X, I, 0, beta, dx, CUDA_X, lddx, 0, 1);
}
// dW[O,I] += dy[N,O]^T x[N,I]   (fp32 accumulation)
static void lin_dw(Ctx* c, float* dW, const floatX* dy, int lddy, const floatX* x, int ldx, int N, int O, int I) {
    if (!dW) return;
    mm(c, 1, 0, O, I, N, 1.f, dy, CUDA_X, lddy, 0, x, CUDA_X, ldx, 0, 1.f, dW, F32, I, 0, 1);
}
// fp32 versions (memory projections)
static void lin32(Ctx* c, float* y, int ldy, const float* x, int ldx, const float* W, int N, int O, int I, float beta) {
    mm(c, 0, 1, N, O, I, 1.f, x, F32, ldx, 0, W, F32, I, 0, beta, y, F32, ldy, 0, 1);
}
static void lin32_dx(Ctx* c, float* dx, int lddx, const float* dy, int lddy, const float* W, int N, int O, int I, float beta) {
    mm(c, 0, 0, N, I, O, 1.f, dy, F32, lddy, 0, W, F32, I, 0, beta, dx, F32, lddx, 0, 1);
}
static void lin32_dw(Ctx* c, float* dW, const float* dy, int lddy, const float* x, int ldx, int N, int O, int I) {
    if (!dW) return;
    mm(c, 1, 0, O, I, N, 1.f, dy, F32, lddy, 0, x, F32, ldx, 0, 1.f, dW, F32, I, 0, 1);
}

// =============================================================================================== small kernels
__global__ void k_pack(const float* const* src, floatX* const* dst, const long* n, int count) {
    int i = blockIdx.y;
    if (i >= count) return;
    const float* s = src[i]; floatX* d = dst[i]; long m = n[i];
    for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x; j < m; j += (long)gridDim.x * blockDim.x) stX(d + j, s[j]);
}
__global__ void k_cast(floatX* y, const float* x, long n) {
    for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x; j < n; j += (long)gridDim.x * blockDim.x) stX(y + j, x[j]);
}
__global__ void k_scale(float* x, long n, const float* s) {
    float f = *s;
    for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x; j < n; j += (long)gridDim.x * blockDim.x) x[j] *= f;
}
__global__ void k_add(float* y, const float* x, long n) {
    for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x; j < n; j += (long)gridDim.x * blockDim.x) y[j] += x[j];
}
// zero the per-sample blocks of samples with mask[b] != 0
__global__ void k_mask_zero(float* x, const uint8_t* mask, long per_b, int B) {
    long n = per_b * B;
    for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x; j < n; j += (long)gridDim.x * blockDim.x)
        if (mask[j / per_b]) x[j] = 0.f;
}
// x[b, t] = emb[ids[b, t0 + t]] (t < Tr) or mem[t - Tr]
__global__ void k_embed(float* x, const int64_t* ids, int ids_ld, int t0, const float* emb, const float* mem, int Tr, int Tm, int D) {
    int row = blockIdx.x, T1 = Tr + Tm, b = row / T1, t = row % T1;
    const float* src = t < Tr ? emb + ids[(long)b * ids_ld + t0 + t] * (long)D : mem + (long)(t - Tr) * D;
    float* dst = x + (long)row * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) dst[d] = src[d];
}
__global__ void k_embed_bwd(const float* dx, const int64_t* ids, int ids_ld, int t0, float* demb, float* dmem, int Tr, int Tm, int D) {
    int row = blockIdx.x, T1 = Tr + Tm, b = row / T1, t = row % T1;
    float* dst = t < Tr ? (demb ? demb + ids[(long)b * ids_ld + t0 + t] * (long)D : NULL) : (dmem ? dmem + (long)(t - Tr) * D : NULL);
    if (!dst) return;
    const float* src = dx + (long)row * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) atomicAdd(dst + d, src[d]);
}

// ---------------------------------------------------------------- rmsnorm
// xo = xa (+ xb) (+ num/den per memory head); h = xo * rsqrt(mean(xo^2)+eps) * w.  Row r of the output reads source row
// map(r) = (r / rpb_dst) * rpb_src + r % rpb_dst (to normalize only the real rows of a [B, T1] block).
__global__ void k_rmsnorm(float* __restrict__ xo, const float* __restrict__ xa, const floatX* __restrict__ xb, const float* __restrict__ num, const float* __restrict__ den, int Hm,
                          floatX* __restrict__ h, float* __restrict__ rstd, const float* __restrict__ w, int D, float eps, int rpb_dst, int rpb_src) {
    int r = blockIdx.x;
    long src = (long)(r / rpb_dst) * rpb_src + r % rpb_dst;
    const float* a = xa + src * D;
    float* o = xo ? xo + src * D : NULL;
    float ss = 0.f;
    int dh = D / Hm;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = a[d];
        if (xb) v += toF(xb[src * D + d]);
        if (num) v += num[src * D + d] / den[src * Hm + d / dh];
        if (o) o[d] = v;
        ss += v * v;
    }
    ss = block_sum(ss);
    float rs = rsqrtf(ss / D + eps);
    if (threadIdx.x == 0 && rstd) rstd[r] = rs;
    const float* xv = o ? o : a;
    floatX* hr = h + (long)r * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) stX(hr + d, xv[d] * rs * w[d]);
}
// dx[map(r)] += rstd*(dxh - xh*mean(dxh*xh)), dxh = dy*w, xh = x*rstd ; dw += sum_r dy*xh. grid-stride rows.
__global__ void k_rmsnorm_bwd(float* __restrict__ dx, const floatX* __restrict__ dy, const float* __restrict__ x, const float* __restrict__ rstd, const float* __restrict__ w, float* __restrict__ dw,
                              int R, int D, int rpb_dst, int rpb_src) {
    extern __shared__ float sdw[];
    for (int d = threadIdx.x; d < D; d += blockDim.x) sdw[d] = 0.f;
    for (int r = blockIdx.x; r < R; r += gridDim.x) {
        long src = (long)(r / rpb_dst) * rpb_src + r % rpb_dst;
        const float* xr = x + src * D;
        const floatX* dyr = dy + (long)r * D;
        float rs = rstd[r], acc = 0.f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) { float xh = xr[d] * rs; acc += toF(dyr[d]) * w[d] * xh; }
        acc = block_sum(acc) / D;
        float* dxr = dx + src * D;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            float g = toF(dyr[d]), xh = xr[d] * rs;
            dxr[d] += rs * (g * w[d] - xh * acc);
            sdw[d] += g * xh;
        }
    }
    __syncthreads();
    if (dw) for (int d = threadIdx.x; d < D; d += blockDim.x) atomicAdd(dw + d, sdw[d]);
}

// ---------------------------------------------------------------- DPFP features (nu = 3) of the memory
// per (row, head): x = proj[row, h*dmh:(h+1)*dmh]; x2 = [relu(x), relu(-x)]; psi_f = x2[i] * x2[(i-j) mod 2dmh],
// f = (j-1)*2dmh + i, j = 1..3; phi = psi / max(|psi|, 1e-12). den = z[b,h].phi + 1e-5 ; nrm2 = |phi|^2.
__global__ void k_dpfp(const float* proj, int ldp, float* phi, float* den, float* nrm2, const float* z, int R, int rpb, int dm, int Hm) {
    int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (wid >= R * Hm) return;
    int row = wid / Hm, h = wid % Hm, dmh = dm / Hm, n2 = 2 * dmh, dkh = 3 * n2;
    const float* x = proj + (long)row * ldp + h * dmh;
    float* p = phi + (long)row * (dkh * Hm) + h * dkh;
    float ss = 0.f;
    for (int f = lane; f < dkh; f += 32) {
        int j = f / n2 + 1, i = f % n2, i2 = (i - j + n2) % n2;
        float a = i < dmh ? fmaxf(x[i], 0.f) : fmaxf(-x[i - dmh], 0.f);
        float c = i2 < dmh ? fmaxf(x[i2], 0.f) : fmaxf(-x[i2 - dmh], 0.f);
        float v = a * c;
        p[f] = v;
        ss += v * v;
    }
    ss = warp_sum(ss);
    float inv = 1.f / fmaxf(sqrtf(ss), 1e-12f), dz = 0.f, n = 0.f;
    const float* zz = z ? z + (long)(row / rpb) * (dkh * Hm) + h * dkh : NULL;
    __syncwarp();
    for (int f = lane; f < dkh; f += 32) {
        float v = p[f] * inv;
        p[f] = v;
        n += v * v;
        if (zz) dz += zz[f] * v;
    }
    dz = warp_sum(dz); n = warp_sum(n);
    if (lane == 0) {
        if (den) den[row * Hm + h] = dz + 1e-5f;
        if (nrm2) nrm2[row * Hm + h] = n;
    }
}
// dproj from dphi (recomputes psi). dynamic smem: (blockDim/32) * dkh floats
__global__ void k_dpfp_bwd(const float* proj, int ldp, const float* dphi, float* dproj, int lddp, int R, int dm, int Hm) {
    extern __shared__ float sm[];
    int wl = threadIdx.x >> 5, wid = blockIdx.x * (blockDim.x >> 5) + wl, lane = threadIdx.x & 31;
    if (wid >= R * Hm) return;
    int row = wid / Hm, h = wid % Hm, dmh = dm / Hm, n2 = 2 * dmh, dkh = 3 * n2;
    float* dpsi = sm + wl * dkh;
    const float* x = proj + (long)row * ldp + h * dmh;
    const float* dp = dphi + (long)row * (dkh * Hm) + h * dkh;
#define X2(i) ((i) < dmh ? fmaxf(x[(i)], 0.f) : fmaxf(-x[(i) - dmh], 0.f))
    float ss = 0.f;
    for (int f = lane; f < dkh; f += 32) {
        int j = f / n2 + 1, i = f % n2, i2 = (i - j + n2) % n2;
        float v = X2(i) * X2(i2);
        ss += v * v;
    }
    ss = warp_sum(ss);
    float nrm = sqrtf(ss), nn = fmaxf(nrm, 1e-12f), dot = 0.f;
    for (int f = lane; f < dkh; f += 32) {
        int j = f / n2 + 1, i = f % n2, i2 = (i - j + n2) % n2;
        dot += X2(i) * X2(i2) / nn * dp[f];
    }
    dot = warp_sum(dot);
    for (int f = lane; f < dkh; f += 32) {
        int j = f / n2 + 1, i = f % n2, i2 = (i - j + n2) % n2;
        float ph = X2(i) * X2(i2) / nn;
        dpsi[f] = nrm > 1e-12f ? (dp[f] - ph * dot) / nn : dp[f] / nn;
    }
    __syncwarp();
    for (int k = lane; k < dmh; k += 32) {
        float g[2];
        for (int s = 0; s < 2; s++) {
            int i = k + s * dmh;
            float acc = 0.f;
            for (int j = 1; j <= 3; j++) {
                int im = (i - j + n2) % n2, ip = (i + j) % n2;
                acc += dpsi[(j - 1) * n2 + i] * X2(im) + dpsi[(j - 1) * n2 + ip] * X2(ip);
            }
            g[s] = acc;
        }
        float xv = x[k];
        dproj[(long)row * lddp + h * dmh + k] = (xv > 0.f ? g[0] : 0.f) - (xv < 0.f ? g[1] : 0.f);
    }
#undef X2
}
// read backward helper: dnum = dxr/den ; dden[row,h] = -sum_{d in h} dxr*num/den^2. block per row.
__global__ void k_read_bwd(const float* dxr, const float* num, const float* den, float* dnum, float* dden, int D, int Hm) {
    int r = blockIdx.x, dh = D / Hm;
    for (int h = 0; h < Hm; h++) {
        float de = den[r * Hm + h], acc = 0.f;
        for (int d = h * dh + threadIdx.x; d < (h + 1) * dh; d += blockDim.x) {
            long i = (long)r * D + d;
            float g = dxr[i];
            dnum[i] = g / de;
            acc += g * num[i];
        }
        acc = block_sum(acc);
        if (threadIdx.x == 0) dden[r * Hm + h] = -acc / (de * de);
    }
}
// dphi += dden[row,h] * z[b,h,:]  (+ coef[row,h] * dz[b,h,:]) ; one warp per (row, h)
__global__ void k_dphi_add(float* dphi, const float* dden, const float* z, const float* coef, const float* dz, int R, int rpb, int dkh, int Hm) {
    int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (wid >= R * Hm) return;
    int row = wid / Hm, h = wid % Hm;
    long zo = (long)(row / rpb) * dkh * Hm + h * dkh;
    float a = dden ? dden[wid] : 0.f, cf = coef ? coef[wid] : 0.f;
    float* p = dphi + (long)row * dkh * Hm + h * dkh;
    for (int f = lane; f < dkh; f += 32) p[f] += a * z[zo + f] + (coef ? cf * dz[zo + f] : 0.f);
}
// dz[b,h,:] += sum_m w[row, h] * phi[row, h, :] over the rows of sample b. block per (b, h).
__global__ void k_dz_acc(float* dz, const float* w, const float* phi, int rpb, int dkh, int Hm) {
    int b = blockIdx.x / Hm, h = blockIdx.x % Hm;
    for (int f = threadIdx.x; f < dkh; f += blockDim.x) {
        float acc = 0.f;
        for (int m = 0; m < rpb; m++) { long row = (long)b * rpb + m; acc += w[row * Hm + h] * phi[row * dkh * Hm + h * dkh + f]; }
        dz[((long)b * Hm + h) * dkh + f] += acc;
    }
}
// write: coef (empty memory -> 1) and z_new = z_old + sum_m coef*phik. block per (b, h).
__global__ void k_write_z(const float* z_old, float* z_new, const float* phik, const float* denk, const float* nrmk, float* coef, int M, int dkh, int Hm) {
    int b = blockIdx.x / Hm, h = blockIdx.x % Hm;
    const float* zo = z_old + ((long)b * Hm + h) * dkh;
    float nz = 0.f;
    for (int f = threadIdx.x; f < dkh; f += blockDim.x) nz += zo[f] != 0.f ? 1.f : 0.f;
    nz = block_sum(nz);
    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        long r = (long)b * M + m;
        float cf = 1.f - denk[r * Hm + h] / nrmk[r * Hm + h];
        cf = fminf(fmaxf(cf, 0.f), 1.f);
        coef[r * Hm + h] = nz == 0.f ? 1.f : cf;
    }
    __syncthreads();
    for (int f = threadIdx.x; f < dkh; f += blockDim.x) {
        float acc = zo[f];
        for (int m = 0; m < M; m++) { long r = (long)b * M + m; acc += coef[r * Hm + h] * phik[r * dkh * Hm + h * dkh + f]; }
        z_new[((long)b * Hm + h) * dkh + f] = acc;
    }
}
// E = (nmv - prevn/denk) * sigmoid(mbp + bias). one block per row.
__global__ void k_write_E(float* E, const float* nmv, const float* prevn, const float* denk, const float* mbp, const float* mbb,
                          int D, int Hm, int nmb) {
    int r = blockIdx.x, dh = D / Hm;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        long i = (long)r * D + d;
        int h = d / dh, mi = nmb == D ? d : h;
        float mb = sigm(mbp[(long)r * nmb + mi] + mbb[mi]);
        E[i] = (nmv[i] - prevn[i] / denk[r * Hm + h]) * mb;
    }
}
// write backward, part 1: from dE: dnmv = dE*mb ; dprevn = -dE*mb/denk ; ddenk = sum_h dE*mb*prevn/denk^2 ;
// dmbp = sum (dE * delta) * mb(1-mb) (per head or per channel). block per row.
__global__ void k_write_bwd1(const float* dE, const float* nmv, const float* prevn, const float* denk, const float* mbp, const float* mbb,
                             float* dnmv, float* dprevn, float* ddenk, float* dmbp, int D, int Hm, int nmb) {
    int r = blockIdx.x, dh = D / Hm;
    for (int h = 0; h < Hm; h++) {
        float de = denk[r * Hm + h], a1 = 0.f, a2 = 0.f;
        for (int d = h * dh + threadIdx.x; d < (h + 1) * dh; d += blockDim.x) {
            long i = (long)r * D + d;
            int mi = nmb == D ? d : h;
            float mb = sigm(mbp[(long)r * nmb + mi] + mbb[mi]);
            float delta = nmv[i] - prevn[i] / de, g = dE[i];
            dnmv[i] = g * mb;
            dprevn[i] = -g * mb / de;
            a1 += g * mb * prevn[i] / (de * de);
            if (nmb == D) dmbp[(long)r * nmb + d] = g * delta * mb * (1.f - mb);
            else a2 += g * delta;
        }
        a1 = block_sum(a1);
        if (nmb != D) a2 = block_sum(a2);
        if (threadIdx.x == 0) {
            ddenk[r * Hm + h] = a1;
            if (nmb != D) { float mb = sigm(mbp[(long)r * nmb + h] + mbb[h]); dmbp[(long)r * nmb + h] = a2 * mb * (1.f - mb); }
        }
    }
}
// gather / scatter the memory rows (t >= Tr) of a [B, T1, D] block
__global__ void k_gather_mem(float* xm, const float* x, int Tr, int Tm, int D) {
    int r = blockIdx.x, b = r / Tm, m = r % Tm;
    const float* s = x + ((long)b * (Tr + Tm) + Tr + m) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) xm[(long)r * D + d] = s[d];
}
__global__ void k_scatter_add_mem(float* x, const float* xm, int Tr, int Tm, int D) {
    int r = blockIdx.x, b = r / Tm, m = r % Tm;
    float* s = x + ((long)b * (Tr + Tm) + Tr + m) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) s[d] += xm[(long)r * D + d];
}
// bias grad: db[j] += sum_r dy[r, j]
__global__ void k_colsum(float* db, const float* dy, int R, int C) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < C; j += gridDim.x * blockDim.x) {
        float acc = 0.f;
        for (int r = 0; r < R; r++) acc += dy[(long)r * C + j];
        db[j] += acc;
    }
}

// ---------------------------------------------------------------- GDN-2 token prep: short conv + SiLU, l2norm, gates
// one warp per (token, head); lane l handles the KD/32 consecutive channels l*KD/32 ..
template <int V, typename T>
static __device__ __forceinline__ void conv_vec(float* acc, const T* P, int NP, int col, const float* w, const float* cs, long csrow,
                                                long rowbase, int t, int W) {
#pragma unroll
    for (int j = 0; j < V; j++) acc[j] = 0.f;
    for (int i = 0; i < W; i++) {
        int tt = t - (W - 1) + i;
        float x[V];
        if (tt >= 0) ldv<V>(P + (rowbase + tt) * NP + col, x);
        else {
#pragma unroll
            for (int j = 0; j < V; j++) x[j] = cs ? cs[(csrow + j) * W + W + tt] : 0.f;
        }
#pragma unroll
        for (int j = 0; j < V; j++) acc[j] += w[j * W + i] * x[j];
    }
}
template <int KD>
__global__ void k_prep_fwd(const floatX* __restrict__ P, int NP, const floatX* __restrict__ fb,
                           const float* __restrict__ wq, const float* __restrict__ wk, const float* __restrict__ wv, const float* __restrict__ cs_in,
                           const float* __restrict__ A_log, const float* __restrict__ dtb,
                           floatS* __restrict__ cq, floatS* __restrict__ ck, float* __restrict__ cg, floatS* __restrict__ cb, floatS* __restrict__ cwv,
                           int B, int Tr, int Tm, int H, int nr, int nch, int W, float qscale) {
    // one warp per run of PRUN consecutive chunk positions of one (b, h); the conv inputs slide through registers (W = 4)
    constexpr int V = KD / 32, PRUN = 8, CW = 4;
    long gw = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    int npos = nch * CH;
    if (gw * PRUN >= (long)B * H * npos) return;
    int pos0 = (gw * PRUN) % npos; long bh = (gw * PRUN) / npos; int h = bh % H, b = bh / H;
    int T1 = Tr + Tm, KDt = H * KD, Cc = 3 * KDt, ch = h * KD + lane * V;
    long rb = (long)b * T1, csb = (long)b * Cc;
    float wts[3][V][CW];
#pragma unroll
    for (int p = 0; p < 3; p++)
#pragma unroll
        for (int j = 0; j < V; j++)
#pragma unroll
            for (int i = 0; i < CW; i++) wts[p][j][i] = (p == 0 ? wq : (p == 1 ? wk : wv))[(long)(ch + j) * CW + i];
    float win[3][CW - 1][V];   // inputs at t-3, t-2, t-1
    int tprev = -1000;
    float ea = EXPF(A_log[h]);
    for (int r = 0; r < PRUN; r++) {
        int pos = pos0 + r;
        long o = (bh * npos + pos) * KD + lane * V;
        int t = tok_of_pos(pos, Tr, Tm, nr);
        if (t < 0) {
            float z[V] = {};
            stv<V>(cq + o, z); stv<V>(ck + o, z); stv<V>(cb + o, z); stv<V>(cwv + o, z); stv<V>(cg + o, z);
            continue;
        }
        if (t != tprev + 1) {   // (re)load the window
#pragma unroll
            for (int p = 0; p < 3; p++)
#pragma unroll
                for (int s = 0; s < CW - 1; s++) {
                    int tt = t - (CW - 1) + s;
                    if (tt >= 0) ldv<V>(P + (rb + tt) * NP + p * KDt + ch, win[p][s]);
                    else {
#pragma unroll
                        for (int j = 0; j < V; j++) win[p][s][j] = cs_in ? cs_in[(csb + p * KDt + ch + j) * CW + CW + tt] : 0.f;
                    }
                }
        }
        tprev = t;
        long row = rb + t;
        float cur[3][V], y[3][V];
#pragma unroll
        for (int p = 0; p < 3; p++) {
            ldv<V>(P + row * NP + p * KDt + ch, cur[p]);
#pragma unroll
            for (int j = 0; j < V; j++) {
                float a = wts[p][j][CW - 1] * cur[p][j];
#pragma unroll
                for (int s = 0; s < CW - 1; s++) a += wts[p][j][s] * win[p][s][j];
                y[p][j] = silu(a);
            }
#pragma unroll
            for (int s = 0; s < CW - 2; s++)
#pragma unroll
                for (int j = 0; j < V; j++) win[p][s][j] = win[p][s + 1][j];
#pragma unroll
            for (int j = 0; j < V; j++) win[p][CW - 2][j] = cur[p][j];
        }
        float sq = 0.f, sk = 0.f;
#pragma unroll
        for (int j = 0; j < V; j++) { sq += y[0][j] * y[0][j]; sk += y[1][j] * y[1][j]; }
        sq = warp_sum(sq); sk = warp_sum(sk);
        float iq = qscale / sqrtf(sq + 1e-6f), ik = 1.f / sqrtf(sk + 1e-6f);
        float bb[V], ww[V], f[V], g[V];
        ldv<V>(P + row * NP + 3 * KDt + ch, bb);
        ldv<V>(P + row * NP + 4 * KDt + ch, ww);
        ldv<V>(fb + row * KDt + ch, f);
#pragma unroll
        for (int j = 0; j < V; j++) {
            y[0][j] *= iq; y[1][j] *= ik; bb[j] = sigm(bb[j]); ww[j] = sigm(ww[j]) * y[2][j];
            g[j] = -ea * softplus(f[j] + dtb[ch + j]);
        }
        stv<V>(cq + o, y[0]); stv<V>(ck + o, y[1]); stv<V>(cb + o, bb); stv<V>(cwv + o, ww); stv<V>(cg + o, g);
    }
}
// conv state after position Pc: the last W inputs x[Pc-W .. Pc-1] (from the previous state for negative positions)
__global__ void k_conv_state(float* __restrict__ cs_out, const float* __restrict__ cs_in, const floatX* __restrict__ P, int NP, int B, int T1, int Pc, int Cc, int W) {
    long n = (long)B * Cc * W;
    for (long e = (long)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (long)gridDim.x * blockDim.x) {
        int j = e % W; long bc = e / W; int cc = bc % Cc, b = bc / Cc;
        int tt = Pc - W + j;
        cs_out[e] = tt >= 0 ? toF(P[((long)b * T1 + tt) * NP + cc]) : (cs_in ? cs_in[bc * W + W + tt] : 0.f);
    }
}
// backward 1: per (h, block of rows): recompute conv/silu/norms, chunk grads -> dc (pre-silu conv grads, token-major
// [N, Cc]), dP (b, w gate columns), df ; accumulates dA_log, ddt_bias.
template <int KD>
__global__ void __launch_bounds__(256, 2) k_prep_bwd1(const floatX* __restrict__ P, int NP, const floatX* __restrict__ fb,
                            const float* __restrict__ wq, const float* __restrict__ wk, const float* __restrict__ wv, const float* __restrict__ cs_in,
                            const float* __restrict__ A_log, const float* __restrict__ dtb,
                            const floatS* __restrict__ dq, const floatS* __restrict__ dk, const floatS* __restrict__ db, const floatS* __restrict__ dg, const floatS* __restrict__ dwv,
                            floatS* __restrict__ dc, floatX* __restrict__ dP, floatX* __restrict__ df, float* __restrict__ dAlog, float* __restrict__ ddtb,
                            int B, int Tr, int Tm, int H, int nr, int nch, int W, float qscale, int rows_per_block) {
    constexpr int V = KD / 32;
    int h = blockIdx.x, lane = threadIdx.x & 31, wl = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int T1 = Tr + Tm, KDt = H * KD, Cc = 3 * KDt, N = B * T1, ch = h * KD + lane * V;
    float acc_dtb[V], acc_A = 0.f;
#pragma unroll
    for (int j = 0; j < V; j++) acc_dtb[j] = 0.f;
    float ea = EXPF(A_log[h]);
    int r0 = blockIdx.y * rows_per_block, r1 = min(N, r0 + rows_per_block);
    for (int row = r0 + wl; row < r1; row += nw) {
        int b = row / T1, t = row % T1;
        long rb = (long)b * T1, csb = (long)b * Cc, ri = row;
        long o = (((long)b * H + h) * nch * CH + pos_of_tok(t, Tr, nr)) * KD + lane * V;
        float cq[V], ckk[V], cv[V], q[V], k[V], sq = 0.f, sk = 0.f;
        conv_vec<V>(cq, P, NP, ch, wq + (long)ch * W, cs_in, csb + ch, rb, t, W);
        conv_vec<V>(ckk, P, NP, KDt + ch, wk + (long)ch * W, cs_in, csb + KDt + ch, rb, t, W);
        conv_vec<V>(cv, P, NP, 2 * KDt + ch, wv + (long)ch * W, cs_in, csb + 2 * KDt + ch, rb, t, W);
#pragma unroll
        for (int j = 0; j < V; j++) { q[j] = silu(cq[j]); k[j] = silu(ckk[j]); sq += q[j] * q[j]; sk += k[j] * k[j]; }
        sq = warp_sum(sq); sk = warp_sum(sk);
        float nq = sqrtf(sq + 1e-6f), nk = sqrtf(sk + 1e-6f), dotq = 0.f, dotk = 0.f;
        float gq[V], gk[V], gb[V], gg[V], gwv[V];
        ldv<V>(dq + o, gq); ldv<V>(dk + o, gk); ldv<V>(db + o, gb); ldv<V>(dg + o, gg); ldv<V>(dwv + o, gwv);
#pragma unroll
        for (int j = 0; j < V; j++) { gq[j] *= qscale; dotq += q[j] / nq * gq[j]; dotk += k[j] / nk * gk[j]; }
        dotq = warp_sum(dotq); dotk = warp_sum(dotk);
        float bb[V], ww[V], f[V], o_dc[V], o_b[V], o_w[V], o_f[V];
        ldv<V>(P + ri * NP + 3 * KDt + ch, bb);
        ldv<V>(P + ri * NP + 4 * KDt + ch, ww);
        ldv<V>(fb + ri * KDt + ch, f);
#pragma unroll
        for (int j = 0; j < V; j++) o_dc[j] = (gq[j] - q[j] / nq * dotq) / nq * dsilu(cq[j]);
        stv<V>(dc + ri * Cc + ch, o_dc);
#pragma unroll
        for (int j = 0; j < V; j++) o_dc[j] = (gk[j] - k[j] / nk * dotk) / nk * dsilu(ckk[j]);
        stv<V>(dc + ri * Cc + KDt + ch, o_dc);
#pragma unroll
        for (int j = 0; j < V; j++) {
            float bs = sigm(bb[j]), ws = sigm(ww[j]);
            o_dc[j] = gwv[j] * ws * dsilu(cv[j]);
            o_w[j] = gwv[j] * silu(cv[j]) * ws * (1.f - ws);
            o_b[j] = gb[j] * bs * (1.f - bs);
            float x = f[j] + dtb[ch + j];
            acc_A += gg[j] * (-ea * softplus(x));
            o_f[j] = gg[j] * (-ea) * (x > 20.f ? 1.f : sigm(x));
            acc_dtb[j] += o_f[j];
        }
        stv<V>(dc + ri * Cc + 2 * KDt + ch, o_dc);
        stv<V>(dP + ri * NP + 4 * KDt + ch, o_w);
        stv<V>(dP + ri * NP + 3 * KDt + ch, o_b);
        stv<V>(df + ri * KDt + ch, o_f);
    }
    __shared__ float sred[8 * KD + 8];
    acc_A = warp_sum(acc_A);
    if (lane == 0) sred[8 * KD + wl] = acc_A;
#pragma unroll
    for (int j = 0; j < V; j++) sred[wl * KD + lane * V + j] = acc_dtb[j];
    __syncthreads();
    if (threadIdx.x == 0) { float s = 0.f; for (int w = 0; w < nw; w++) s += sred[8 * KD + w]; atomicAdd(dAlog + h, s); }
    for (int c = threadIdx.x; c < KD; c += blockDim.x) {
        float s = 0.f;
        for (int w = 0; w < nw; w++) s += sred[w * KD + c];
        atomicAdd(ddtb + h * KD + c, s);
    }
}
// backward 2: conv input grads. grid (Cc / 128, 1, B): one thread per (channel cc, sample b) walks the tokens with a sliding
// window (W = 4). dX(t) = sum_i w[i] dc(t+W-1-i) (+ carried-state grad) ; dw[i] += sum_t dc(t) X(t-W+1+i) ; d conv state in.
template <int W>
__global__ void __launch_bounds__(128) k_prep_bwd2(const floatX* __restrict__ P, int NP, const floatS* __restrict__ dc,
                                                   const float* __restrict__ convw, const float* __restrict__ cs_in,
                                                   const float* __restrict__ dcs_out, float* __restrict__ dcs_in, floatX* __restrict__ dP,
                                                   float* __restrict__ dconvw, int B, int T1, int Pc, int Cc) {
    int cc = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.z;
    if (cc >= Cc) return;
    float w[W], dw[W], g[W], x[W];   // g[j] = dc(t + j), x[j] = X(t - W + 1 + j)
#pragma unroll
    for (int i = 0; i < W; i++) { w[i] = convw[cc * W + i]; dw[i] = 0.f; }
    long rb = (long)b * T1, sb = ((long)b * Cc + cc) * W;
    auto DC = [&](int t) { return t < T1 ? toF(dc[(rb + t) * Cc + cc]) : 0.f; };
    auto XX = [&](int t) { return t >= 0 ? toF(P[(rb + t) * NP + cc]) : (cs_in ? cs_in[sb + W + t] : 0.f); };
    // gradient of the incoming conv state: entry s holds x(s - W)
    if (dcs_in) {
#pragma unroll
        for (int s = 0; s < W; s++) {
            int tp = s - W;
            float acc = 0.f;
#pragma unroll
            for (int i = 0; i < W; i++) { int tt = tp + W - 1 - i; if (tt >= 0) acc += w[i] * DC(tt); }
            int j = tp - (Pc - W);
            if (dcs_out && j >= 0 && j < W) acc += dcs_out[sb + j];
            dcs_in[sb + s] = acc;
        }
    }
#pragma unroll
    for (int j = 0; j < W; j++) { g[j] = DC(j); x[j] = XX(j - W + 1); }
    for (int t = 0; t < T1; t++) {
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < W; i++) acc += w[i] * g[W - 1 - i];
        int j = t - (Pc - W);
        if (dcs_out && j >= 0 && j < W) acc += dcs_out[sb + j];
        stX(dP + (rb + t) * NP + cc, acc);
#pragma unroll
        for (int i = 0; i < W; i++) dw[i] += g[0] * x[i];
#pragma unroll
        for (int i = 0; i < W - 1; i++) { g[i] = g[i + 1]; x[i] = x[i + 1]; }
        g[W - 1] = DC(t + W);
        x[W - 1] = XX(t + 1);
    }
#pragma unroll
    for (int i = 0; i < W; i++) atomicAdd(dconvw + cc * W + i, dw[i]);
}

// ---------------------------------------------------------------- gated output norm
template <int KD>
__global__ void k_onorm_fwd(const floatS* __restrict__ co, const floatX* __restrict__ gg, const float* __restrict__ g2b, const float* __restrict__ won, floatX* __restrict__ on, float* __restrict__ rstd,
                            int B, int Tr, int Tm, int H, int nr, int nch, float eps, int swish) {
    constexpr int V = KD / 32;
    long gw = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31, T1 = Tr + Tm;
    if (gw >= (long)B * T1 * H) return;
    int h = gw % H; long row = gw / H; int b = row / T1, t = row % T1;
    long o = (((long)b * H + h) * nch * CH + pos_of_tok(t, Tr, nr)) * KD + lane * V;
    int VDt = H * KD, c0 = lane * V, ch = h * KD + c0;
    float x[V], g[V], y[V], ss = 0.f;
    ldv<V>(co + o, x);
    ldv<V>(gg + row * VDt + ch, g);
#pragma unroll
    for (int j = 0; j < V; j++) ss += x[j] * x[j];
    ss = warp_sum(ss);
    float rs = rsqrtf(ss / KD + eps);
    if (lane == 0) rstd[gw] = rs;
#pragma unroll
    for (int j = 0; j < V; j++) {
        float gv = g[j] + g2b[ch + j];
        float a = swish ? gv * sigm(gv) : sigm(gv);
        y[j] = x[j] * rs * won[c0 + j] * a;
    }
    stv<V>(on + row * VDt + ch, y);
}
// backward: grid (H, blocks over chunk positions B*nch*CH); writes do (zero on padding), dgg; dw_on, dg2b
template <int KD>
__global__ void k_onorm_bwd(const floatS* __restrict__ co, const floatX* __restrict__ gg, const float* __restrict__ g2b, const float* __restrict__ won, const floatX* __restrict__ don,
                            floatS* __restrict__ dco, floatX* __restrict__ dgg, float* __restrict__ dwon, float* __restrict__ dg2b,
                            int B, int Tr, int Tm, int H, int nr, int nch, float eps, int swish, int per_block) {
    constexpr int V = KD / 32;
    int h = blockIdx.x, lane = threadIdx.x & 31, wl = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int T1 = Tr + Tm, npos = nch * CH, VDt = H * KD, c0 = lane * V, ch = h * KD + c0;
    float aw[V], ab[V];
#pragma unroll
    for (int j = 0; j < V; j++) aw[j] = ab[j] = 0.f;
    long p0 = (long)blockIdx.y * per_block, p1 = min((long)B * npos, p0 + per_block);
    for (long p = p0 + wl; p < p1; p += nw) {
        int b = p / npos, pos = p % npos;
        long o = (((long)b * H + h) * npos + pos) * KD + c0;
        int t = tok_of_pos(pos, Tr, Tm, nr);
        if (t < 0) { float z[V] = {}; stv<V>(dco + o, z); continue; }
        long row = (long)b * T1 + t;
        float x[V], g[V], d[V], dxh[V], dgo[V], ss = 0.f, dot = 0.f;
        ldv<V>(co + o, x); ldv<V>(gg + row * VDt + ch, g); ldv<V>(don + row * VDt + ch, d);
#pragma unroll
        for (int j = 0; j < V; j++) ss += x[j] * x[j];
        ss = warp_sum(ss);
        float rs = rsqrtf(ss / KD + eps);
#pragma unroll
        for (int j = 0; j < V; j++) {
            float gv = g[j] + g2b[ch + j], s = sigm(gv);
            float a = swish ? gv * s : s, da = swish ? s * (1.f + gv * (1.f - s)) : s * (1.f - s);
            float xh = x[j] * rs;
            dxh[j] = d[j] * won[c0 + j] * a;
            dot += dxh[j] * xh;
            dgo[j] = d[j] * xh * won[c0 + j] * da;
            aw[j] += d[j] * xh * a;
            ab[j] += dgo[j];
        }
        stv<V>(dgg + row * VDt + ch, dgo);
        dot = warp_sum(dot) / KD;
#pragma unroll
        for (int j = 0; j < V; j++) dxh[j] = rs * (dxh[j] - x[j] * rs * dot);
        stv<V>(dco + o, dxh);
    }
    __shared__ float sred[2][8 * KD];
#pragma unroll
    for (int j = 0; j < V; j++) { sred[0][wl * KD + c0 + j] = aw[j]; sred[1][wl * KD + c0 + j] = ab[j]; }
    __syncthreads();
    for (int c = threadIdx.x; c < KD; c += blockDim.x) {
        float s0 = 0.f, s1 = 0.f;
        for (int w = 0; w < nw; w++) { s0 += sred[0][w * KD + c]; s1 += sred[1][w * KD + c]; }
        atomicAdd(dwon + c, s0);
        atomicAdd(dg2b + h * KD + c, s1);
    }
}

// ---------------------------------------------------------------- MLP
__global__ void k_swiglu(floatX* __restrict__ m, const floatX* __restrict__ a, long N, int F) {
    long n = N * F;
    for (long e = (long)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (long)gridDim.x * blockDim.x) {
        long r = e / F; int j = e % F;
        float a1 = toF(a[r * 2 * F + j]), a2 = toF(a[r * 2 * F + F + j]);
        stX(m + e, silu(a1) * a2);
    }
}
__global__ void k_swiglu_bwd(floatX* __restrict__ da, const floatX* __restrict__ dm, const floatX* __restrict__ a, long N, int F) {
    long n = N * F;
    for (long e = (long)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (long)gridDim.x * blockDim.x) {
        long r = e / F; int j = e % F;
        float a1 = toF(a[r * 2 * F + j]), a2 = toF(a[r * 2 * F + F + j]), g = toF(dm[e]);
        stX(da + r * 2 * F + j, g * a2 * dsilu(a1));
        stX(da + r * 2 * F + F + j, g * silu(a1));
    }
}
// residual add: y = a + b (fp32 + floatX)
__global__ void k_resid(float* __restrict__ y, const float* __restrict__ a, const floatX* __restrict__ b, long n) {
    for (long e = (long)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (long)gridDim.x * blockDim.x) y[e] = a[e] + toF(b[e]);
}

// ---------------------------------------------------------------- loss
// targets of the rows of one piece: row (b, t) -> labels[b, start + t + 1] (or -100 past the end)
__global__ void k_targets(int64_t* tgt, const int64_t* labels, int B, int Tr, int T, int start) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= B * Tr) return;
    int b = r / Tr, p = start + r % Tr;
    tgt[r] = p + 1 < T ? labels[(long)b * T + p + 1] : -100;
}
// cross entropy over a row of logits; optional in-place gradient (softmax - onehot) * inv_n (0 for ignored rows).
// row r reads logits at ptr + map(r) * V where map(r) = (r / rpb) * ld_b + off + r % rpb
// targets / token losses use the piece row gr = roff + r: (b = gr / Tr, p = start + gr % Tr)
template <typename T>
__global__ void k_ce(T* logits, int V, int rpb, long ld_b, long off, int roff, const int64_t* tgt, float* tl, int Tr, int T_minus_1,
                     int start, const float* inv_n, int write_grad) {
    int r = blockIdx.x, gr = roff + r;
    long lrow = rpb > 0 ? (long)(r / rpb) * ld_b + off + r % rpb : r;
    T* x = logits + lrow * V;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < V; j += blockDim.x) mx = fmaxf(mx, toF(x[j]));
    __shared__ float smx[33];
    {
        int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        if (lane == 0) smx[w] = mx;
        __syncthreads();
        if (threadIdx.x == 0) { float m = -INFINITY; for (int i = 0; i < (int)(blockDim.x >> 5); i++) m = fmaxf(m, smx[i]); smx[32] = m; }
        __syncthreads();
        mx = smx[32];
    }
    float s = 0.f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) s += expf(toF(x[j]) - mx);
    s = block_sum(s);
    float lse = mx + logf(s);
    int64_t y = tgt[gr];
    if (y >= V) y = -100;
    if (threadIdx.x == 0 && tl) {
        int b = gr / Tr, p = start + gr % Tr;
        if (p < T_minus_1) tl[(long)b * T_minus_1 + p] = y >= 0 ? lse - toF(x[y]) : 0.f;
    }
    if (write_grad) {
        __syncthreads();
        float sc = y >= 0 ? *inv_n : 0.f;
        for (int j = threadIdx.x; j < V; j += blockDim.x) {
            float g = expf(toF(x[j]) - lse) - (j == y ? 1.f : 0.f);
            stX(x + j, g * sc);
        }
    }
}

// =============================================================================================== GDN-2 scan kernels
// intra-chunk forward. grid = number of (b, h, chunk), block 256, dyn smem K1_SMEM floats.
// Sub-chunks of 16 rows: for an off-diagonal block (rows I > columns J) the decay e^{gc_i - gc_j} is factored through the
// reference r = 16 I - 1 (j <= r < i) as e^{gc_i - gc_r} e^{gc_r - gc_j}: both factors <= 1 (no overflow) and the block is a
// plain product of pre-scaled operands (tensor cores in the bf16 build); only the 4 diagonal blocks use exact pairwise
// exponentials. A = (I + L)^-1 is inverted blockwise: the 4 diagonal 16x16 blocks by substitution, then the off-diagonal ones.
#define SB 16
#define DT 16
#define LDL 68
#define LDO 24
#define K1_TILES (4 * CH * (DT + 1) + (int)((2 * 48 * LDO + 96 * LDO) * sizeof(floatS) / 4))
#define K1_POST (CH * LDL + 3 * SB * (SB + 1))
#define K1_SMEM (CH * LDL + (K1_TILES > K1_POST ? K1_TILES : K1_POST) + 256)
template <int KD>
__global__ void __launch_bounds__(256, 2) k_intra_fwd(const floatS* __restrict__ cq, const floatS* __restrict__ ck,
                                                      const float* __restrict__ cg, const floatS* __restrict__ cb,
                                                      float* __restrict__ gc, floatS* __restrict__ y, floatS* __restrict__ Qg, floatS* __restrict__ Kt, float* __restrict__ Am, floatS* __restrict__ AmX, floatS* __restrict__ Mm,
                                                      float* __restrict__ gl) {
    extern __shared__ __align__(16) float sm[];
    float* sL = sm;                       // [CH][LDL]
    float* un = sL + CH * LDL;            // union: d-tiles | sM | (sA + sT)
    float* sq = un; float* sk = sq + CH * (DT + 1); float* sbk = sk + CH * (DT + 1); float* sgc = sbk + CH * (DT + 1);
    floatS* sQh = (floatS*)(sgc + CH * (DT + 1));   // [48][LDO] rows 16..63: q_i e^{gc_i - gc_r(I)}
    floatS* sBh = sQh + 48 * LDO;                   // [48][LDO] same with b k
    floatS* sKr = sBh + 48 * LDO;                   // [96][LDO] k_j e^{gc_r(I) - gc_j}, rows 8 I (I - 1) + j (I = 1..3, j < 16 I)
    float* sM = un;                                 // [CH][LDL] after the tile loop
    float* sA = un; float* sT = sA + CH * LDL;      // after M is written out
    float* sgl = sm + K1_SMEM - 256;
    const long cid = blockIdx.x, base = cid * CH * KD;
    const int tid = threadIdx.x;
    // 1. gc = cumsum(g) over the chunk rows (256/KD threads per column)
    {
        constexpr int TPC = 256 / KD, RPT = CH / TPC;
        int d = tid % KD, part = tid / KD;
        float s = 0.f, loc[RPT];
#pragma unroll
        for (int r = 0; r < RPT; r++) { s += cg[base + (part * RPT + r) * KD + d]; loc[r] = s; }
        sL[part * KD + d] = s;
        __syncthreads();
        float off = 0.f;
        for (int p = 0; p < part; p++) off += sL[p * KD + d];
#pragma unroll
        for (int r = 0; r < RPT; r++) gc[base + (part * RPT + r) * KD + d] = loc[r] + off;
        if (part == TPC - 1) { sgl[d] = loc[RPT - 1] + off; gl[cid * KD + d] = loc[RPT - 1] + off; }
        __syncthreads();
    }
    // 2. pairwise sums
    // diagonal blocks: thread -> (block I, row rr, column group jg): columns j = jg + 4 q (q < 4) of the block
    const int dI = tid >> 6, drr = (tid >> 2) & 15, djg = tid & 3, di = 16 * dI + drr;
    float dLa[4] = {}, dMa[4] = {};
#ifndef SCAN32
    // off-diagonal blocks: 12 (pair, matrix) jobs on tensor cores; warp w does jobs w and w + 8
    const int warp = tid >> 5;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> facc[2];
    wmma::fill_fragment(facc[0], 0.f); wmma::fill_fragment(facc[1], 0.f);

#else
    // fp32 build: off-diagonal on CUDA cores. thread -> (pair p = tid / 43 ..) simple mapping: row i, column block J < I
    const int i = tid >> 2, J = tid & 3, I = i >> 4;
    float aL[SB] = {}, aM[SB] = {};
#endif
    for (int d0 = 0; d0 < KD; d0 += DT) {
        for (int e = tid; e < CH * DT; e += 256) {
            int c = e / DT, x = e % DT;
            long g_ = base + c * KD + d0 + x;
            float q = toF(cq[g_]), k = toF(ck[g_]), b = toF(cb[g_]), g = gc[g_];
            sq[c * (DT + 1) + x] = q; sk[c * (DT + 1) + x] = k; sbk[c * (DT + 1) + x] = b * k; sgc[c * (DT + 1) + x] = g;
            float eg = EXPF(g);
            stX(y + g_, b * k * eg); stX(Qg + g_, q * eg); stX(Kt + g_, k * EXPF(sgl[d0 + x] - g));
        }
        __syncthreads();
        for (int e = tid; e < 96 * DT; e += 256) {
            int row = e / DT, x = e % DT;
            int Ir = row < 16 ? 1 : (row < 48 ? 2 : 3), j = row - 8 * Ir * (Ir - 1);
            float gr = sgc[(16 * Ir - 1) * (DT + 1) + x];
            stX(sKr + row * LDO + x, sk[j * (DT + 1) + x] * EXPF(gr - sgc[j * (DT + 1) + x]));
            if (row < 48) {
                int ir = 16 + row, Ii = ir >> 4;
                float ef = EXPF(sgc[ir * (DT + 1) + x] - sgc[(16 * Ii - 1) * (DT + 1) + x]);
                stX(sQh + row * LDO + x, sq[ir * (DT + 1) + x] * ef);
                stX(sBh + row * LDO + x, sbk[ir * (DT + 1) + x] * ef);
            }
        }
        __syncthreads();
        // diagonal blocks (exact)
#pragma unroll 4
        for (int x = 0; x < DT; x++) {
            float gi = sgc[di * (DT + 1) + x], qi = sq[di * (DT + 1) + x], bi = sbk[di * (DT + 1) + x];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                int jj = djg + 4 * q;
                if (jj <= drr) {
                    int j = 16 * dI + jj;
                    float k = sk[j * (DT + 1) + x] * EXPF(gi - sgc[j * (DT + 1) + x]);
                    dLa[q] += bi * k; dMa[q] += qi * k;
                }
            }
        }
#ifndef SCAN32
#pragma unroll
        for (int s2 = 0; s2 < 2; s2++) {
            int jb = warp + 8 * s2;
            if (jb < 12) {
                int p = jb >> 1, mat = jb & 1, I = p < 1 ? 1 : (p < 3 ? 2 : 3), J = p - I * (I - 1) / 2;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, floatS, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, floatS, wmma::col_major> b;
                wmma::load_matrix_sync(a, (mat ? sQh : sBh) + (16 * I - 16) * LDO, LDO);
                wmma::load_matrix_sync(b, sKr + (8 * I * (I - 1) + 16 * J) * LDO, LDO);
                wmma::mma_sync(facc[s2], a, b, facc[s2]);
            }
        }
#else
        if (J < I) {
            const floatS* kr = sKr + (8 * I * (I - 1) + 16 * J) * LDO;
            const floatS* qh = sQh + (i - 16) * LDO; const floatS* bh = sBh + (i - 16) * LDO;
            for (int x = 0; x < DT; x++) {
                float qv = toF(qh[x]), bv = toF(bh[x]);
#pragma unroll
                for (int jj = 0; jj < SB; jj++) { float k = toF(kr[jj * LDO + x]); aL[jj] += bv * k; aM[jj] += qv * k; }
            }
        }
#endif
        __syncthreads();
    }
    // 3. assemble L (sL) and M (sM -> global): zero, off-diagonal blocks, diagonal blocks
    for (int e = tid; e < CH * LDL; e += 256) { sL[e] = 0.f; sM[e] = 0.f; }
    __syncthreads();
#ifndef SCAN32
#pragma unroll
    for (int s2 = 0; s2 < 2; s2++) {
        int jb = warp + 8 * s2;
        if (jb < 12) {
            int p = jb >> 1, mat = jb & 1, I = p < 1 ? 1 : (p < 3 ? 2 : 3), J = p - I * (I - 1) / 2;
            wmma::store_matrix_sync((mat ? sM : sL) + 16 * I * LDL + 16 * J, facc[s2], LDL, wmma::mem_row_major);
        }
    }
#else
    if (J < I)
        for (int jj = 0; jj < SB; jj++) { sL[i * LDL + 16 * J + jj] = aL[jj]; sM[i * LDL + 16 * J + jj] = aM[jj]; }
#endif
#pragma unroll
    for (int q = 0; q < 4; q++) {
        int jj = djg + 4 * q, j = 16 * dI + jj;
        if (jj < drr) sL[di * LDL + j] = dLa[q];
        if (jj <= drr) sM[di * LDL + j] = dMa[q];
    }
    __syncthreads();
    floatS* Mo = Mm + cid * CH * CH;
    for (int e = tid; e < CH * CH; e += 256) stX(Mo + e, sM[(e / CH) * LDL + e % CH]);
    __syncthreads();
    for (int e = tid; e < CH * LDL; e += 256) sA[e] = 0.f;
    __syncthreads();
    // 4a. diagonal blocks: column c of A_II by forward substitution (one thread per (I, c))
    if (tid < 64) {
        int Ib = tid >> 4, c = tid & 15, o = 16 * Ib;
        float col[SB];
#pragma unroll
        for (int r = 0; r < SB; r++) col[r] = r == c ? 1.f : 0.f;
#pragma unroll
        for (int r = 1; r < SB; r++) {
            float s = col[r];
#pragma unroll
            for (int j = 0; j < SB; j++) if (j < r) s -= sL[(o + r) * LDL + o + j] * col[j];
            col[r] = r > c ? s : col[r];
        }
#pragma unroll
        for (int r = 0; r < SB; r++) sA[(o + r) * LDL + o + c] = col[r];
    }
    __syncthreads();
    // 4b. off-diagonal blocks by distance: A_IJ = -A_II sum_{K=J}^{I-1} L_IK A_KJ
    for (int dist = 1; dist < 4; dist++) {
        int nb = 4 - dist;
        for (int e = tid; e < nb * 256; e += 256) {
            int bI = e / 256 + dist, bJ = bI - dist, r = (e % 256) / 16, cc = e % 16;
            float s = 0.f;
            for (int K = bJ; K < bI; K++)
#pragma unroll
                for (int m = 0; m < 16; m++) s += sL[(16 * bI + r) * LDL + 16 * K + m] * sA[(16 * K + m) * LDL + 16 * bJ + cc];
            sT[(e / 256) * 16 * 17 + r * 17 + cc] = s;
        }
        __syncthreads();
        for (int e = tid; e < nb * 256; e += 256) {
            int bI = e / 256 + dist, bJ = bI - dist, r = (e % 256) / 16, cc = e % 16;
            float s = 0.f;
#pragma unroll
            for (int m = 0; m < 16; m++) s -= sA[(16 * bI + r) * LDL + 16 * bI + m] * sT[(e / 256) * 16 * 17 + m * 17 + cc];
            sA[(16 * bI + r) * LDL + 16 * bJ + cc] = s;
        }
        __syncthreads();
    }
    float* Ao = Am + cid * CH * CH;
    for (int e = tid; e < CH * CH; e += 256) {
        float v = sA[(e / CH) * LDL + e % CH];
        Ao[e] = v;
        if (AmX) stX(AmX + cid * CH * CH + e, v);
    }
}

#ifdef SCAN32
// sequential state pass, forward. grid (B*H, KD/32), block 256. dyn smem (KD*32 + KD*(CH+1) + CH*32) floats.
// S0[n] = state at the start of chunk n ; vn = u - Wm S0 ; S1 = e^{gl} S0 + Kt^T vn. S_out <- state at start of chunk ncarry.
template <int KD>
__global__ void __launch_bounds__(256) k_state_fwd(const float* __restrict__ u, const float* __restrict__ Wm, const float* __restrict__ Kt,
                                                   const float* __restrict__ gl, float* __restrict__ S0, float* __restrict__ vn,
                                                   const float* __restrict__ S_in, float* __restrict__ S_out, int ncarry, int nch) {
    extern __shared__ float sm[];
    float* sS = sm; float* sB = sS + KD * 32; float* sV = sB + KD * (CH + 1);
    const int VD = KD;
    const long bh = blockIdx.x;
    const int v0 = blockIdx.y * 32, tid = threadIdx.x, v = tid & 31, grp = tid >> 5;
    constexpr int R = KD / 8;
    for (int e = tid; e < KD * 32; e += 256) { int k = e >> 5, vv = e & 31; sS[e] = S_in ? S_in[bh * KD * VD + (long)k * VD + v0 + vv] : 0.f; }
    __syncthreads();
    for (int n = 0; n < nch; n++) {
        long cb = bh * nch + n;
        if (n == ncarry && S_out)
            for (int e = tid; e < KD * 32; e += 256) { int k = e >> 5, vv = e & 31; S_out[bh * KD * VD + (long)k * VD + v0 + vv] = sS[e]; }
        for (int e = tid; e < KD * 32; e += 256) { int k = e >> 5, vv = e & 31; S0[cb * KD * VD + (long)k * VD + v0 + vv] = sS[e]; }
        const float* W_ = Wm + cb * CH * KD;
        for (int e = tid; e < CH * KD; e += 256) { int c = e / KD, k = e % KD; sB[k * (CH + 1) + c] = W_[e]; }
        __syncthreads();
        float acc[8];
#pragma unroll
        for (int r = 0; r < 8; r++) acc[r] = u[cb * CH * VD + (long)(grp * 8 + r) * VD + v0 + v];
        for (int k = 0; k < KD; k++) {
            float s = sS[k * 32 + v];
            const float* w = sB + k * (CH + 1) + grp * 8;
#pragma unroll
            for (int r = 0; r < 8; r++) acc[r] -= w[r] * s;
        }
#pragma unroll
        for (int r = 0; r < 8; r++) { int c = grp * 8 + r; sV[c * 32 + v] = acc[r]; vn[cb * CH * VD + (long)c * VD + v0 + v] = acc[r]; }
        __syncthreads();
        const float* K_ = Kt + cb * CH * KD;
        for (int e = tid; e < CH * KD; e += 256) sB[e] = K_[e];
        __syncthreads();
        float a2[R];
#pragma unroll
        for (int r = 0; r < R; r++) { int k = grp * R + r; a2[r] = EXPF(gl[cb * KD + k]) * sS[k * 32 + v]; }
        for (int c = 0; c < CH; c++) {
            float x = sV[c * 32 + v];
            const float* kt = sB + c * KD + grp * R;
#pragma unroll
            for (int r = 0; r < R; r++) a2[r] += kt[r] * x;
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < R; r++) sS[(grp * R + r) * 32 + v] = a2[r];
        __syncthreads();
    }
    if (ncarry == nch && S_out)
        for (int e = tid; e < KD * 32; e += 256) { int k = e >> 5, vv = e & 31; S_out[bh * KD * VD + (long)k * VD + v0 + vv] = sS[e]; }
}
// sequential state pass, backward (reverse over chunks). dSe in: Qg^T do per chunk ; out: grad of the state at the END of
// chunk n. dvn = pre_dvn (M^T do) + Kt dS ; dS <- e^{gl} dS + Qg^T do - Wm^T dvn. The carried-state grad joins at the end
// of chunk ncarry-1. dS_in out: grad of the initial state (may alias dS_out).
template <int KD>
__global__ void __launch_bounds__(256) k_state_bwd(const float* __restrict__ pre_dvn, float* dvn, float* dSe,
                                                   const float* __restrict__ Kt, const float* __restrict__ Wm, const float* __restrict__ gl,
                                                   const float* dS_out, int ncarry, float* dS_in, int nch) {
    extern __shared__ float sm[];
    float* sS = sm; float* sB = sS + KD * 32; float* sV = sB + KD * (CH + 1);
    const int VD = KD;
    const long bh = blockIdx.x;
    const int v0 = blockIdx.y * 32, tid = threadIdx.x, v = tid & 31, grp = tid >> 5;
    constexpr int R = KD / 8;
    for (int e = tid; e < KD * 32; e += 256) sS[e] = 0.f;
    __syncthreads();
    for (int n = nch - 1; n >= 0; n--) {
        long cb = bh * nch + n;
        if (n == ncarry - 1 && dS_out)
            for (int e = tid; e < KD * 32; e += 256) { int k = e >> 5, vv = e & 31; sS[e] += dS_out[bh * KD * VD + (long)k * VD + v0 + vv]; }
        __syncthreads();
        float p[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            int k = grp * R + r;
            long gi = cb * KD * VD + (long)k * VD + v0 + v;
            p[r] = dSe[gi];
            dSe[gi] = sS[k * 32 + v];
        }
        const float* K_ = Kt + cb * CH * KD;
        for (int e = tid; e < CH * KD; e += 256) { int c = e / KD, k = e % KD; sB[k * (CH + 1) + c] = K_[e]; }
        __syncthreads();
        float acc[8];
#pragma unroll
        for (int r = 0; r < 8; r++) acc[r] = pre_dvn[cb * CH * VD + (long)(grp * 8 + r) * VD + v0 + v];
        for (int k = 0; k < KD; k++) {
            float s = sS[k * 32 + v];
            const float* w = sB + k * (CH + 1) + grp * 8;
#pragma unroll
            for (int r = 0; r < 8; r++) acc[r] += w[r] * s;
        }
#pragma unroll
        for (int r = 0; r < 8; r++) { int c = grp * 8 + r; sV[c * 32 + v] = acc[r]; dvn[cb * CH * VD + (long)c * VD + v0 + v] = acc[r]; }
        __syncthreads();
        const float* W_ = Wm + cb * CH * KD;
        for (int e = tid; e < CH * KD; e += 256) sB[e] = W_[e];
        __syncthreads();
        float a2[R];
#pragma unroll
        for (int r = 0; r < R; r++) { int k = grp * R + r; a2[r] = EXPF(gl[cb * KD + k]) * sS[k * 32 + v] + p[r]; }
        for (int c = 0; c < CH; c++) {
            float x = sV[c * 32 + v];
            const float* w = sB + c * KD + grp * R;
#pragma unroll
            for (int r = 0; r < R; r++) a2[r] -= w[r] * x;
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < R; r++) sS[(grp * R + r) * 32 + v] = a2[r];
        __syncthreads();
    }
    for (int e = tid; e < KD * 32; e += 256) { int k = e >> 5, vv = e & 31; dS_in[bh * KD * VD + (long)k * VD + v0 + vv] = sS[e]; }
}
#else
// sequential state passes on tensor cores (bf16 build): bf16 operands, fp32 accumulation, the state itself stays fp32.
// grid (B*H, KD/32), block 256 (8 warps). smem: sS f32 [KD][32] | sSb bf16 [KD][40] | sWK bf16 [CH][KD+8] | sV f32 [CH][32] |
// sVb bf16 [CH][40]
typedef __nv_bfloat16 bf16;
#ifndef TVS
#define TVS 32      // state columns per block
#endif
#define TC_LDV (TVS + 8)
// ARMT_SPLIT_STATE: the state enters the forward products as hi + lo bf16 parts (~fp32 accuracy). Measured on the 370M
// model it changes neither the loss nor the gradient error distribution (bf16 noise dominates) and costs ~8% of a
// training step, so it is off by default.
#ifdef ARMT_SPLIT_STATE
#define TC_SPLIT_SMEM(KD) (KD * TC_LDV * 2)
#else
#define TC_SPLIT_SMEM(KD) 0
#endif
template <int KD> constexpr int tc_smem() { return KD * TVS * 4 + KD * TC_LDV * 2 + CH * (KD + 8) * 2 + CH * TVS * 4 + CH * TC_LDV * 2
                                                   + TC_SPLIT_SMEM(KD); }
// load a [CH][KD] bf16 chunk into smem with row stride KD+8 (optionally negated)
template <int KD>
static __device__ __forceinline__ void tc_load_chunk(bf16* dst, const bf16* src, bool neg) {
    constexpr int R8 = KD / 8;
    for (int e = threadIdx.x; e < CH * R8; e += 256) {
        int r = e / R8, c8 = e % R8;
        uint4 v = ((const uint4*)(src + (long)r * KD))[c8];
        if (neg) {
            __nv_bfloat162* h = (__nv_bfloat162*)&v;
#pragma unroll
            for (int j = 0; j < 4; j++) h[j] = __hneg2(h[j]);
        }
        *(uint4*)(dst + r * (KD + 8) + c8 * 8) = v;
    }
}
template <int KD>
__global__ void __launch_bounds__(256) k_state_fwd(const bf16* __restrict__ u, const bf16* __restrict__ Wm, const bf16* __restrict__ Kt,
                                                   const float* __restrict__ gl, bf16* __restrict__ S0, bf16* __restrict__ S0lo,
                                                   bf16* __restrict__ vn,
                                                   const float* __restrict__ S_in, float* __restrict__ S_out, int ncarry, int nch) {
    // with ARMT_SPLIT_STATE the state enters the products as hi + lo (two bf16 parts, ~fp32 accuracy)
    extern __shared__ __align__(16) unsigned char smraw[];
    constexpr int LDK = KD + 8, VD = KD;
    float* sS = (float*)smraw; bf16* sSb = (bf16*)(sS + KD * TVS); bf16* sWK = sSb + KD * TC_LDV;
    float* sV = (float*)(sWK + CH * LDK); bf16* sVb = (bf16*)(sV + CH * TVS); bf16* sSl = sVb + CH * TC_LDV;
    const long bh = blockIdx.x;
    const int v0 = blockIdx.y * TVS, tid = threadIdx.x, warp = tid >> 5;
    for (int e = tid; e < KD * TVS; e += 256) { int k = e / TVS, vv = e % TVS; sS[e] = S_in ? S_in[bh * KD * VD + (long)k * VD + v0 + vv] : 0.f; }
    __syncthreads();
    for (int n = 0; n < nch; n++) {
        long cb = bh * nch + n;
        for (int e = tid; e < KD * TVS; e += 256) {
            int k = e / TVS, vv = e % TVS;
            if (n == ncarry && S_out) S_out[bh * KD * VD + (long)k * VD + v0 + vv] = sS[e];
            bf16 x = __float2bfloat16(sS[e]);
            sSb[k * TC_LDV + vv] = x;
            S0[cb * KD * VD + (long)k * VD + v0 + vv] = x;
#ifdef ARMT_SPLIT_STATE
            bf16 xl = __float2bfloat16(sS[e] - __bfloat162float(x));
            sSl[k * TC_LDV + vv] = xl;
            S0lo[cb * KD * VD + (long)k * VD + v0 + vv] = xl;
#endif
        }
        tc_load_chunk<KD>(sWK, Wm + cb * CH * KD, true);
        for (int e = tid; e < CH * TVS; e += 256) { int c = e / TVS, vv = e % TVS; sV[e] = __bfloat162float(u[cb * CH * VD + (long)c * VD + v0 + vv]); }
        __syncthreads();
        for (int tile = warp; tile < (CH / 16) * (TVS / 16); tile += 8) {   // vn = u - Wm S
            int m = tile / (TVS / 16), nn = tile % (TVS / 16);
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::load_matrix_sync(acc, sV + 16 * m * TVS + 16 * nn, TVS, wmma::mem_row_major);
#pragma unroll
            for (int kk = 0; kk < KD / 16; kk++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
                wmma::load_matrix_sync(a, sWK + 16 * m * LDK + 16 * kk, LDK);
                wmma::load_matrix_sync(b, sSb + 16 * kk * TC_LDV + 16 * nn, TC_LDV);
                wmma::mma_sync(acc, a, b, acc);
#ifdef ARMT_SPLIT_STATE
                wmma::load_matrix_sync(b, sSl + 16 * kk * TC_LDV + 16 * nn, TC_LDV);
                wmma::mma_sync(acc, a, b, acc);
#endif
            }
            wmma::store_matrix_sync(sV + 16 * m * TVS + 16 * nn, acc, TVS, wmma::mem_row_major);
        }
        __syncthreads();
        for (int e = tid; e < CH * TVS; e += 256) {
            int c = e / TVS, vv = e % TVS;
            bf16 x = __float2bfloat16(sV[e]);
            sVb[c * TC_LDV + vv] = x;
            vn[cb * CH * VD + (long)c * VD + v0 + vv] = x;
        }
        for (int e = tid; e < KD * TVS; e += 256) sS[e] *= EXPF(gl[cb * KD + (e / TVS)]);
        tc_load_chunk<KD>(sWK, Kt + cb * CH * KD, false);
        __syncthreads();
        // S += Kt^T vn : (KD/16) x 2 tiles
        for (int tile = warp; tile < (KD / 16) * (TVS / 16); tile += 8) {
            int mt = tile / (TVS / 16), nt = tile % (TVS / 16);
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::load_matrix_sync(acc, sS + 16 * mt * TVS + 16 * nt, TVS, wmma::mem_row_major);
#pragma unroll
            for (int kk = 0; kk < CH / 16; kk++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::col_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
                wmma::load_matrix_sync(a, sWK + 16 * kk * LDK + 16 * mt, LDK);
                wmma::load_matrix_sync(b, sVb + 16 * kk * TC_LDV + 16 * nt, TC_LDV);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sS + 16 * mt * TVS + 16 * nt, acc, TVS, wmma::mem_row_major);
        }
        __syncthreads();
    }
    if (ncarry == nch && S_out)
        for (int e = tid; e < KD * TVS; e += 256) { int k = e / TVS, vv = e % TVS; S_out[bh * KD * VD + (long)k * VD + v0 + vv] = sS[e]; }
}
// reverse pass: dvn = pre_dvn + Kt dS ; dS <- e^{gl} dS + pre_dS - Wm^T dvn. dSe: in pre_dS (Qg^T do), out dS at chunk ends.
template <int KD>
__global__ void __launch_bounds__(256) k_state_bwd(const bf16* __restrict__ pre_dvn, bf16* dvn, bf16* dSe,
                                                   const bf16* __restrict__ Kt, const bf16* __restrict__ Wm, const float* __restrict__ gl,
                                                   const float* dS_out, int ncarry, float* dS_in, int nch) {
    extern __shared__ __align__(16) unsigned char smraw[];
    constexpr int LDK = KD + 8, VD = KD, PR = KD * TVS / 256;
    float* sS = (float*)smraw; bf16* sSb = (bf16*)(sS + KD * TVS); bf16* sWK = sSb + KD * TC_LDV;
    float* sV = (float*)(sWK + CH * LDK); bf16* sVb = (bf16*)(sV + CH * TVS);
    const long bh = blockIdx.x;
    const int v0 = blockIdx.y * TVS, tid = threadIdx.x, warp = tid >> 5;
    for (int e = tid; e < KD * TVS; e += 256) sS[e] = 0.f;
    __syncthreads();
    for (int n = nch - 1; n >= 0; n--) {
        long cb = bh * nch + n;
        float p[PR];
#pragma unroll
        for (int r = 0; r < PR; r++) {    // all loads first: dSe is read (pre_dS) and then overwritten (dS at chunk end)
            int e = tid + 256 * r, k = e / TVS, vv = e % TVS;
            p[r] = __bfloat162float(dSe[cb * KD * VD + (long)k * VD + v0 + vv]);
        }
        if (n == ncarry - 1 && dS_out)
#pragma unroll
            for (int r = 0; r < PR; r++) {
                int e = tid + 256 * r, k = e / TVS, vv = e % TVS;
                sS[e] += dS_out[bh * KD * VD + (long)k * VD + v0 + vv];
            }
#pragma unroll
        for (int r = 0; r < PR; r++) {
            int e = tid + 256 * r, k = e / TVS, vv = e % TVS;
            bf16 xb = __float2bfloat16(sS[e]);
            dSe[cb * KD * VD + (long)k * VD + v0 + vv] = xb;
            sSb[k * TC_LDV + vv] = xb;
        }
        tc_load_chunk<KD>(sWK, Kt + cb * CH * KD, false);
        for (int e = tid; e < CH * TVS; e += 256) { int c = e / TVS, vv = e % TVS; sV[e] = __bfloat162float(pre_dvn[cb * CH * VD + (long)c * VD + v0 + vv]); }
        __syncthreads();
        for (int tile = warp; tile < (CH / 16) * (TVS / 16); tile += 8) {   // dvn = pre_dvn + Kt dS
            int m = tile / (TVS / 16), nn = tile % (TVS / 16);
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::load_matrix_sync(acc, sV + 16 * m * TVS + 16 * nn, TVS, wmma::mem_row_major);
#pragma unroll
            for (int kk = 0; kk < KD / 16; kk++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
                wmma::load_matrix_sync(a, sWK + 16 * m * LDK + 16 * kk, LDK);
                wmma::load_matrix_sync(b, sSb + 16 * kk * TC_LDV + 16 * nn, TC_LDV);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sV + 16 * m * TVS + 16 * nn, acc, TVS, wmma::mem_row_major);
        }
        __syncthreads();
        for (int e = tid; e < CH * TVS; e += 256) {
            int c = e / TVS, vv = e % TVS;
            float x = sV[e];
            dvn[cb * CH * VD + (long)c * VD + v0 + vv] = __float2bfloat16(x);
            sVb[c * TC_LDV + vv] = __float2bfloat16(-x);
        }
#pragma unroll
        for (int r = 0; r < PR; r++) { int e = tid + 256 * r; sS[e] = EXPF(gl[cb * KD + (e / TVS)]) * sS[e] + p[r]; }
        tc_load_chunk<KD>(sWK, Wm + cb * CH * KD, false);
        __syncthreads();
        for (int tile = warp; tile < (KD / 16) * (TVS / 16); tile += 8) {   // dS += Wm^T (-dvn)
            int mt = tile / (TVS / 16), nt = tile % (TVS / 16);
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::load_matrix_sync(acc, sS + 16 * mt * TVS + 16 * nt, TVS, wmma::mem_row_major);
#pragma unroll
            for (int kk = 0; kk < CH / 16; kk++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::col_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
                wmma::load_matrix_sync(a, sWK + 16 * kk * LDK + 16 * mt, LDK);
                wmma::load_matrix_sync(b, sVb + 16 * kk * TC_LDV + 16 * nt, TC_LDV);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sS + 16 * mt * TVS + 16 * nt, acc, TVS, wmma::mem_row_major);
        }
        __syncthreads();
    }
    for (int e = tid; e < KD * TVS; e += 256) { int k = e / TVS, vv = e % TVS; dS_in[bh * KD * VD + (long)k * VD + v0 + vv] = sS[e]; }
}
#endif
#ifndef SCAN32
// intra-chunk backward on tensor cores (bf16 build). Same sums as the fp32 kernel below, with the decays of the off-diagonal
// 16x16 blocks factored through a midpoint (never > 1): row side (sums over j < i) through r(I) = 16 I - 1 like the forward,
// column side (sums over i > j) through s(J) = 16 J + 15. Diagonal blocks exact (fp32).
#define B1_LD 72
#define B1_SMEM_BYTES (2 * CH * B1_LD * 2 + 2 * 4 * 16 * 17 * 4 + 5 * CH * (DT + 1) * 4 + 4 * CH * 20 * 4 + KD_MAX_B1 * 4)
#define KD_MAX_B1 128
template <int KD>
__global__ void __launch_bounds__(512) k_intra_bwd(const floatS* __restrict__ cq, const floatS* __restrict__ ck, const floatS* __restrict__ cb,
                                                   const float* __restrict__ gc, const floatS* __restrict__ Qg, const floatS* __restrict__ Kt,
                                                   const floatS* __restrict__ y, const float* __restrict__ dL, const float* __restrict__ dM,
                                                   const floatS* __restrict__ dQg, const floatS* __restrict__ dKt, const floatS* __restrict__ dy,
                                                   const floatS* __restrict__ S0, const floatS* __restrict__ dSe, const float* __restrict__ gl,
                                                   floatS* __restrict__ dq, floatS* __restrict__ dk, floatS* __restrict__ db, floatS* __restrict__ dg) {
    extern __shared__ __align__(32) unsigned char smraw_b1[];
    unsigned char* smraw = smraw_b1;
    bf16* sdLb = (bf16*)smraw;                          // [CH][72] masked dL (bf16)
    bf16* sdMb = sdLb + CH * B1_LD;                     // [CH][72] masked dM (bf16)
    float* sdLd = (float*)(sdMb + CH * B1_LD);          // [4][16][17] diagonal blocks of dL (fp32)
    float* sdMd = sdLd + 4 * 16 * 17;
    float* sq = sdMd + 4 * 16 * 17; float* sk = sq + CH * (DT + 1); float* sbk = sk + CH * (DT + 1); float* sgc = sbk + CH * (DT + 1);
    float* sdgc = sgc + CH * (DT + 1);
    float* sOut = sdgc + CH * (DT + 1);                 // [4][CH][20]: dqM_off, dbkL_off (rows 16..), dkL_off, dkM_off (rows ..47)
    bf16* sKr = (bf16*)sOut;                            // operands alias sOut (used before the products are stored)
    bf16* sQs = sKr + 96 * LDO; bf16* sBs = sQs + 96 * LDO;
    float* sdgl = sOut + 4 * CH * 20;
    const int VD = KD;
    const long cid = blockIdx.x, base = cid * CH * KD;
    const int tid = threadIdx.x, lane = tid & 31, wl = tid >> 5;
    for (int d = wl; d < KD; d += 16) {     // dgl = e^{gl} rowsum(S0 dSe) + colsum(dKt Kt)
        const floatS* s0 = S0 + cid * KD * VD + (long)d * VD;
        const floatS* ds = dSe + cid * KD * VD + (long)d * VD;
        float a = 0.f;
        for (int v = lane * 2; v < VD; v += 64) { float x[2], z[2]; ldv<2>(s0 + v, x); ldv<2>(ds + v, z); a += x[0] * z[0] + x[1] * z[1]; }
        a = warp_sum(a);
        if (lane == 0) sdgl[d] = EXPF(gl[cid * KD + d]) * a;
    }
    __syncthreads();
    if (tid < KD) {
        float a = 0.f;
        for (int c = 0; c < CH; c++) a += toF(dKt[base + c * KD + tid]) * toF(Kt[base + c * KD + tid]);
        sdgl[tid] += a;
    }
    const float* dLc = dL + cid * CH * CH; const float* dMc = dM + cid * CH * CH;
    for (int e = tid; e < CH * CH; e += 512) {
        int a = e / CH, c = e % CH;
        float l = c < a ? dLc[e] : 0.f, m = c <= a ? dMc[e] : 0.f;
        sdLb[a * B1_LD + c] = __float2bfloat16(l);
        sdMb[a * B1_LD + c] = __float2bfloat16(m);
        if ((a >> 4) == (c >> 4)) { int o = (a >> 4) * 272 + (a & 15) * 17 + (c & 15); sdLd[o] = l; sdMd[o] = m; }
    }
    __syncthreads();
    const int warp = wl;
    // jobs (k-steps): 0-3: col J=0 dkL, col J=0 dkM, row I=3 dqM, row I=3 dbkL (3) ; 4-7: J=1 / I=2 (2) ; 8-11: J=2 / I=1 (1)
    for (int d0 = 0; d0 < KD; d0 += DT) {
        for (int e = tid; e < CH * DT; e += 512) {
            int c = e / DT, x = e % DT;
            long g_ = base + c * KD + d0 + x;
            float k = toF(ck[g_]);
            sq[c * (DT + 1) + x] = toF(cq[g_]); sk[c * (DT + 1) + x] = k; sbk[c * (DT + 1) + x] = toF(cb[g_]) * k;
            sgc[c * (DT + 1) + x] = gc[g_];
        }
        __syncthreads();
        for (int e = tid; e < 96 * DT; e += 512) {
            int row = e / DT, x = e % DT;
            int Ir = row < 16 ? 1 : (row < 48 ? 2 : 3), j = row - 8 * Ir * (Ir - 1);
            stX(sKr + row * LDO + x, sk[j * (DT + 1) + x] * EXPF(sgc[(16 * Ir - 1) * (DT + 1) + x] - sgc[j * (DT + 1) + x]));
            int Jc = row < 48 ? 0 : (row < 80 ? 1 : 2), i = 16 * (Jc + 1) + row - (Jc == 0 ? 0 : (Jc == 1 ? 48 : 80));
            float ef = EXPF(sgc[i * (DT + 1) + x] - sgc[(16 * Jc + 15) * (DT + 1) + x]);
            stX(sQs + row * LDO + x, sq[i * (DT + 1) + x] * ef);
            stX(sBs + row * LDO + x, sbk[i * (DT + 1) + x] * ef);
        }
        __syncthreads();
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        const bool has_job = warp < 12;
        if (has_job) {
            int jb = warp, grp = jb >> 2, kind = jb & 3;   // grp 0: J0/I3, 1: J1/I2, 2: J2/I1
            wmma::fill_fragment(acc, 0.f);
            if (kind < 2) {          // column side: out[j in J][d] = sum_{i >= 16(J+1)} dX[i][j] Ys_J[i][d]
                int J = grp, o = J == 0 ? 0 : (J == 1 ? 48 : 80);
                const bf16* X = kind == 0 ? sdLb : sdMb;
                const bf16* Y = kind == 0 ? sBs : sQs;
                for (int t = 0; t < 3 - J; t++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::col_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
                    wmma::load_matrix_sync(a, X + (16 * (J + 1) + 16 * t) * B1_LD + 16 * J, B1_LD);
                    wmma::load_matrix_sync(b, Y + (o + 16 * t) * LDO, LDO);
                    wmma::mma_sync(acc, a, b, acc);
                }
            } else {                 // row side: out[i in I][d] = sum_{j < 16 I} dX[i][j] Kr_I[j][d]
                int I = 3 - grp;
                const bf16* X = kind == 2 ? sdMb : sdLb;
                for (int t = 0; t < I; t++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
                    wmma::load_matrix_sync(a, X + 16 * I * B1_LD + 16 * t, B1_LD);
                    wmma::load_matrix_sync(b, sKr + (8 * I * (I - 1) + 16 * t) * LDO, LDO);
                    wmma::mma_sync(acc, a, b, acc);
                }
            }
        }
        __syncthreads();             // operands (aliasing sOut) are no longer read
        if (has_job) {
            int jb = warp, grp = jb >> 2, kind = jb & 3;
            int out = kind == 0 ? 2 : (kind == 1 ? 3 : (kind == 2 ? 0 : 1));   // 0 dqM 1 dbkL 2 dkL 3 dkM
            int row0 = kind < 2 ? 16 * grp : 16 * (3 - grp);
            wmma::store_matrix_sync(sOut + out * CH * 20 + row0 * 20, acc, 20, wmma::mem_row_major);
        }
        __syncthreads();
        for (int e = tid; e < CH * DT; e += 512) {
            int c = e / DT, dd = e % DT, B_ = c >> 4, cc = c & 15;
            float gci = sgc[c * (DT + 1) + dd];
            float dqM = 0.f, dbkL = 0.f, dkL = 0.f, dkM = 0.f;
            if (B_ > 0) {
                float f = EXPF(gci - sgc[(16 * B_ - 1) * (DT + 1) + dd]);
                dqM = sOut[0 * CH * 20 + c * 20 + dd] * f; dbkL = sOut[1 * CH * 20 + c * 20 + dd] * f;
            }
            if (B_ < 3) {
                float f = EXPF(sgc[(16 * B_ + 15) * (DT + 1) + dd] - gci);
                dkL = sOut[2 * CH * 20 + c * 20 + dd] * f; dkM = sOut[3 * CH * 20 + c * 20 + dd] * f;
            }
            const float* Ld = sdLd + B_ * 272; const float* Md = sdMd + B_ * 272;
            for (int jj = 0; jj <= cc; jj++) {          // diagonal block, row side
                int j = 16 * B_ + jj;
                float kj = sk[j * (DT + 1) + dd] * EXPF(gci - sgc[j * (DT + 1) + dd]);
                dqM += Md[cc * 17 + jj] * kj; dbkL += Ld[cc * 17 + jj] * kj;
            }
            for (int ii = cc; ii < 16; ii++) {          // diagonal block, column side
                int i = 16 * B_ + ii;
                float e2 = EXPF(sgc[i * (DT + 1) + dd] - gci);
                dkL += Ld[ii * 17 + cc] * sbk[i * (DT + 1) + dd] * e2;
                dkM += Md[ii * 17 + cc] * sq[i * (DT + 1) + dd] * e2;
            }
            long g_ = base + c * KD + d0 + dd;
            float q = sq[c * (DT + 1) + dd], k = sk[c * (DT + 1) + dd], bk = sbk[c * (DT + 1) + dd], b = toF(cb[g_]), eg = EXPF(gci);
            float dQ = toF(dQg[g_]), dK = toF(dKt[g_]), dY = toF(dy[g_]);
            stX(dq + g_, dqM + dQ * eg);
            stX(dk + g_, dkL + dkM + dK * EXPF(gl[cid * KD + d0 + dd] - gci) + dY * b * eg + dbkL * b);
            stX(db + g_, dY * k * eg + dbkL * k);
            float dgcv = bk * dbkL - k * dkL + q * dqM - k * dkM + dQ * toF(Qg[g_]) - dK * toF(Kt[g_]) + dY * toF(y[g_]);
            if (c == CH - 1) dgcv += sdgl[d0 + dd];
            sdgc[c * (DT + 1) + dd] = dgcv;
        }
        __syncthreads();
        if (tid < DT) {
            float s = 0.f;
            for (int c = CH - 1; c >= 0; c--) { s += sdgc[c * (DT + 1) + tid]; stX(dg + base + c * KD + d0 + tid, s); }
        }
        __syncthreads();
    }
}
#else
// intra-chunk backward. grid = chunks, block 256, dyn smem (2*CH*65 + 5*CH*33 + KD) floats
template <int KD>
__global__ void __launch_bounds__(256) k_intra_bwd(const floatS* __restrict__ cq, const floatS* __restrict__ ck, const floatS* __restrict__ cb,
                                                   const float* __restrict__ gc, const floatS* __restrict__ Qg, const floatS* __restrict__ Kt,
                                                   const floatS* __restrict__ y, const float* __restrict__ dL, const float* __restrict__ dM,
                                                   const floatS* __restrict__ dQg, const floatS* __restrict__ dKt, const floatS* __restrict__ dy,
                                                   const floatS* __restrict__ S0, const floatS* __restrict__ dSe, const float* __restrict__ gl,
                                                   floatS* __restrict__ dq, floatS* __restrict__ dk, floatS* __restrict__ db, floatS* __restrict__ dg) {
    extern __shared__ float sm[];
    float* sdL = sm; float* sdM = sdL + CH * 65;
    float* sq = sdM + CH * 65; float* sk = sq + CH * 33; float* sbk = sk + CH * 33; float* sgc = sbk + CH * 33; float* sdgc = sgc + CH * 33;
    float* sdgl = sdgc + CH * 33;
    const int VD = KD;
    const long cid = blockIdx.x, base = cid * CH * KD;
    const int tid = threadIdx.x, lane = tid & 31, wl = tid >> 5;
    // dgl = e^{gl} * rowsum(S0 * dSe) + colsum(dKt * Kt)
    for (int d = wl; d < KD; d += 8) {
        const floatS* s0 = S0 + cid * KD * VD + (long)d * VD;
        const floatS* ds = dSe + cid * KD * VD + (long)d * VD;
        float a = 0.f;
        for (int v = lane; v < VD; v += 32) a += toF(s0[v]) * toF(ds[v]);
        a = warp_sum(a);
        if (lane == 0) sdgl[d] = EXPF(gl[cid * KD + d]) * a;
    }
    __syncthreads();
    if (tid < KD) {
        float a = 0.f;
        for (int c = 0; c < CH; c++) a += toF(dKt[base + c * KD + tid]) * toF(Kt[base + c * KD + tid]);
        sdgl[tid] += a;
    }
    const float* dLc = dL + cid * CH * CH; const float* dMc = dM + cid * CH * CH;
    for (int e = tid; e < CH * CH; e += 256) {
        int a = e / CH, c = e % CH;
        sdL[a * 65 + c] = c < a ? dLc[e] : 0.f;
        sdM[a * 65 + c] = c <= a ? dMc[e] : 0.f;
    }
    __syncthreads();
    const int dd = lane, i0 = wl;
    for (int d0 = 0; d0 < KD; d0 += 32) {
        for (int e = tid; e < CH * 32; e += 256) {
            int c = e >> 5, x = e & 31;
            long g_ = base + c * KD + d0 + x;
            sq[c * 33 + x] = toF(cq[g_]); sk[c * 33 + x] = toF(ck[g_]); sbk[c * 33 + x] = toF(cb[g_]) * toF(ck[g_]); sgc[c * 33 + x] = gc[g_];
        }
        __syncthreads();
        float gld = gl[cid * KD + d0 + dd];
        for (int r = 0; r < 8; r++) {
            int c = i0 + 8 * r;
            float gci = sgc[c * 33 + dd];
            float dbkL = 0.f, dqM = 0.f;
            for (int j = 0; j <= c; j++) {
                float kj = sk[j * 33 + dd] * EXPF(gci - sgc[j * 33 + dd]);
                dqM += sdM[c * 65 + j] * kj;
                dbkL += sdL[c * 65 + j] * kj;
            }
            float dkL = 0.f, dkM = 0.f;
            for (int i = c; i < CH; i++) {
                float e = EXPF(sgc[i * 33 + dd] - gci);
                dkL += sdL[i * 65 + c] * sbk[i * 33 + dd] * e;
                dkM += sdM[i * 65 + c] * sq[i * 33 + dd] * e;
            }
            long g_ = base + c * KD + d0 + dd;
            float q = sq[c * 33 + dd], k = sk[c * 33 + dd], bk = sbk[c * 33 + dd], b = toF(cb[g_]), eg = EXPF(gci);
            float dQ = toF(dQg[g_]), dK = toF(dKt[g_]), dY = toF(dy[g_]);
            stX(dq + g_, dqM + dQ * eg);
            stX(dk + g_, dkL + dkM + dK * EXPF(gld - gci) + dY * b * eg + dbkL * b);
            stX(db + g_, dY * k * eg + dbkL * k);
            float dgcv = bk * dbkL - k * dkL + q * dqM - k * dkM + dQ * toF(Qg[g_]) - dK * toF(Kt[g_]) + dY * toF(y[g_]);
            if (c == CH - 1) dgcv += sdgl[d0 + dd];
            sdgc[c * 33 + dd] = dgcv;
        }
        __syncthreads();
        if (tid < 32) {
            float s = 0.f;
            for (int c = CH - 1; c >= 0; c--) { s += sdgc[c * 33 + tid]; stX(dg + base + c * KD + d0 + tid, s); }
        }
        __syncthreads();
    }
}

#define B1_SMEM_BYTES ((2 * CH * 65 + 5 * CH * 33 + 128) * 4)
#endif

// =============================================================================================== host: helpers
static int nblk(long n, int t) { long b = (n + t - 1) / t; return (int)(b > 65535L * 16 ? 65535L * 16 : b); }
#define LAUNCH1D(kernel, n, ...) kernel<<<nblk((n), 256), 256, 0, c->stream>>>(__VA_ARGS__)

static void zero(Ctx* c, void* p, size_t bytes) { if (p && bytes) CK(cudaMemsetAsync(p, 0, bytes, c->stream)); }

// scratch for one (layer, segment) step of N = B*T1 rows
typedef struct {
    int B, Tr, Tm, T1, N, nr, nm, nch, BM;
    long nchB;   // B*H*nch
} Step;

static Step make_step(Ctx* c, int B, int Tr, int Tm) {
    Step s;
    s.B = B; s.Tr = Tr; s.Tm = Tm; s.T1 = Tr + Tm; s.N = B * s.T1;
    s.nr = (Tr + CH - 1) / CH; s.nm = (Tm + CH - 1) / CH; s.nch = s.nr + s.nm; s.BM = B * Tm;
    s.nchB = (long)B * c->d.H * s.nch;
    return s;
}

// =============================================================================================== host: GDN-2 scan
#ifdef SCAN32
#define ST_SMEM(KD) ((KD * 32 + KD * (CH + 1) + CH * 32) * 4)
#else
#define ST_SMEM(KD) tc_smem<KD>()
#endif
template <int KD> static void set_smem_attrs() {
    static bool done = false;
    if (done) return;
    done = true;
    CK(cudaFuncSetAttribute(k_intra_fwd<KD>, cudaFuncAttributeMaxDynamicSharedMemorySize, K1_SMEM * 4));
    CK(cudaFuncSetAttribute(k_state_fwd<KD>, cudaFuncAttributeMaxDynamicSharedMemorySize, ST_SMEM(KD)));
    CK(cudaFuncSetAttribute(k_state_bwd<KD>, cudaFuncAttributeMaxDynamicSharedMemorySize, ST_SMEM(KD)));
    CK(cudaFuncSetAttribute(k_intra_bwd<KD>, cudaFuncAttributeMaxDynamicSharedMemorySize, B1_SMEM_BYTES));
}
#define SC CUDA_S
// chunk-major scan inputs come from k_prep_fwd (B_CQ, B_CK, B_CB, B_CWV: floatS ; B_CG: fp32)
template <int KD>
static void scan_fwd(Ctx* c, const Step& s, const float* S_in, float* S_out, int ncarry) {
    set_smem_attrs<KD>();
    long CSK = s.nchB * CH * KD, CSC = s.nchB * CH * CH;
    floatS *cq = BF(c, B_CQ, floatS, CSK), *ck = BF(c, B_CK, floatS, CSK), *cb = BF(c, B_CB, floatS, CSK), *cwv = BF(c, B_CWV, floatS, CSK);
    float *cg = BF(c, B_CG, float, CSK), *gc = BF(c, B_GC, float, CSK), *gl = BF(c, B_GL, float, s.nchB * KD), *Am = BF(c, B_AM, float, CSC);
    floatS *y = BF(c, B_Y, floatS, CSK), *Qg = BF(c, B_QG, floatS, CSK), *Kt = BF(c, B_KT, floatS, CSK), *Wm = BF(c, B_WM, floatS, CSK);
    floatS *u = BF(c, B_U, floatS, CSK), *vn = BF(c, B_VN, floatS, CSK), *o = BF(c, B_O, floatS, CSK), *Mm = BF(c, B_MMAT, floatS, CSC);
    floatS* S0 = BF(c, B_S0, floatS, s.nchB * KD * KD);
#ifdef SCAN32
    floatS* AmX = Am;
    floatS* AmW = NULL;
#else
    floatS* AmX = BF(c, B_AMX, floatS, CSC);
    floatS* AmW = AmX;
#endif
    int nb = (int)s.nchB;
    long long sK = CH * KD, sC = CH * CH, sS = (long long)KD * KD;
    k_intra_fwd<KD><<<nb, 256, K1_SMEM * 4, c->stream>>>(cq, ck, cg, cb, gc, y, Qg, Kt, Am, AmW, Mm, gl);
    mm(c, 0, 0, CH, KD, CH, 1.f, AmX, SC, CH, sC, cwv, SC, KD, sK, 0.f, u, SC, KD, sK, nb);     // u = A wv
    mm(c, 0, 0, CH, KD, CH, 1.f, AmX, SC, CH, sC, y, SC, KD, sK, 0.f, Wm, SC, KD, sK, nb);      // Wm = A y
#ifdef SCAN32
    dim3 g((unsigned)(s.B * c->d.H), KD / 32);
#else
    dim3 g((unsigned)(s.B * c->d.H), KD / TVS);
#endif
#ifdef SCAN32
    k_state_fwd<KD><<<g, 256, ST_SMEM(KD), c->stream>>>(u, Wm, Kt, gl, S0, vn, S_in, S_out, ncarry, s.nch);
    mm(c, 0, 0, CH, KD, KD, 1.f, Qg, SC, KD, sK, S0, SC, KD, sS, 0.f, o, SC, KD, sK, nb);       // o = Qg S0
#elif !defined(ARMT_SPLIT_STATE)
    k_state_fwd<KD><<<g, 256, ST_SMEM(KD), c->stream>>>(u, Wm, Kt, gl, S0, NULL, vn, S_in, S_out, ncarry, s.nch);
    mm(c, 0, 0, CH, KD, KD, 1.f, Qg, SC, KD, sK, S0, SC, KD, sS, 0.f, o, SC, KD, sK, nb);       // o = Qg S0
#else
    floatS* S0lo = BF(c, B_S0LO, floatS, s.nchB * KD * KD);
    k_state_fwd<KD><<<g, 256, ST_SMEM(KD), c->stream>>>(u, Wm, Kt, gl, S0, S0lo, vn, S_in, S_out, ncarry, s.nch);
    float* o32 = BF(c, B_O32, float, CSK);                                                      // o = Qg (S0 + S0lo), fp32 accum
    mm(c, 0, 0, CH, KD, KD, 1.f, Qg, SC, KD, sK, S0, SC, KD, sS, 0.f, o32, F32, KD, sK, nb);
    mm(c, 0, 0, CH, KD, KD, 1.f, Qg, SC, KD, sK, S0lo, SC, KD, sS, 1.f, o32, F32, KD, sK, nb);
    mm(c, 0, 0, CH, KD, CH, 1.f, Mm, SC, CH, sC, vn, SC, KD, sK, 1.f, o32, F32, KD, sK, nb);
    LAUNCH1D(k_cast, CSK, o, o32, CSK);
    return;
#endif
    mm(c, 0, 0, CH, KD, CH, 1.f, Mm, SC, CH, sC, vn, SC, KD, sK, 1.f, o, SC, KD, sK, nb);       //   + M vn
}

// do (B_DO, floatS) given; produces dq, dk, db, dg, dwv (fp32, chunk-major) and dS_in
template <int KD>
static void scan_bwd(Ctx* c, const Step& s, const float* dS_out, float* dS_in, int ncarry) {
    long CSK = s.nchB * CH * KD, CSC = s.nchB * CH * CH;
    int nb = (int)s.nchB;
    floatS *cq = BF(c, B_CQ, floatS, CSK), *ck = BF(c, B_CK, floatS, CSK), *cb = BF(c, B_CB, floatS, CSK), *cwv = BF(c, B_CWV, floatS, CSK);
    float *gc = BF(c, B_GC, float, CSK), *gl = BF(c, B_GL, float, s.nchB * KD), *Am = BF(c, B_AM, float, CSC);
    floatS *y = BF(c, B_Y, floatS, CSK), *Qg = BF(c, B_QG, floatS, CSK), *Kt = BF(c, B_KT, floatS, CSK), *Wm = BF(c, B_WM, floatS, CSK);
    floatS *vn = BF(c, B_VN, floatS, CSK), *Mm = BF(c, B_MMAT, floatS, CSC), *S0 = BF(c, B_S0, floatS, s.nchB * KD * KD);
#ifdef SCAN32
    floatS* AmX = Am;
#else
    floatS* AmX = BF(c, B_AMX, floatS, CSC);
#endif
    floatS *dO = BF(c, B_DO, floatS, CSK), *dSe = BF(c, B_DSE, floatS, s.nchB * KD * KD), *dvn = BF(c, B_DVN, floatS, CSK);
    floatS *pre_dvn = BF(c, B_PREDVN, floatS, CSK), *dWm = BF(c, B_DWM, floatS, CSK);
    floatS *dQg = BF(c, B_DQG, floatS, CSK), *dKt = BF(c, B_DKT, floatS, CSK), *dy = BF(c, B_DY, floatS, CSK);
    float *dM = BF(c, B_DMMAT, float, CSC), *dA = BF(c, B_DAM, float, CSC), *T1 = BF(c, B_T1, float, CSC), *dL = BF(c, B_DL, float, CSC);
    floatS *dwv = BF(c, B_DWV, floatS, CSK), *dq = BF(c, B_DQ, floatS, CSK), *dk = BF(c, B_DK, floatS, CSK), *db = BF(c, B_DB, floatS, CSK);
    floatS* dg = BF(c, B_DG, floatS, CSK);
    long long sK = CH * KD, sC = CH * CH, sS = (long long)KD * KD;
    mm(c, 1, 0, CH, KD, CH, 1.f, Mm, SC, CH, sC, dO, SC, KD, sK, 0.f, pre_dvn, SC, KD, sK, nb);  // pre_dvn = M^T do
    mm(c, 1, 0, KD, KD, CH, 1.f, Qg, SC, KD, sK, dO, SC, KD, sK, 0.f, dSe, SC, KD, sS, nb);      // pre_dS = Qg^T do
#ifdef SCAN32
    dim3 g((unsigned)(s.B * c->d.H), KD / 32);
#else
    dim3 g((unsigned)(s.B * c->d.H), KD / TVS);
#endif
    k_state_bwd<KD><<<g, 256, ST_SMEM(KD), c->stream>>>(pre_dvn, dvn, dSe, Kt, Wm, gl, dS_out, ncarry, dS_in, s.nch);
    mm(c, 0, 1, CH, KD, KD, 1.f, dO, SC, KD, sK, S0, SC, KD, sS, 0.f, dQg, SC, KD, sK, nb);     // dQg = do S0^T
    mm(c, 0, 1, CH, KD, KD, 1.f, vn, SC, KD, sK, dSe, SC, KD, sS, 0.f, dKt, SC, KD, sK, nb);    // dKt = vn dSe^T
    mm(c, 0, 1, CH, KD, KD, -1.f, dvn, SC, KD, sK, S0, SC, KD, sS, 0.f, dWm, SC, KD, sK, nb);    // dWm = -dvn S0^T
    mm(c, 0, 1, CH, CH, KD, 1.f, dO, SC, KD, sK, vn, SC, KD, sK, 0.f, dM, F32, CH, sC, nb);      // dM = do vn^T
    mm(c, 1, 0, CH, KD, CH, 1.f, AmX, SC, CH, sC, dvn, SC, KD, sK, 0.f, dwv, SC, KD, sK, nb);   // dwv = A^T du
    mm(c, 1, 0, CH, KD, CH, 1.f, AmX, SC, CH, sC, dWm, SC, KD, sK, 0.f, dy, SC, KD, sK, nb);    // dy = A^T dWm
    mm(c, 0, 1, CH, CH, KD, 1.f, dvn, SC, KD, sK, cwv, SC, KD, sK, 0.f, dA, F32, CH, sC, nb);    // dA = du wv^T
    mm(c, 0, 1, CH, CH, KD, 1.f, dWm, SC, KD, sK, y, SC, KD, sK, 1.f, dA, F32, CH, sC, nb);      //    + dWm y^T
    mm(c, 0, 1, CH, CH, CH, 1.f, dA, F32, CH, sC, Am, F32, CH, sC, 0.f, T1, F32, CH, sC, nb);    // T1 = dA A^T
    mm(c, 1, 0, CH, CH, CH, -1.f, Am, F32, CH, sC, T1, F32, CH, sC, 0.f, dL, F32, CH, sC, nb);   // dL = -A^T T1
#ifdef SCAN32
    const int b1_threads = 256;
#else
    const int b1_threads = 512;
#endif
    k_intra_bwd<KD><<<nb, b1_threads, B1_SMEM_BYTES, c->stream>>>(cq, ck, cb, gc, Qg, Kt, y, dL, dM, dQg, dKt, dy, S0, dSe, gl,
                                                                                 dq, dk, db, dg);
}

// =============================================================================================== host: one layer
// forward of layer li over one segment piece. x: [B, T1, D] input (fp32), xo: output. st_in: state before the piece;
// st_out: carried state (may be NULL to discard; may alias st_in for S, Wm, z). write: memory tokens present.
template <int KD>
static void layer_fwd(Ctx* c, int li, const Step& s, const float* x, float* xo, const LState* st_in, const LState* st_out) {
    Dims& d = c->d;
    int D = d.D, N = s.N, NP = d.NP, KDt = d.KDt, VDt = d.VDt, Hv = d.Hv, F = d.F;
    float eps = d.c.norm_eps;
    bool mem = d.has_mem[li];
    float *xr = BF(c, B_XR, float, (long)N * D), *rstd1 = BF(c, B_RSTD1, float, N);
    floatX* h1 = BF(c, B_H1, floatX, (long)N * D);
    // 1. memory read + norm_1
    if (mem) {
        float *mq = BF(c, B_MQ, float, (long)N * d.dm), *phiq = BF(c, B_PHIQ, float, (long)N * d.dk), *denq = BF(c, B_DENQ, float, (long)N * d.Hm);
        float* numq = BF(c, B_NUMQ, float, (long)N * D);
        lin32(c, mq, d.dm, x, D, PP(c, PI(li, L_MQ)), N, d.dm, D, 0.f);
        int nw = N * d.Hm;
        k_dpfp<<<(nw + 7) / 8, 256, 0, c->stream>>>(mq, d.dm, phiq, denq, NULL, st_in->z, N, s.T1, d.dm, d.Hm);
        int dkh = d.dk / d.Hm, dvh = D / d.Hm;
        for (int h = 0; h < d.Hm; h++)   // num[b] = phi[b] W[b]
            mm(c, 0, 0, s.T1, dvh, dkh, 1.f, phiq + h * dkh, F32, d.dk, (long long)s.T1 * d.dk, st_in->Wm + (long)h * dkh * dvh, F32, dvh,
               (long long)d.dk * D / d.Hm, 0.f, numq + h * dvh, F32, D, (long long)s.T1 * D, s.B);
        k_rmsnorm<<<N, 256, 0, c->stream>>>(xr, x, NULL, numq, denq, d.Hm, h1, rstd1, PP(c, PI(li, L_N1)), D, eps, 1, 1);
    } else {
        CK(cudaMemcpyAsync(xr, x, (size_t)N * D * 4, cudaMemcpyDeviceToDevice, c->stream));
        k_rmsnorm<<<N, 256, 0, c->stream>>>(NULL, x, NULL, NULL, NULL, 1, h1, rstd1, PP(c, PI(li, L_N1)), D, eps, 1, 1);
    }
    // 2. projections
    floatX *P = BF(c, B_P, floatX, (long)N * NP), *fb = BF(c, B_FB, floatX, (long)N * KDt), *gg = BF(c, B_GG, floatX, (long)N * VDt);
    lin(c, P, NP, h1, D, c->w_in[li], N, NP, D, 0.f);
    int of1 = 3 * KDt + 2 * VDt, og1 = of1 + Hv;
    lin(c, fb, KDt, P + of1, NP, c->w_f2[li], N, KDt, Hv, 0.f);
    lin(c, gg, VDt, P + og1, NP, c->w_g2[li], N, VDt, Hv, 0.f);
    // 3. conv / gates -> chunk-major scan inputs ; carried conv state
    long CSK = s.nchB * CH * KD;
    floatS *cq = BF(c, B_CQ, floatS, CSK), *ck = BF(c, B_CK, floatS, CSK), *cb = BF(c, B_CB, floatS, CSK), *cwv = BF(c, B_CWV, floatS, CSK);
    float* cg = BF(c, B_CG, float, CSK);
    const float* wconv = PP(c, PI(li, L_QC));   // q, k, v conv weights are contiguous only in the grad layout: pass separately
    if (d.W != 4) armt_fail("conv_size must be 4");
    long nwarps = s.nchB * CH / 8;   // runs of 8 positions (CH is a multiple of 8)
    k_prep_fwd<KD><<<(unsigned)((nwarps + 7) / 8), 256, 0, c->stream>>>(P, NP, fb, wconv, PP(c, PI(li, L_KC)), PP(c, PI(li, L_VC)), st_in->cs,
                                                                       PP(c, PI(li, L_ALOG)), PP(c, PI(li, L_DTB)), cq, ck, cg, cb, cwv,
                                                                       s.B, s.Tr, s.Tm, d.H, s.nr, s.nch, d.W, 1.f / sqrtf((float)KD));
    int Pc = (s.Tm == 0 || d.c.mem_in_stream) ? s.T1 : s.Tr;
    int ncarry = (s.Tm == 0 || d.c.mem_in_stream) ? s.nch : s.nr;
    if (st_out && st_out->cs) {
        float* tmp = BF(c, B_CSTMP, float, (long)s.B * d.Cc * d.W);
        LAUNCH1D(k_conv_state, (long)s.B * d.Cc * d.W, tmp, st_in->cs, P, NP, s.B, s.T1, Pc, d.Cc, d.W);
        CK(cudaMemcpyAsync(st_out->cs, tmp, (size_t)s.B * d.Cc * d.W * 4, cudaMemcpyDeviceToDevice, c->stream));
    }
    // 4. scan
    scan_fwd<KD>(c, s, st_in->S, st_out ? st_out->S : NULL, ncarry);
    // 5. gated norm, o_proj, residual + norm_2
    floatX *on = BF(c, B_ON, floatX, (long)N * VDt), *ao = BF(c, B_AO, floatX, (long)N * D);
    float* rstdo = BF(c, B_RSTDO, float, (long)N * d.H);
    long nw2 = (long)N * d.H;
    k_onorm_fwd<KD><<<(unsigned)((nw2 + 7) / 8), 256, 0, c->stream>>>(BF(c, B_O, floatS, CSK), gg, PP(c, PI(li, L_G2B)), PP(c, PI(li, L_ON)), on,
                                                                     rstdo, s.B, s.Tr, s.Tm, d.H, s.nr, s.nch, eps, d.c.gate_swish);
    lin(c, ao, D, on, VDt, c->w_o[li], N, D, VDt, 0.f);
    float *xmid = BF(c, B_XMID, float, (long)N * D), *rstd2 = BF(c, B_RSTD2, float, N);
    floatX* h2 = BF(c, B_H2, floatX, (long)N * D);
    k_rmsnorm<<<N, 256, 0, c->stream>>>(xmid, xr, ao, NULL, NULL, 1, h2, rstd2, PP(c, PI(li, L_N2)), D, eps, 1, 1);
    // 6. MLP
    floatX *a = BF(c, B_A, floatX, (long)N * 2 * F), *m = BF(c, B_MM, floatX, (long)N * F), *mo = BF(c, B_MO, floatX, (long)N * D);
    lin(c, a, 2 * F, h2, D, c->w_12[li], N, 2 * F, D, 0.f);
    LAUNCH1D(k_swiglu, (long)N * F, m, a, (long)N, F);
    lin(c, mo, D, m, F, c->w_3[li], N, D, F, 0.f);
    LAUNCH1D(k_resid, (long)N * D, xo, xmid, mo, (long)N * D);
    // 7. memory write from the memory tokens' outputs
    if (mem && s.Tm > 0 && st_out && st_out->Wm) {
        int BM = s.BM, dkh = d.dk / d.Hm, dvh = D / d.Hm;
        float *xm = BF(c, B_XM, float, (long)BM * D), *mk = BF(c, B_MK, float, (long)BM * d.dm), *phik = BF(c, B_PHIK, float, (long)BM * d.dk);
        float *denk = BF(c, B_DENK, float, (long)BM * d.Hm), *nrmk = BF(c, B_NRMK, float, (long)BM * d.Hm), *nmv = BF(c, B_NMV, float, (long)BM * D);
        float *mbp = BF(c, B_MBP, float, (long)BM * d.nmb), *prevn = BF(c, B_PREVN, float, (long)BM * D), *E = BF(c, B_E, float, (long)BM * D);
        float* coef = BF(c, B_COEF, float, (long)BM * d.Hm);
        k_gather_mem<<<BM, 256, 0, c->stream>>>(xm, xo, s.Tr, s.Tm, D);
        lin32(c, mk, d.dm, xm, D, PP(c, PI(li, L_MK)), BM, d.dm, D, 0.f);
        lin32(c, nmv, D, xm, D, PP(c, PI(li, L_MV)), BM, D, D, 0.f);
        lin32(c, mbp, d.nmb, xm, D, PP(c, PI(li, L_MB)), BM, d.nmb, D, 0.f);
        k_dpfp<<<(BM * d.Hm + 7) / 8, 256, 0, c->stream>>>(mk, d.dm, phik, denk, nrmk, st_in->z, BM, s.Tm, d.dm, d.Hm);
        for (int h = 0; h < d.Hm; h++)
            mm(c, 0, 0, s.Tm, dvh, dkh, 1.f, phik + h * dkh, F32, d.dk, (long long)s.Tm * d.dk, st_in->Wm + (long)h * dkh * dvh, F32, dvh,
               (long long)d.dk * D / d.Hm, 0.f, prevn + h * dvh, F32, D, (long long)s.Tm * D, s.B);
        k_write_E<<<BM, 256, 0, c->stream>>>(E, nmv, prevn, denk, mbp, PP(c, PI(li, L_MBB)), D, d.Hm, d.nmb);
        if (st_out->Wm != st_in->Wm) CK(cudaMemcpyAsync(st_out->Wm, st_in->Wm, (size_t)s.B * d.dk * D / d.Hm * 4, cudaMemcpyDeviceToDevice, c->stream));
        for (int h = 0; h < d.Hm; h++)   // W += phik^T E
            mm(c, 1, 0, dkh, dvh, s.Tm, 1.f, phik + h * dkh, F32, d.dk, (long long)s.Tm * d.dk, E + h * dvh, F32, D, (long long)s.Tm * D, 1.f,
               st_out->Wm + (long)h * dkh * dvh, F32, dvh, (long long)d.dk * D / d.Hm, s.B);
        k_write_z<<<s.B * d.Hm, 256, 0, c->stream>>>(st_in->z, st_out->z, phik, denk, nrmk, coef, s.Tm, dkh, d.Hm);
    } else if (mem && st_out && st_out->Wm && st_out->Wm != st_in->Wm) {
        CK(cudaMemcpyAsync(st_out->Wm, st_in->Wm, (size_t)s.B * d.dk * D / d.Hm * 4, cudaMemcpyDeviceToDevice, c->stream));
        CK(cudaMemcpyAsync(st_out->z, st_in->z, (size_t)s.B * d.dk * 4, cudaMemcpyDeviceToDevice, c->stream));
    }
}

// backward of layer li over one segment (after layer_fwd recomputed it). dx: in = grad of the output (incl. memory rows),
// out = grad of the input. dst: in = grad of the carried state, out = grad of the input state (in place).
template <int KD>
static void layer_bwd(Ctx* c, int li, const Step& s, const float* x, const float* xo, const LState* st_in, float* dx, const LState* dst) {
    Dims& d = c->d;
    int D = d.D, N = s.N, NP = d.NP, KDt = d.KDt, VDt = d.VDt, Hv = d.Hv, F = d.F;
    float eps = d.c.norm_eps;
    bool mem = d.has_mem[li];
    int dkh = d.dk / d.Hm, dvh = D / d.Hm;
    int rb_blocks = 2 * c->sms;
    // ---- memory write backward
    if (mem && s.Tm > 0) {
        int BM = s.BM;
        float *xm = BF(c, B_XM, float, (long)BM * D), *phik = BF(c, B_PHIK, float, (long)BM * d.dk), *denk = BF(c, B_DENK, float, (long)BM * d.Hm);
        float *nmv = BF(c, B_NMV, float, (long)BM * D), *mbp = BF(c, B_MBP, float, (long)BM * d.nmb), *prevn = BF(c, B_PREVN, float, (long)BM * D);
        float *E = BF(c, B_E, float, (long)BM * D), *coef = BF(c, B_COEF, float, (long)BM * d.Hm), *mk = BF(c, B_MK, float, (long)BM * d.dm);
        float *dE = BF(c, B_DE, float, (long)BM * D), *dphik = BF(c, B_DPHIK, float, (long)BM * d.dk), *dprevn = BF(c, B_DPREVN, float, (long)BM * D);
        float *dnmv = BF(c, B_DNMV, float, (long)BM * D), *dmbp = BF(c, B_DMBP, float, (long)BM * d.nmb), *ddenk = BF(c, B_DDENK, float, (long)BM * d.Hm);
        float *dmk = BF(c, B_DMK, float, (long)BM * d.dm), *dxm = BF(c, B_DXM, float, (long)BM * D);
        for (int h = 0; h < d.Hm; h++) {
            long long sW = (long long)d.dk * D / d.Hm;
            // dE = phik dW_new ; dphik = E dW_new^T
            mm(c, 0, 0, s.Tm, dvh, dkh, 1.f, phik + h * dkh, F32, d.dk, (long long)s.Tm * d.dk, dst->Wm + (long)h * dkh * dvh, F32, dvh, sW, 0.f,
               dE + h * dvh, F32, D, (long long)s.Tm * D, s.B);
            mm(c, 0, 1, s.Tm, dkh, dvh, 1.f, E + h * dvh, F32, D, (long long)s.Tm * D, dst->Wm + (long)h * dkh * dvh, F32, dvh, sW, 0.f,
               dphik + h * dkh, F32, d.dk, (long long)s.Tm * d.dk, s.B);
        }
        k_write_bwd1<<<BM, 256, 0, c->stream>>>(dE, nmv, prevn, denk, mbp, PP(c, PI(li, L_MBB)), dnmv, dprevn, ddenk, dmbp, D, d.Hm, d.nmb);
        for (int h = 0; h < d.Hm; h++) {
            long long sW = (long long)d.dk * D / d.Hm;
            // dW_old += phik^T dprevn ; dphik += dprevn W_old^T
            mm(c, 1, 0, dkh, dvh, s.Tm, 1.f, phik + h * dkh, F32, d.dk, (long long)s.Tm * d.dk, dprevn + h * dvh, F32, D, (long long)s.Tm * D, 1.f,
               dst->Wm + (long)h * dkh * dvh, F32, dvh, sW, s.B);
            mm(c, 0, 1, s.Tm, dkh, dvh, 1.f, dprevn + h * dvh, F32, D, (long long)s.Tm * D, st_in->Wm + (long)h * dkh * dvh, F32, dvh, sW, 1.f,
               dphik + h * dkh, F32, d.dk, (long long)s.Tm * d.dk, s.B);
        }
        // dphik += ddenk z_old + coef dz_new ; dz_old += sum ddenk phik  (dz_new == dz_old buffer: coef term first)
        k_dphi_add<<<(BM * d.Hm + 7) / 8, 256, 0, c->stream>>>(dphik, ddenk, st_in->z, coef, dst->z, BM, s.Tm, dkh, d.Hm);
        k_dz_acc<<<s.B * d.Hm, 256, 0, c->stream>>>(dst->z, ddenk, phik, s.Tm, dkh, d.Hm);
        k_dpfp_bwd<<<(BM * d.Hm + 7) / 8, 256, 8 * dkh * 4, c->stream>>>(mk, d.dm, dphik, dmk, d.dm, BM, d.dm, d.Hm);
        lin32_dx(c, dxm, D, dmk, d.dm, PP(c, PI(li, L_MK)), BM, d.dm, D, 0.f);
        lin32_dx(c, dxm, D, dnmv, D, PP(c, PI(li, L_MV)), BM, D, D, 1.f);
        lin32_dx(c, dxm, D, dmbp, d.nmb, PP(c, PI(li, L_MB)), BM, d.nmb, D, 1.f);
        lin32_dw(c, GP(c, PI(li, L_MK)), dmk, d.dm, xm, D, BM, d.dm, D);
        lin32_dw(c, GP(c, PI(li, L_MV)), dnmv, D, xm, D, BM, D, D);
        lin32_dw(c, GP(c, PI(li, L_MB)), dmbp, d.nmb, xm, D, BM, d.nmb, D);
        if (GP(c, PI(li, L_MBB))) k_colsum<<<(d.nmb + 255) / 256, 256, 0, c->stream>>>(GP(c, PI(li, L_MBB)), dmbp, BM, d.nmb);
        k_scatter_add_mem<<<BM, 256, 0, c->stream>>>(dx, dxm, s.Tr, s.Tm, D);
    }
    // ---- MLP backward
    floatX* dxX;
#ifdef ARMT_FP32
    dxX = dx;
#else
    dxX = BF(c, B_DXX, floatX, (long)N * D);
    LAUNCH1D(k_cast, (long)N * D, dxX, dx, (long)N * D);
#endif
    floatX *a = BF(c, B_A, floatX, (long)N * 2 * F), *m = BF(c, B_MM, floatX, (long)N * F), *h2 = BF(c, B_H2, floatX, (long)N * D);
    floatX *dmm = BF(c, B_DMM, floatX, (long)N * F), *da = BF(c, B_DA, floatX, (long)N * 2 * F), *dh2 = BF(c, B_DH2, floatX, (long)N * D);
    lin_dx(c, dmm, F, dxX, D, c->w_3[li], N, D, F, 0.f);
    lin_dw(c, GP(c, PI(li, L_W3)), dxX, D, m, F, N, D, F);
    LAUNCH1D(k_swiglu_bwd, (long)N * F, da, dmm, a, (long)N, F);
    lin_dx(c, dh2, D, da, 2 * F, c->w_12[li], N, 2 * F, D, 0.f);
    lin_dw(c, GPG(c, PI(li, L_W1), PI(li, L_W2)), da, 2 * F, h2, D, N, 2 * F, D);
    float *xmid = BF(c, B_XMID, float, (long)N * D), *rstd2 = BF(c, B_RSTD2, float, N);
    k_rmsnorm_bwd<<<rb_blocks, 256, D * 4, c->stream>>>(dx, dh2, xmid, rstd2, PP(c, PI(li, L_N2)), GP(c, PI(li, L_N2)), N, D, 1, 1);
    // ---- attention backward
#ifndef ARMT_FP32
    LAUNCH1D(k_cast, (long)N * D, dxX, dx, (long)N * D);
#endif
    floatX *on = BF(c, B_ON, floatX, (long)N * VDt), *don = BF(c, B_DON, floatX, (long)N * VDt), *gg = BF(c, B_GG, floatX, (long)N * VDt);
    floatX *dgg = BF(c, B_DGG, floatX, (long)N * VDt), *P = BF(c, B_P, floatX, (long)N * NP), *dP = BF(c, B_DP, floatX, (long)N * NP);
    floatX *fb = BF(c, B_FB, floatX, (long)N * KDt), *df = BF(c, B_DF, floatX, (long)N * KDt), *h1 = BF(c, B_H1, floatX, (long)N * D);
    floatX* dh1 = BF(c, B_DH1, floatX, (long)N * D);
    lin_dx(c, don, VDt, dxX, D, c->w_o[li], N, D, VDt, 0.f);
    lin_dw(c, GP(c, PI(li, L_O)), dxX, D, on, VDt, N, D, VDt);
    long CSK = s.nchB * CH * KD;
    floatS* dO = BF(c, B_DO, floatS, CSK);
    {
        int per = 512;
        dim3 g(d.H, (unsigned)(((long)s.B * s.nch * CH + per - 1) / per));
        float* gwon = GP(c, PI(li, L_ON)); float* gg2b = GP(c, PI(li, L_G2B));
        float* junk = BF(c, B_ZERO, float, 2 * (long)d.VDt + 64);
        k_onorm_bwd<KD><<<g, 256, 0, c->stream>>>(BF(c, B_O, floatS, CSK), gg, PP(c, PI(li, L_G2B)), PP(c, PI(li, L_ON)), don, dO, dgg,
                                                 gwon ? gwon : junk, gg2b ? gg2b : junk, s.B, s.Tr, s.Tm, d.H, s.nr, s.nch, eps, d.c.gate_swish, per);
    }
    int of1 = 3 * KDt + 2 * VDt, og1 = of1 + Hv;
    lin_dx(c, dP + og1, NP, dgg, VDt, c->w_g2[li], N, VDt, Hv, 0.f);
    lin_dw(c, GP(c, PI(li, L_G2)), dgg, VDt, P + og1, NP, N, VDt, Hv);
    int Pc = (s.Tm == 0 || d.c.mem_in_stream) ? s.T1 : s.Tr;
    int ncarry = (s.Tm == 0 || d.c.mem_in_stream) ? s.nch : s.nr;
    scan_bwd<KD>(c, s, dst->S, dst->S, ncarry);
    floatS* dc = BF(c, B_DC, floatS, (long)N * d.Cc);
    {
        int rpb = 64;
        dim3 g(d.H, (N + rpb - 1) / rpb);
        float* gA = GP(c, PI(li, L_ALOG)); float* gdt = GP(c, PI(li, L_DTB));
        float* junk = BF(c, B_ZERO, float, 2 * (long)d.VDt + 64);
        k_prep_bwd1<KD><<<g, 256, 0, c->stream>>>(P, NP, fb, PP(c, PI(li, L_QC)), PP(c, PI(li, L_KC)), PP(c, PI(li, L_VC)), st_in->cs,
                                                 PP(c, PI(li, L_ALOG)), PP(c, PI(li, L_DTB)),
                                                 BF(c, B_DQ, floatS, CSK), BF(c, B_DK, floatS, CSK), BF(c, B_DB, floatS, CSK), BF(c, B_DG, floatS, CSK),
                                                 BF(c, B_DWV, floatS, CSK), dc, dP, df, gA ? gA : junk, gdt ? gdt : junk,
                                                 s.B, s.Tr, s.Tm, d.H, s.nr, s.nch, d.W, 1.f / sqrtf((float)KD), rpb);
    }
    lin_dx(c, dP + of1, NP, df, KDt, c->w_f2[li], N, KDt, Hv, 0.f);
    lin_dw(c, GP(c, PI(li, L_F2)), df, KDt, P + of1, NP, N, KDt, Hv);
    {
        // conv weights: q, k, v live in separate tensors; their grads are contiguous (L_QC, L_KC, L_VC adjacent in the layout).
        // the kernel wants one [Cc][W] weight array: stage a contiguous copy
        float* wcat = BF(c, B_WCAT, float, (long)d.Cc * d.W);
        CK(cudaMemcpyAsync(wcat, PP(c, PI(li, L_QC)), (size_t)KDt * d.W * 4, cudaMemcpyDeviceToDevice, c->stream));
        CK(cudaMemcpyAsync(wcat + (long)KDt * d.W, PP(c, PI(li, L_KC)), (size_t)KDt * d.W * 4, cudaMemcpyDeviceToDevice, c->stream));
        CK(cudaMemcpyAsync(wcat + 2L * KDt * d.W, PP(c, PI(li, L_VC)), (size_t)VDt * d.W * 4, cudaMemcpyDeviceToDevice, c->stream));
        float* dcs_tmp = BF(c, B_DCSTMP, float, (long)s.B * d.Cc * d.W);
        float* gconv = GPG(c, PI(li, L_QC), PI(li, L_VC));
        float* junk = gconv ? gconv : BF(c, B_CSTMP, float, (long)s.B * d.Cc * d.W + (long)d.Cc * d.W);
        if (d.W != 4) armt_fail("conv_size must be 4");
        dim3 g((d.Cc + 127) / 128, 1, s.B);
        k_prep_bwd2<4><<<g, 128, 0, c->stream>>>(P, NP, dc, wcat, st_in->cs, dst->cs, dcs_tmp, dP, junk, s.B, s.T1, Pc, d.Cc);
        if (dst->cs) CK(cudaMemcpyAsync(dst->cs, dcs_tmp, (size_t)s.B * d.Cc * d.W * 4, cudaMemcpyDeviceToDevice, c->stream));
    }
    lin_dx(c, dh1, D, dP, NP, c->w_in[li], N, NP, D, 0.f);
    lin_dw(c, GPG(c, PI(li, L_Q), PI(li, L_G1)), dP, NP, h1, D, N, NP, D);
    float *xr = BF(c, B_XR, float, (long)N * D), *rstd1 = BF(c, B_RSTD1, float, N);
    k_rmsnorm_bwd<<<rb_blocks, 256, D * 4, c->stream>>>(dx, dh1, xr, rstd1, PP(c, PI(li, L_N1)), GP(c, PI(li, L_N1)), N, D, 1, 1);
    // ---- memory read backward
    if (mem) {
        float *mq = BF(c, B_MQ, float, (long)N * d.dm), *phiq = BF(c, B_PHIQ, float, (long)N * d.dk), *denq = BF(c, B_DENQ, float, (long)N * d.Hm);
        float *numq = BF(c, B_NUMQ, float, (long)N * D), *dnum = BF(c, B_DNUMQ, float, (long)N * D), *dden = BF(c, B_DDENQ, float, (long)N * d.Hm);
        float *dphi = BF(c, B_DPHIQ, float, (long)N * d.dk), *dmq = BF(c, B_DMQ, float, (long)N * d.dm);
        k_read_bwd<<<N, 256, 0, c->stream>>>(dx, numq, denq, dnum, dden, D, d.Hm);
        long long sW = (long long)d.dk * D / d.Hm;
        for (int h = 0; h < d.Hm; h++) {
            // dW += phi^T dnum ; dphi = dnum W^T
            mm(c, 1, 0, dkh, dvh, s.T1, 1.f, phiq + h * dkh, F32, d.dk, (long long)s.T1 * d.dk, dnum + h * dvh, F32, D, (long long)s.T1 * D, 1.f,
               dst->Wm + (long)h * dkh * dvh, F32, dvh, sW, s.B);
            mm(c, 0, 1, s.T1, dkh, dvh, 1.f, dnum + h * dvh, F32, D, (long long)s.T1 * D, st_in->Wm + (long)h * dkh * dvh, F32, dvh, sW, 0.f,
               dphi + h * dkh, F32, d.dk, (long long)s.T1 * d.dk, s.B);
        }
        k_dphi_add<<<(N * d.Hm + 7) / 8, 256, 0, c->stream>>>(dphi, dden, st_in->z, NULL, NULL, N, s.T1, dkh, d.Hm);
        k_dz_acc<<<s.B * d.Hm, 256, 0, c->stream>>>(dst->z, dden, phiq, s.T1, dkh, d.Hm);
        k_dpfp_bwd<<<(N * d.Hm + 7) / 8, 256, 8 * dkh * 4, c->stream>>>(mq, d.dm, dphi, dmq, d.dm, N, d.dm, d.Hm);
        lin32_dx(c, dx, D, dmq, d.dm, PP(c, PI(li, L_MQ)), N, d.dm, D, 1.f);
        lin32_dw(c, GP(c, PI(li, L_MQ)), dmq, d.dm, x, D, N, d.dm, D);
    }
    (void)xo;
}

// =============================================================================================== host: weights
static void pack_weights(Ctx* c) {
    Dims& d = c->d;
    // packed floatX buffer: per layer w_in [NP x D], f2, g2, o, w12, w3 ; lm_head
    long per = (long)d.NP * d.D + (long)d.KDt * d.Hv + (long)d.VDt * d.Hv + (long)d.D * d.VDt + 2L * d.F * d.D + (long)d.D * d.F;
    long tot = per * d.L + (long)d.V * d.D;
    floatX* p = BF(c, B_PACK, floatX, tot);
    const int maxn = 32 * MAXL;
    const float* src[maxn]; floatX* dst[maxn]; long n[maxn];
    int cnt = 0;
    for (int i = 0; i < d.L; i++) {
        c->w_in[i] = p;
        for (int j = L_Q; j <= L_G1; j++) { src[cnt] = PP(c, PI(i, j)); dst[cnt] = p; n[cnt] = d.numel[PI(i, j)]; p += n[cnt]; cnt++; }
        c->w_f2[i] = p; src[cnt] = PP(c, PI(i, L_F2)); dst[cnt] = p; n[cnt] = d.numel[PI(i, L_F2)]; p += n[cnt]; cnt++;
        c->w_g2[i] = p; src[cnt] = PP(c, PI(i, L_G2)); dst[cnt] = p; n[cnt] = d.numel[PI(i, L_G2)]; p += n[cnt]; cnt++;
        c->w_o[i] = p; src[cnt] = PP(c, PI(i, L_O)); dst[cnt] = p; n[cnt] = d.numel[PI(i, L_O)]; p += n[cnt]; cnt++;
        c->w_12[i] = p;
        src[cnt] = PP(c, PI(i, L_W1)); dst[cnt] = p; n[cnt] = d.numel[PI(i, L_W1)]; p += n[cnt]; cnt++;
        src[cnt] = PP(c, PI(i, L_W2)); dst[cnt] = p; n[cnt] = d.numel[PI(i, L_W2)]; p += n[cnt]; cnt++;
        c->w_3[i] = p; src[cnt] = PP(c, PI(i, L_W3)); dst[cnt] = p; n[cnt] = d.numel[PI(i, L_W3)]; p += n[cnt]; cnt++;
    }
    c->w_lm = p; src[cnt] = d.c.tie_emb ? PP(c, P_EMB) : PP(c, P_LMHEAD); dst[cnt] = p; n[cnt] = (long)d.V * d.D; p += n[cnt]; cnt++;
    // descriptor table to device
    size_t tb = (size_t)cnt * (2 * sizeof(void*) + sizeof(long));
    char* dtab = (char*)ensure(&c->buf[B_PACKTAB], tb);
    CK(cudaEventSynchronize(c->pack_done));   // the previous copy out of htab is done (long ago, normally)
    char* htab = c->htab;
    memcpy(htab, src, cnt * sizeof(void*));
    memcpy(htab + cnt * sizeof(void*), dst, cnt * sizeof(void*));
    memcpy(htab + 2 * cnt * sizeof(void*), n, cnt * sizeof(long));
    CK(cudaMemcpyAsync(dtab, htab, tb, cudaMemcpyHostToDevice, c->stream));
    dim3 g(256, cnt);
    k_pack<<<g, 256, 0, c->stream>>>((const float* const*)dtab, (floatX* const*)(dtab + cnt * sizeof(void*)),
                                     (const long*)(dtab + 2 * cnt * sizeof(void*)), cnt);
    CK(cudaEventRecord(c->pack_done, c->stream));
}

// =============================================================================================== host: final norm / loss / logits
// rows of x (one piece): real rows (b, t < Tr) of [B, T1, D]. Writes token losses, logits (fp32, [B, T, V] or last), and, if
// grad, the grad of x's real rows into dxo (fp32 [B, T1, D], memory rows untouched) + lm_head / norm grads.
static void head(Ctx* c, const Step& s, const float* x, int start, int T, const int64_t* labels, float* tl, float* logits, int logits_last,
                 int is_last_piece, const float* inv_n, float* dxo) {
    Dims& d = c->d;
    int D = d.D, V = d.V, R = s.B * s.Tr;
    floatX* hf = BF(c, B_HF, floatX, (long)R * D);
    float* rstdf = BF(c, B_RSTDF, float, R);
    k_rmsnorm<<<R, 256, 0, c->stream>>>(NULL, x, NULL, NULL, NULL, 1, hf, rstdf, PP(c, P_NORMF), D, d.c.norm_eps, s.Tr, s.T1);
    int64_t* tgt = BF(c, B_TGT, int64_t, R);
    if (labels) k_targets<<<(R + 255) / 256, 256, 0, c->stream>>>(tgt, labels, s.B, s.Tr, T, start);
    if (logits && !logits_last) {
        mm(c, 0, 1, s.Tr, V, D, 1.f, hf, CUDA_X, D, (long long)s.Tr * D, c->w_lm, CUDA_X, D, 0, 0.f, logits + (long)start * V, F32, V,
           (long long)T * V, s.B);
        if (labels && !dxo) k_ce<float><<<R, 512, 0, c->stream>>>(logits, V, s.Tr, T, start, 0, tgt, tl, s.Tr, T - 1, start, inv_n, 0);
    }
    if (logits && logits_last && is_last_piece)
        mm(c, 0, 1, 1, V, D, 1.f, hf + (long)(s.Tr - 1) * D, CUDA_X, D, (long long)s.Tr * D, c->w_lm, CUDA_X, D, 0, 0.f, logits, F32, V, V, s.B);
    if (labels && (dxo || !logits || logits_last)) {
        int LC = 4096;
        floatX* lg = BF(c, B_LOGITS, floatX, (long)LC * V);
        floatX* dhf = dxo ? BF(c, B_DHF, floatX, (long)R * D) : NULL;
        for (int r0 = 0; r0 < R; r0 += LC) {
            int rn = R - r0 < LC ? R - r0 : LC;
            lin(c, lg, V, hf + (long)r0 * D, D, c->w_lm, rn, V, D, 0.f);
            k_ce<floatX><<<rn, 512, 0, c->stream>>>(lg, V, 0, 0, 0, r0, tgt, tl, s.Tr, T - 1, start, inv_n, dxo != NULL);
            if (dxo) {
                lin_dx(c, dhf + (long)r0 * D, D, lg, V, c->w_lm, rn, V, D, 0.f);
                float* gl = GP(c, P_LMHEAD);
                if (gl) mm(c, 1, 0, V, D, rn, 1.f, lg, CUDA_X, V, 0, hf + (long)r0 * D, CUDA_X, D, 0, 1.f, gl, F32, D, 0, 1);
            }
        }
        if (dxo) k_rmsnorm_bwd<<<2 * c->sms, 256, D * 4, c->stream>>>(dxo, dhf, x, rstdf, PP(c, P_NORMF), GP(c, P_NORMF), R, D, s.Tr, s.T1);
    }
}

// =============================================================================================== API
#define API extern "C"

typedef struct {
    const int64_t* ids;          // [B, T] device
    int B, T;
    const int64_t* labels;       // [B, T] device or NULL
    const float* inv_n;          // device scalar 1/#valid (for the gradient) or NULL
    float* token_losses;         // [B, T-1] device or NULL
    float* logits;               // [B, T, V] or [B, V] (logits_last) device fp32, or NULL
    int logits_last;
    int position;                // tokens already in the current segment (state continuation)
    int train;                   // record for backward (state must be zero / NULL)
    const uint8_t* reset;        // [n_pieces, B] device masks (state dropout), or NULL
    float** state;               // per layer 4 pointers (S, cs, Wm, z): in and out (updated in place); NULL = zero state, no output
    float* grad;                 // flat grad buffer (train)
    int repack;                  // weights changed since the last call: re-pack the floatX copies
    void* stream;
} ArmtForwardArgs;

static std::string g_err;
API const char* armt_last_error() { return g_err.c_str(); }
#define API_TRY try {
#define API_CATCH(ret) } catch (const std::exception& e_) { g_err = e_.what(); return ret; }
API void armt_set_allocator(armt_alloc_fn a, armt_free_fn f) { g_alloc = a; g_free = f; }
API void* armt_create(const ArmtConfig* cfg) {
    API_TRY
    if (cfg->n_layers > MAXL) armt_fail("at most " + std::to_string(MAXL) + " layers");
    if (cfg->head_dim != 64 && cfg->head_dim != 128) armt_fail("head_dim must be 64 or 128");
    if (cfg->conv_size != 4) armt_fail("conv_size must be 4");
    Ctx* c = (Ctx*)calloc(1, sizeof(Ctx));
    make_dims(&c->d, cfg);
    CB(cublasCreate(&c->blas));
    int dev; CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&c->sms, cudaDevAttrMultiProcessorCount, dev));
    CK(cudaMallocHost((void**)&c->htab, 32 * MAXL * 24));
    CK(cudaEventCreateWithFlags(&c->pack_done, cudaEventDisableTiming));
    c->gmask = (uint8_t*)calloc(c->d.n_params, 1);
    memset(c->gmask, 1, c->d.n_params);
    return c;
    API_CATCH(NULL)
}
// release(0): frees the training stores (layer inputs, per-segment states, their gradients), the bulk of a training
// step's memory, back to the allocator's cache; called after every backward. The packed weights and the per-step scratch
// stay (no re-pack, no re-allocation on the next micro-step). release(1): frees everything (gives all memory back).
API int armt_release(void* p, int all) {
    API_TRY
    Ctx* c = (Ctx*)p;
    if (all) {
        for (int i = 0; i < B_COUNT; i++) buf_free(&c->buf[i]);
        c->w_lm = NULL;   // the packed weights are gone: re-pack on the next forward
    } else {
        const int stores[] = {B_XS, B_DXS, B_ST_S, B_ST_CS, B_ST_W, B_ST_Z, B_DST_S, B_DST_CS, B_DST_W, B_DST_Z, B_XOT};
        for (int id : stores) buf_free(&c->buf[id]);
    }
    return 0;
    API_CATCH(-1)
}
API void armt_destroy(void* p) {
    Ctx* c = (Ctx*)p;
    if (!c) return;
    try { for (int i = 0; i < B_COUNT; i++) buf_free(&c->buf[i]); } catch (...) {}
    cublasDestroy(c->blas);
    cudaFreeHost(c->htab);
    cudaEventDestroy(c->pack_done);
    free(c->tr_Tr); free(c->tr_Tm); free(c->tr_start); free(c->gmask);
    free(c);
}
// which parameters need gradients (1 byte each, canonical order); the others' gradient work is skipped
API void armt_set_grad_mask(void* p, const uint8_t* mask) { Ctx* c = (Ctx*)p; memcpy(c->gmask, mask, c->d.n_params); }
API long armt_num_params(void* p) { return ((Ctx*)p)->d.n_params; }
API long armt_param_numel(void* p, int i) { return ((Ctx*)p)->d.numel[i]; }
API long armt_param_offset(void* p, int i) { return ((Ctx*)p)->d.offset[i]; }
API long armt_grad_size(void* p) { return ((Ctx*)p)->d.total; }
API void armt_set_tf32(void* p, int on) { ((Ctx*)p)->d.c.allow_tf32 = on; }

static void set_stream(Ctx* c, void* stream) {
    c->stream = (cudaStream_t)stream;
    g_cur_stream = c->stream;
    CB(cublasSetStream(c->blas, c->stream));
}

template <int KD>
static void forward_impl(Ctx* c, const ArmtForwardArgs* a) {
    Dims& d = c->d;
    int B = a->B, T = a->T, S = d.c.segment_size, D = d.D;
    // pieces at absolute segment boundaries
    int np = 0, pos = a->position;
    std::vector<int> pst, pen, pw;   // pieces at absolute segment boundaries (any number: long inputs are fine)
    for (int st = 0; st < T;) {
        int en = st + S - pos < T ? st + S - pos : T;
        int done = pos + en - st == S;
        pst.push_back(st); pen.push_back(en); pw.push_back(done && d.M > 0); np++;
        pos = done ? 0 : pos + en - st;
        st = en;
    }
    int T1max = 0;
    for (int k = 0; k < np; k++) { int t1 = pen[k] - pst[k] + (pw[k] ? d.M : 0); if (t1 > T1max) T1max = t1; }
    long rowsmax = (long)B * T1max;
    long nS = (long)B * d.H * KD * KD, nCs = (long)B * d.Cc * d.W, nW = (long)B * d.dk * D / d.Hm, nZ = (long)B * d.dk;
    c->grad = a->train ? a->grad : NULL;
    if (a->train) c->tr_grad = a->grad;
    if (a->repack || !c->w_lm) pack_weights(c);
    if (!a->train) {
        float *x0 = BF(c, B_X0, float, rowsmax * D), *x1 = BF(c, B_X1, float, rowsmax * D);
        long nzb = nS > nW ? (nS > nCs ? nS : nCs) : (nW > nCs ? nW : nCs);
        float* zbuf = BF(c, B_ZERO, float, nzb);
        zero(c, zbuf, sizeof(float) * nzb);
        for (int k = 0; k < np; k++) {
            int Tr = pen[k] - pst[k], Tm = pw[k] ? d.M : 0;
            Step s = make_step(c, B, Tr, Tm);
            k_embed<<<s.N, 256, 0, c->stream>>>(x0, a->ids, T, pst[k], PP(c, P_EMB), PP(c, P_MEMTOK), Tr, Tm, D);
            if (a->reset && a->state) for (int i = 0; i < d.L; i++) {
                LAUNCH1D(k_mask_zero, nS, a->state[4 * i + 0], a->reset + (long)k * B, (long)d.H * KD * KD, B);
                LAUNCH1D(k_mask_zero, nCs, a->state[4 * i + 1], a->reset + (long)k * B, (long)d.Cc * d.W, B);
            }
            float *xi = x0, *xo = x1;
            for (int i = 0; i < d.L; i++) {
                LState in, out;
                if (a->state) { in.S = a->state[4 * i]; in.cs = a->state[4 * i + 1]; in.Wm = a->state[4 * i + 2]; in.z = a->state[4 * i + 3]; }
                else { in.S = in.cs = NULL; in.Wm = in.z = zbuf; }
                if (!d.has_mem[i]) { in.Wm = in.z = NULL; }
                out = in;
                if (!a->state) {   // no state kept: carry within this call through scratch state buffers
                    float* st = BF(c, B_IST_S, float, nS * d.L); float* scs = BF(c, B_IST_CS, float, nCs * d.L);
                    float* sw = BF(c, B_IST_W, float, nW * d.L); float* sz = BF(c, B_IST_Z, float, nZ * d.L);
                    if (k == 0) { zero(c, st + nS * i, nS * 4); zero(c, scs + nCs * i, nCs * 4); zero(c, sw + nW * i, nW * 4); zero(c, sz + nZ * i, nZ * 4); }
                    in.S = st + nS * i; in.cs = scs + nCs * i;
                    in.Wm = d.has_mem[i] ? sw + nW * i : NULL; in.z = d.has_mem[i] ? sz + nZ * i : NULL;
                    if (a->reset) {
                        LAUNCH1D(k_mask_zero, nS, in.S, a->reset + (long)k * B, (long)d.H * KD * KD, B);
                        LAUNCH1D(k_mask_zero, nCs, in.cs, a->reset + (long)k * B, (long)d.Cc * d.W, B);
                    }
                    out = in;
                }
                if (d.has_mem[i] && !in.Wm) { in.Wm = zbuf; in.z = zbuf; out.Wm = out.z = NULL; }
                layer_fwd<KD>(c, i, s, xi, xo, &in, &out);
                float* t_ = xi; xi = xo; xo = t_;
            }
            head(c, s, xi, pst[k], T, a->labels, a->token_losses, a->logits, a->logits_last, k == np - 1, a->inv_n, NULL);
        }
        return;
    }
    // ---------------- training: store layer inputs and per-segment input states
    if (a->position != 0 || a->state) armt_fail("a training forward must start from an empty state");
    long xs_per = rowsmax * D;
    float* XS = BF(c, B_XS, float, xs_per * (d.L + 1) * np);
    float* DXS = BF(c, B_DXS, float, xs_per * np);
    float *SS = BF(c, B_ST_S, float, nS * d.L * np), *SCS = BF(c, B_ST_CS, float, nCs * d.L * np);
    float *SW = BF(c, B_ST_W, float, nW * d.L * np), *SZ = BF(c, B_ST_Z, float, nZ * d.L * np);
    for (int i = 0; i < d.L; i++) {   // initial (k = 0) state of every layer
        long si = (long)i * np;
        zero(c, SS + nS * si, nS * 4); zero(c, SCS + nCs * si, nCs * 4); zero(c, SW + nW * si, nW * 4); zero(c, SZ + nZ * si, nZ * 4);
    }
    c->tr_B = B; c->tr_T = T; c->tr_nseg = np; c->tr_T1max = T1max; c->tr_reset = a->reset;
    if (np > c->tr_cap) {
        c->tr_cap = np;
        c->tr_Tr = (int*)realloc(c->tr_Tr, np * sizeof(int));
        c->tr_Tm = (int*)realloc(c->tr_Tm, np * sizeof(int));
        c->tr_start = (int*)realloc(c->tr_start, np * sizeof(int));
    }
    for (int k = 0; k < np; k++) {
        int Tr = pen[k] - pst[k], Tm = pw[k] ? d.M : 0;
        if (k == np - 1) Tm = 0;   // the last memory write feeds nothing that is trained
        c->tr_Tr[k] = Tr; c->tr_Tm[k] = Tm; c->tr_start[k] = pst[k];
        Step s = make_step(c, B, Tr, Tm);
        float* x0 = XS + xs_per * ((long)k * (d.L + 1));
        k_embed<<<s.N, 256, 0, c->stream>>>(x0, a->ids, T, pst[k], PP(c, P_EMB), PP(c, P_MEMTOK), Tr, Tm, D);
        for (int i = 0; i < d.L; i++) {
            long si = (long)i * np + k;
            LState in = {SS + nS * si, SCS + nCs * si, d.has_mem[i] ? SW + nW * si : NULL, d.has_mem[i] ? SZ + nZ * si : NULL};
            if (a->reset && k > 0) {
                LAUNCH1D(k_mask_zero, nS, in.S, a->reset + (long)k * B, (long)d.H * KD * KD, B);
                LAUNCH1D(k_mask_zero, nCs, in.cs, a->reset + (long)k * B, (long)d.Cc * d.W, B);
            }
            LState out = {NULL, NULL, NULL, NULL};
            if (k + 1 < np) out = {SS + nS * (si + 1), SCS + nCs * (si + 1), d.has_mem[i] ? SW + nW * (si + 1) : NULL, d.has_mem[i] ? SZ + nZ * (si + 1) : NULL};
            layer_fwd<KD>(c, i, s, x0 + xs_per * i, x0 + xs_per * (i + 1), &in, &out);
        }
        float* dx = DXS + xs_per * k;
        zero(c, dx, (size_t)s.N * D * 4);
        head(c, s, x0 + xs_per * d.L, pst[k], T, a->labels, a->token_losses, a->logits, a->logits_last, k == np - 1, a->inv_n, dx);
    }
}

template <int KD>
static void backward_impl(Ctx* c, const int64_t* ids, const float* gscale) {
    Dims& d = c->d;
    if (!c->tr_grad || !c->buf[B_XS].p) armt_fail("backward without a recorded training forward");
    c->grad = c->tr_grad;
    int B = c->tr_B, T = c->tr_T, np = c->tr_nseg, D = d.D;
    long xs_per = (long)B * c->tr_T1max * D;
    long nS = (long)B * d.H * KD * KD, nCs = (long)B * d.Cc * d.W, nW = (long)B * d.dk * D / d.Hm, nZ = (long)B * d.dk;
    float *XS = (float*)c->buf[B_XS].p, *DXS = (float*)c->buf[B_DXS].p;
    float *SS = (float*)c->buf[B_ST_S].p, *SCS = (float*)c->buf[B_ST_CS].p, *SW = (float*)c->buf[B_ST_W].p, *SZ = (float*)c->buf[B_ST_Z].p;
    float *dS = BF(c, B_DST_S, float, nS), *dCs = BF(c, B_DST_CS, float, nCs), *dW = BF(c, B_DST_W, float, nW), *dZ = BF(c, B_DST_Z, float, nZ);
    for (int i = d.L - 1; i >= 0; i--) {
        zero(c, dS, nS * 4); zero(c, dCs, nCs * 4); zero(c, dW, nW * 4); zero(c, dZ, nZ * 4);
        for (int k = np - 1; k >= 0; k--) {
            Step s = make_step(c, B, c->tr_Tr[k], c->tr_Tm[k]);
            long si = (long)i * np + k;
            float* x0 = XS + xs_per * ((long)k * (d.L + 1));
            LState in = {SS + nS * si, SCS + nCs * si, d.has_mem[i] ? SW + nW * si : NULL, d.has_mem[i] ? SZ + nZ * si : NULL};
            float* xo_tmp = BF(c, B_XOT, float, xs_per);
            LState disc = {NULL, NULL, d.has_mem[i] ? BF(c, B_WDISC, float, nW) : NULL, d.has_mem[i] ? BF(c, B_ZDISC, float, nZ) : NULL};
            layer_fwd<KD>(c, i, s, x0 + xs_per * i, xo_tmp, &in, &disc);
            float* dx = DXS + xs_per * k;
            LState dst = {dS, dCs, d.has_mem[i] ? dW : NULL, d.has_mem[i] ? dZ : NULL};
            layer_bwd<KD>(c, i, s, x0 + xs_per * i, x0 + xs_per * (i + 1), &in, dx, &dst);
            if (c->tr_reset && k > 0) {
                LAUNCH1D(k_mask_zero, nS, dS, c->tr_reset + (long)k * B, (long)d.H * KD * KD, B);
                LAUNCH1D(k_mask_zero, nCs, dCs, c->tr_reset + (long)k * B, (long)d.Cc * d.W, B);
            }
        }
    }
    for (int k = 0; k < np; k++) {
        Step s = make_step(c, B, c->tr_Tr[k], c->tr_Tm[k]);
        k_embed_bwd<<<s.N, 256, 0, c->stream>>>(DXS + xs_per * k, ids, T, c->tr_start[k], GP(c, P_EMB), GP(c, P_MEMTOK), s.Tr, s.Tm, D);
    }
    if (gscale) LAUNCH1D(k_scale, d.total, c->grad, d.total, gscale);
}

// forward. params: n_params device pointers in canonical order.
API int armt_forward(void* p, const float** params, const ArmtForwardArgs* a) {
    API_TRY
    Ctx* c = (Ctx*)p;
    c->params = params;
    set_stream(c, a->stream);
    if (c->d.KD == 64) forward_impl<64>(c, a); else forward_impl<128>(c, a);
    CK(cudaGetLastError());
    return 0;
    API_CATCH(-1)
}
// backward of the last training forward: grads (+)= into the flat buffer given to the forward, then scaled by *gscale.
API int armt_backward(void* p, const float** params, const int64_t* ids, const float* gscale, void* stream) {
    API_TRY
    Ctx* c = (Ctx*)p;
    c->params = params;
    set_stream(c, stream);
    if (c->d.KD == 64) backward_impl<64>(c, ids, gscale); else backward_impl<128>(c, ids, gscale);
    CK(cudaGetLastError());
    c->tr_grad = NULL;
    return 0;
    API_CATCH(-1)
}

