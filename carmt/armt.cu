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
#include <ctype.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <map>
#include <mutex>
#include <algorithm>
#include <functional>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

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
    int kind;          // 0: GDN-2 (fla GatedDeltaNet2); 1: Qwen3.5 Gated DeltaNet (one gate and one beta per head)
    int norm_plus1;    // the block and final RMSNorms scale by (1 + w) (Qwen3.5)
    int attn[MAXL];    // 1: an attention layer of a hybrid: no parameters here; it runs outside (armt_hy_* API)
} ArmtConfig;

// canonical parameter order (the Python side passes pointers in this order; grads use a flat buffer in this order)
enum { P_EMB, P_NORMF, P_LMHEAD, P_MEMTOK, P_GLOBAL };
enum { L_N1, L_Q, L_K, L_V, L_B, L_W, L_F1, L_G1, L_QC, L_KC, L_VC, L_F2, L_ALOG, L_DTB, L_G2, L_G2B, L_ON, L_O, L_N2,
       L_W1, L_W2, L_W3, L_MQ, L_MK, L_MV, L_MB, L_MBB, L_COUNT };

typedef struct {
    ArmtConfig c;
    int D, L, H, KD, KDt, VDt, Hv, F, W, NP, Cc, dm, Hm, dk, nmb, M, V;
    int sg;                  // kind 1: scalar gate / beta per head
    int ob, ow, oa, oz;      // columns of the packed projection P: beta (b), w, gate input (kind 1: a), output gate (kind 1: z)
    float woff;              // RMSNorm weight offset (1 for (1 + w) norms)
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
    d->Cc = 2 * d->KDt + d->VDt;
    d->sg = c->kind == 1;
    d->woff = c->norm_plus1 ? 1.f : 0.f;
    if (!d->sg) {   // [q | k | v | b | w | f1 | g1]
        d->NP = 2 * d->KDt + d->VDt + d->KDt + d->VDt + 2 * d->Hv;
        d->ob = 3 * d->KDt; d->ow = 3 * d->KDt + d->VDt; d->oa = -1; d->oz = -1;
    } else {        // [qkv | z | beta | a]: the slots L_Q = in_proj_qkv, L_K = in_proj_z, L_B = in_proj_b, L_F1 = in_proj_a
        d->NP = d->Cc + d->VDt + 2 * d->H;
        d->oz = d->Cc; d->ob = d->ow = d->Cc + d->VDt; d->oa = d->ob + d->H;
    }
    d->M = c->num_mem; d->dm = c->d_mem; d->Hm = c->mem_heads > 0 ? c->mem_heads : 1; d->dk = 6 * c->d_mem;
    d->nmb = c->gating ? d->D : d->Hm; d->V = c->vocab;
    long* n = d->numel;
    n[P_EMB] = (long)d->V * d->D; n[P_NORMF] = d->D; n[P_LMHEAD] = c->tie_emb ? 0 : (long)d->V * d->D; n[P_MEMTOK] = (long)d->M * d->D;
    for (int i = 0; i < d->L; i++) {
        long* p = n + P_GLOBAL + i * L_COUNT;
        int D = d->D;
        d->has_mem[i] = d->M > 0 && c->wrap[i] && !c->attn[i];
        if (c->attn[i]) continue;   // an attention layer: no parameters in this library
        if (d->sg) {
            // Qwen3.5: in_proj_qkv -> L_Q, in_proj_z -> L_K, in_proj_b -> L_B, in_proj_a -> L_F1, conv1d (all channels) -> L_QC,
            // A_log, dt_bias (per head), norm (gated, per channel of a head) -> L_ON, out_proj -> L_O; the other slots are empty.
            // The grad layout of [L_Q .. L_G1] is then exactly the packed [qkv | z | beta | a] matrix.
            p[L_N1] = D; p[L_Q] = (long)d->Cc * D; p[L_K] = (long)d->VDt * D; p[L_B] = (long)d->H * D; p[L_F1] = (long)d->H * D;
            p[L_QC] = (long)d->Cc * d->W; p[L_ALOG] = d->H; p[L_DTB] = d->H; p[L_ON] = d->KD; p[L_O] = (long)D * d->VDt; p[L_N2] = D;
            p[L_W1] = (long)d->F * D; p[L_W2] = (long)d->F * D; p[L_W3] = (long)D * d->F;
        } else {
        p[L_N1] = D; p[L_Q] = (long)d->KDt * D; p[L_K] = (long)d->KDt * D; p[L_V] = (long)d->VDt * D; p[L_B] = (long)d->KDt * D;
        p[L_W] = (long)d->VDt * D; p[L_F1] = (long)d->Hv * D; p[L_G1] = (long)d->Hv * D;
        p[L_QC] = (long)d->KDt * d->W; p[L_KC] = (long)d->KDt * d->W; p[L_VC] = (long)d->VDt * d->W;
        p[L_F2] = (long)d->KDt * d->Hv; p[L_ALOG] = d->H; p[L_DTB] = d->KDt; p[L_G2] = (long)d->VDt * d->Hv; p[L_G2B] = d->VDt;
        p[L_ON] = d->KD; p[L_O] = (long)D * d->VDt; p[L_N2] = D;
        p[L_W1] = (long)d->F * D; p[L_W2] = (long)d->F * D; p[L_W3] = (long)D * d->F;
        }
        if (d->has_mem[i]) {
            p[L_MQ] = (long)d->dm * D; p[L_MK] = (long)d->dm * D; p[L_MV] = (long)D * D; p[L_MB] = (long)d->nmb * D; p[L_MBB] = d->nmb;
        }
    }
    d->n_params = P_GLOBAL + (long)d->L * L_COUNT;
    // Every parameter's gradient starts 256-byte aligned: the weight-gradient GEMMs accumulate into it, and an output that
    // is only 4-byte aligned (e.g. every other layer, behind the memory's single write-gate bias) limits cuBLAS to slow
    // kernels. The groups that one GEMM writes as one matrix stay contiguous: [q .. g1], the three convs, [w1 | w2].
    long off = 0;
    for (long i = 0; i < d->n_params; i++) {
        int j = i >= P_GLOBAL ? (int)((i - P_GLOBAL) % L_COUNT) : -1;
        bool cont = (j > L_Q && j <= L_G1) || j == L_KC || j == L_VC || j == L_W2;
        if (!cont) off = (off + 63) & ~63L;
        d->offset[i] = off; off += n[i];
    }
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
    B_PACK, B_PACKTAB, B_WCAT, B_AMX, B_PREDVN, B_IST_S, B_IST_CS, B_IST_W, B_IST_Z, B_XOT, B_S0LO, B_O32, B_LTWS, B_TUNE,
    B_COUNT
};

typedef struct {
    Dims d;
    cublasHandle_t blas;
    cublasLtHandle_t lt;
    int notune;               // > 0: mm uses cublasGemmEx (GEMMs that read the parameters directly, see lin32)
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
    // hybrid (layer-major) call in progress: armt_hy_begin .. armt_hy_head (.. backward)
    int hy_active, hy_train, hy_np, hy_nw;
    int* hy_w;                // per piece: index among the pieces with memory tokens, or -1
    void* hy_args;            // a copy of the ArmtForwardArgs of armt_hy_begin
    struct Prof* prof;        // ARMT_PROF=1: phase profiler (else NULL)
    int prof_pass;            // 0 forward, 1 the backward's recompute, 2 backward
} Ctx;

// =============================================================================================== optional phase profiler
// ARMT_PROF=1: GPU time per phase of a layer, cuBLAS included, separately for the forward, the backward's recompute and the
// backward. Each prof_mark() records a CUDA event on the library's stream and starts a phase; the time up to the next mark
// is charged to it. The events are summed at the end of every API call, which then synchronizes (profiling runs only).
enum { PR_IDLE, PR_EMBED, PR_MEMREAD, PR_NORM1, PR_PROJ, PR_GATES, PR_PREP, PR_CONVST, PR_INTRA, PR_SG_A, PR_STATE, PR_SG_O,
       PR_ONORM, PR_OPROJ, PR_NORM2, PR_MLP, PR_MEMWRITE, PR_HEAD, PR_SG_PRE, PR_SG_POST, PR_PREP1, PR_PREP2, PR_COPY, PR_COUNT };
static const char* PR_NAME[PR_COUNT] = {
    "idle / untracked", "embeddings", "memory read", "norm 1", "input projections", "gate projections (f2, g2)",
    "token prep (conv, l2norm, gates)", "conv state", "scan: intra-chunk", "scan: GEMMs u = A wv, Wm = A y", "scan: state pass",
    "scan: GEMMs o = Qg S0 + M vn", "gated output norm", "output projection", "norm 2", "MLP", "memory write", "final norm + loss",
    "scan bwd: GEMMs M^T do, Qg^T do", "scan bwd: GEMMs after the state pass", "token prep bwd 1", "token prep bwd 2 (conv)",
    "copies / memsets"};
struct Prof {
    std::vector<cudaEvent_t> ev;
    std::vector<short> sec, pass;
    size_t used = 0;
    double ms[3][PR_COUNT] = {};
};
static void prof_mark(Ctx* c, int sec) {
    Prof* p = c->prof;
    if (!p) return;
    if (p->used == p->ev.size()) {
        cudaEvent_t e;
        CK(cudaEventCreate(&e));
        p->ev.push_back(e); p->sec.push_back(0); p->pass.push_back(0);
    }
    CK(cudaEventRecord(p->ev[p->used], c->stream));
    p->sec[p->used] = (short)sec; p->pass[p->used] = (short)c->prof_pass; p->used++;
}
static void prof_flush(Ctx* c) {
    Prof* p = c->prof;
    if (!p || !p->used) return;
    prof_mark(c, PR_IDLE);
    CK(cudaEventSynchronize(p->ev[p->used - 1]));
    for (size_t i = 0; i + 1 < p->used; i++) {
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, p->ev[i], p->ev[i + 1]));
        p->ms[p->pass[i]][p->sec[i]] += ms;
    }
    p->used = 0;
}

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
// cuBLAS's default heuristic picks slow kernels for many of these shapes, and so do the top candidates of cuBLASLt's own
// heuristic (by up to ~20% for the weight-gradient GEMMs). The first time a shape (with its operands' alignments) is
// seen, mm times on a scratch output: the default cublasGemmEx; cuBLASLt's heuristic candidates, for the operands'
// alignments and for 4-byte alignment (kernels the first query leaves out, which can still be the fastest); and, for the
// large shapes, the fastest few of an exhaustive search over cuBLASLt's configurations (algorithm x tile x stages x CTA
// swizzle x custom option, then split-k x reduction scheme for the best of those). The candidates are timed in rounds of
// single calls, in a fresh random order every round (the GPU mostly runs at its power cap, and its clock drifts), and a
// candidate's median counts. The choice must not depend on timing noise (bf16 results differ slightly between algorithms,
// and DDP ranks / repeated runs should compute the same thing), so it is made once per GPU model and cuBLAS version and
// persisted: ARMT_GEMM_CACHE=<path prefix> (the Python side sets it next to the library) stores every choice in
// <prefix>_<gpu>_<cublasLt version>_t2.txt, written under a file lock, so concurrent processes (DDP ranks) and later runs
// read the first process's choice instead of tuning again. Plans are shared by every context of the process.
// ARMT_GEMM_TUNE=0: always cublasGemmEx; 1: no exhaustive search; 2 (default): as above.
// ARMT_GEMM_LOG=1: print each tuned shape's default and chosen time.
#define LT_WS (32u << 20)
#define LT_CAND 16
#define LT_EXH_FLOP 2e10     // exhaustive search for GEMMs of at least this many flops (~0.1 ms)
struct GemmPlan {
    int lt;                                   // 0: cublasGemmEx (default algorithm); 1: cublasLtMatmul with algo
    cublasLtMatmulDesc_t op = NULL;
    cublasLtMatrixLayout_t la = NULL, lb = NULL, lc = NULL;
    cublasLtMatmulAlgo_t algo;
    size_t ws = 0;
};
struct GemmChoice { int lt; cublasLtMatmulAlgo_t algo; size_t ws; };
static std::map<std::string, GemmPlan*> g_plans;        // process-wide
static std::map<std::string, GemmChoice> g_disk;        // the cache file's entries
static std::mutex g_plans_mu;
static int g_tune = -1, g_tune_log = 0;
static std::string g_cache_file;
static int env_int(const char* name, int def) { const char* v = getenv(name); return v && *v ? atoi(v) : def; }
static int ptr_align(const void* p) { uintptr_t a = (uintptr_t)p; int al = 16; while (al > 1 && (a % al)) al >>= 1; return al; }

static void gemm_tune_init() {
    if (g_tune >= 0) return;
    g_tune = env_int("ARMT_GEMM_TUNE", 2);
    g_tune_log = env_int("ARMT_GEMM_LOG", 0);
    const char* pre = getenv("ARMT_GEMM_CACHE");
    if (g_tune && pre && *pre) {
        int dev = 0; cudaDeviceProp pr;
        if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&pr, dev) == cudaSuccess) {
            std::string name = pr.name;
            for (char& ch : name) if (!isalnum((unsigned char)ch)) ch = '_';
            g_cache_file = std::string(pre) + "_" + name + "_" + std::to_string((long)cublasLtGetVersion()) + "_t2.txt";
        }
    }
}
// cache file: one line per shape: "<key> <lt> <ws> <algo as 16 hex digits x 8>"
static void disk_read(int fd) {
    std::string txt;
    char buf[1 << 16];
    lseek(fd, 0, SEEK_SET);
    for (ssize_t n; (n = read(fd, buf, sizeof(buf))) > 0;) txt.append(buf, (size_t)n);
    size_t pos = 0;
    while (pos < txt.size()) {
        size_t e = txt.find('\n', pos);
        if (e == std::string::npos) break;                       // a partial last line: ignored
        std::string line = txt.substr(pos, e - pos);
        pos = e + 1;
        size_t bar = line.find('|');
        if (bar == std::string::npos) continue;
        GemmChoice ch;
        unsigned long long w[8], ws;
        if (sscanf(line.c_str() + bar + 1, "%d %llu %llx %llx %llx %llx %llx %llx %llx %llx", &ch.lt, &ws, &w[0], &w[1], &w[2], &w[3],
                   &w[4], &w[5], &w[6], &w[7]) != 10) continue;
        static_assert(sizeof(cublasLtMatmulAlgo_t) == 64, "cublasLtMatmulAlgo_t layout");
        memcpy(&ch.algo, w, 64);
        ch.ws = (size_t)ws;
        g_disk.emplace(line.substr(0, bar), ch);                 // the first entry of a key wins
    }
}
static void disk_append(int fd, const std::string& key, const GemmChoice& ch) {
    unsigned long long w[8];
    memcpy(w, &ch.algo, 64);
    char tail[256];
    snprintf(tail, sizeof(tail), "|%d %llu %llx %llx %llx %llx %llx %llx %llx %llx\n", ch.lt, (unsigned long long)ch.ws, w[0], w[1], w[2], w[3],
             w[4], w[5], w[6], w[7]);
    std::string line = key + tail;
    lseek(fd, 0, SEEK_END);
    if (write(fd, line.data(), line.size()) != (ssize_t)line.size()) { /* best effort: the choice still holds in-process */ }
}

// descriptors of one shape (column-major view: C^T[N,M] = op(B)^T op(A)^T)
static GemmPlan* plan_desc(cublasOperation_t oa, cublasOperation_t ob, int M, int N, int K, cudaDataType ta, int lda, long long sA,
                           cudaDataType tb, int ldb, long long sB, cudaDataType tc, int ldc, long long sC, int batch, cublasComputeType_t ct) {
    GemmPlan* p = new GemmPlan();
    p->lt = 0;
    CB(cublasLtMatmulDescCreate(&p->op, ct, CUDA_R_32F));
    CB(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &oa, sizeof(oa)));
    CB(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &ob, sizeof(ob)));
    CB(cublasLtMatrixLayoutCreate(&p->la, tb, oa == CUBLAS_OP_N ? N : K, oa == CUBLAS_OP_N ? K : N, ldb));
    CB(cublasLtMatrixLayoutCreate(&p->lb, ta, ob == CUBLAS_OP_N ? K : M, ob == CUBLAS_OP_N ? M : K, lda));
    CB(cublasLtMatrixLayoutCreate(&p->lc, tc, N, M, ldc));
    if (batch > 1) {
        cublasLtMatrixLayout_t ls[3] = {p->la, p->lb, p->lc};
        long long st[3] = {sB, sA, sC};
        for (int i = 0; i < 3; i++) {
            CB(cublasLtMatrixLayoutSetAttribute(ls[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
            CB(cublasLtMatrixLayoutSetAttribute(ls[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &st[i], sizeof(st[i])));
        }
    }
    return p;
}

struct LtCand { cublasLtMatmulAlgo_t algo; size_t ws; };
static void lt_add(std::vector<LtCand>& v, const cublasLtMatmulAlgo_t& a, size_t ws) {
    for (auto& x : v) if (!memcmp(&x.algo, &a, sizeof(a))) return;
    v.push_back({a, ws});
}
static uint64_t g_shuffle = 0x2545F4914F6CDD1Dull;
static void shuffle(std::vector<int>& v) {   // deterministic (only the timing order depends on it)
    for (int i = (int)v.size() - 1; i > 0; i--) {
        g_shuffle = g_shuffle * 6364136223846793005ull + 1442695040888963407ull;
        std::swap(v[i], v[(g_shuffle >> 33) % (uint64_t)(i + 1)]);
    }
}
// cuBLASLt's heuristic candidates. aA, aB: alignments (bytes) of cuBLASLt's A and B (= our B and A), aC: of C / D
static void lt_heuristic(Ctx* c, GemmPlan* p, uint32_t aA, uint32_t aB, uint32_t aC, std::vector<LtCand>& v) {
    cublasLtMatmulPreference_t pref;
    CB(cublasLtMatmulPreferenceCreate(&pref));
    uint64_t wsmax = LT_WS;
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsmax, sizeof(wsmax)));
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &aA, sizeof(aA)));
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &aB, sizeof(aB)));
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &aC, sizeof(aC)));
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &aC, sizeof(aC)));
    cublasLtMatmulHeuristicResult_t res[LT_CAND];
    int n = 0;
    if (cublasLtMatmulAlgoGetHeuristic(c->lt, p->op, p->la, p->lb, p->lc, p->lc, pref, LT_CAND, res, &n) != CUBLAS_STATUS_SUCCESS) n = 0;
    cublasLtMatmulPreferenceDestroy(pref);
    for (int i = 0; i < n; i++)
        if (res[i].state == CUBLAS_STATUS_SUCCESS && res[i].workspaceSize <= LT_WS) lt_add(v, res[i].algo, res[i].workspaceSize);
}
// Exhaustive search: every configuration cuBLASLt accepts for the shape (algorithm x tile x stages x CTA swizzle x custom
// option, without split-k), quick-timed (one call each, in random order, after an untimed call); then split-k x reduction
// scheme for the `nbase` fastest. The `keep` fastest of each stage join v. tLA, tLB: types of cuBLASLt's A and B.
static void lt_exhaustive(Ctx* c, GemmPlan* p, cublasComputeType_t ct, cudaDataType tLA, cudaDataType tLB, cudaDataType tc,
                          uint32_t aA, uint32_t aB, uint32_t aC, const std::function<float(const LtCand&)>& quick,
                          std::vector<LtCand>& v, int nbase, int keep) {
    int ids[256], nids = 0;
    if (cublasLtMatmulAlgoGetIds(c->lt, ct, CUDA_R_32F, tLA, tLB, tc, tc, 256, ids, &nids) != CUBLAS_STATUS_SUCCESS) return;
    struct Cfg { LtCand k; int splitk; uint32_t redmask; float t; };
    auto check = [&](cublasLtMatmulAlgo_t& a, size_t* ws) {
        cublasLtMatmulHeuristicResult_t hr;
        if (cublasLtMatmulAlgoCheck(c->lt, p->op, p->la, p->lb, p->lc, p->lc, &a, &hr) != CUBLAS_STATUS_SUCCESS) return false;
        if (hr.state != CUBLAS_STATUS_SUCCESS || hr.workspaceSize > LT_WS) return false;
        *ws = hr.workspaceSize;
        return true;
    };
    auto cap = [](const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoCapAttributes_t at) {   // the 4-byte capabilities
        uint32_t x = 0;
        size_t sz = 0;
        cublasLtMatmulAlgoCapGetAttribute(&a, at, &x, sizeof(x), &sz);
        return x;
    };
    auto ids_of = [](const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoCapAttributes_t at, uint32_t undef) {
        size_t sz = 0;
        std::vector<uint32_t> r;
        cublasLtMatmulAlgoCapGetAttribute(&a, at, NULL, 0, &sz);
        r.resize(sz / 4);
        if (sz) cublasLtMatmulAlgoCapGetAttribute(&a, at, r.data(), sz, &sz);
        if (r.empty()) r.push_back(undef);
        return r;
    };
    auto qpass = [&](std::vector<Cfg>& cs) {
        std::vector<int> ord(cs.size());
        for (size_t i = 0; i < ord.size(); i++) ord[i] = (int)i;
        shuffle(ord);
        for (int i : ord) cs[i].t = quick(cs[i].k);
        std::stable_sort(cs.begin(), cs.end(), [](const Cfg& x, const Cfg& y) { return x.t < y.t; });
    };
    std::vector<Cfg> base, sk;
    for (int ii = 0; ii < nids; ii++) {
        cublasLtMatmulAlgo_t al;
        if (cublasLtMatmulAlgoInit(c->lt, ct, CUDA_R_32F, tLA, tLB, tc, tc, ids[ii], &al) != CUBLAS_STATUS_SUCCESS) continue;
        if (cap(al, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES) > aA || cap(al, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES) > aB ||
            cap(al, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES) > aC || cap(al, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES) > aC) continue;
        int splitk = (int)cap(al, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT), custmax = (int)cap(al, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX);
        uint32_t redmask = cap(al, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK), swzmax = cap(al, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT);
        for (uint32_t t : ids_of(al, CUBLASLT_ALGO_CAP_TILE_IDS, CUBLASLT_MATMUL_TILE_UNDEFINED))
            for (uint32_t sg : ids_of(al, CUBLASLT_ALGO_CAP_STAGES_IDS, CUBLASLT_MATMUL_STAGES_UNDEFINED))
                for (uint32_t cu = 0; (int)cu <= custmax; cu++)
                    for (uint32_t sw = 0; sw <= swzmax; sw++) {
                        cublasLtMatmulAlgo_t a = al;
                        uint32_t one = 1, none = CUBLASLT_REDUCTION_SCHEME_NONE;
                        cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &t, 4);
                        cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &sg, 4);
                        cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &cu, 4);
                        cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &sw, 4);
                        cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &one, 4);
                        cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &none, 4);
                        size_t ws;
                        if (check(a, &ws)) base.push_back({{a, ws}, splitk, redmask, 0.f});
                    }
    }
    qpass(base);
    for (int b = 0; b < (int)base.size() && b < nbase; b++) {
        if (!base[b].splitk || base[b].t >= 1e29f) continue;
        for (uint32_t k : {2u, 3u, 4u, 5u, 6u, 8u})
            for (uint32_t red : {(uint32_t)CUBLASLT_REDUCTION_SCHEME_INPLACE, (uint32_t)CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE,
                                 (uint32_t)CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE}) {
                if (!(base[b].redmask & red)) continue;
                cublasLtMatmulAlgo_t a = base[b].k.algo;
                cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &k, 4);
                cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, 4);
                size_t ws;
                if (check(a, &ws)) sk.push_back({{a, ws}, 0, 0, 0.f});
            }
    }
    qpass(sk);
    for (auto* cs : {&base, &sk})
        for (int i = 0; i < (int)cs->size() && i < keep; i++)
            if ((*cs)[i].t < 1e29f) lt_add(v, (*cs)[i].k.algo, (*cs)[i].k.ws);
}

// time the candidates for plan p's shape (see the comment above mm); al*: the alignments (bytes) of A, B, C
static GemmChoice tune_gemm(Ctx* c, GemmPlan* p, cublasOperation_t oa, cublasOperation_t ob, int M, int N, int K, float alpha,
                            const void* A, cudaDataType ta, int lda, long long sA, const void* B, cudaDataType tb, int ldb,
                            long long sB, float beta, cudaDataType tc, int ldc, long long sC, int batch,
                            cublasComputeType_t ct, int alA, int alB, int alC) {
    GemmChoice ch;
    ch.lt = 0; ch.ws = 0;
    memset(&ch.algo, 0, sizeof(ch.algo));
    static bool told = false;
    if (!told && g_tune >= 2) {
        told = true;
        fprintf(stderr, "carmt: timing cuBLAS GEMM algorithms for new shapes (once per GPU model and cuBLAS version%s%s)\n",
                g_cache_file.empty() ? "" : "; cached in ", g_cache_file.c_str());
    }
    size_t es = tc == CUDA_R_32F ? 4 : 2;
    size_t extent = ((size_t)(batch - 1) * sC + (size_t)(M - 1) * ldc + N) * es;
    void* Cs = ensure(&c->buf[B_TUNE], extent);       // the timing runs write here, never into C
    CK(cudaMemsetAsync(Cs, 0, extent, c->stream));
    void* ws = ensure(&c->buf[B_LTWS], LT_WS);
    auto run = [&](const LtCand* k) -> bool {         // k == NULL: cublasGemmEx's default
        if (!k) {
            if (batch == 1)
                return cublasGemmEx(c->blas, oa, ob, N, M, K, &alpha, B, tb, ldb, A, ta, lda, &beta, Cs, tc, ldc, ct,
                                    CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
            return cublasGemmStridedBatchedEx(c->blas, oa, ob, N, M, K, &alpha, B, tb, ldb, sB, A, ta, lda, sA, &beta, Cs, tc,
                                              ldc, sC, batch, ct, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
        }
        return cublasLtMatmul(c->lt, p->op, &alpha, B, p->la, A, p->lb, &beta, Cs, p->lc, Cs, p->lc, &k->algo, ws, k->ws,
                              c->stream) == CUBLAS_STATUS_SUCCESS;
    };
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto quick = [&](const LtCand& k) -> float {
        if (!run(&k)) { cudaGetLastError(); return 1e30f; }
        CK(cudaEventRecord(e0, c->stream));
        if (!run(&k)) { cudaGetLastError(); return 1e30f; }
        CK(cudaEventRecord(e1, c->stream));
        CK(cudaEventSynchronize(e1));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        return ms;
    };
    // candidates (cuBLASLt's A / B are our B / A)
    std::vector<LtCand> v;
    uint32_t aLA = (uint32_t)alB, aLB = (uint32_t)alA, aC = (uint32_t)alC;
    lt_heuristic(c, p, aLA, aLB, aC, v);
    if (aLA > 4 || aLB > 4 || aC > 4) lt_heuristic(c, p, aLA < 4 ? aLA : 4, aLB < 4 ? aLB : 4, aC < 4 ? aC : 4, v);
    double flop = 2.0 * M * N * K * batch;
    if (g_tune >= 2 && flop >= LT_EXH_FLOP) lt_exhaustive(c, p, ct, tb, ta, tc, aLA, aLB, aC, quick, v, 6, 6);
    // careful pass: index 0 = the default, i = v[i - 1]
    int nc = (int)v.size() + 1;
    std::vector<cudaEvent_t> eb(nc), ee(nc);
    for (int i = 0; i < nc; i++) { CK(cudaEventCreate(&eb[i])); CK(cudaEventCreate(&ee[i])); }
    std::vector<int> ord;
    for (int i = 0; i < nc; i++)
        if (run(i ? &v[i - 1] : NULL)) ord.push_back(i);   // untimed first call (module loading); failures drop out
    cudaGetLastError();
    for (int w = 0; w < 4; w++) run(NULL);                  // the clock settles
    std::vector<std::vector<float>> tt(nc);
    const int rounds = 11;
    for (int r = 0; r < rounds; r++) {
        shuffle(ord);
        for (int i : ord) { CK(cudaEventRecord(eb[i], c->stream)); run(i ? &v[i - 1] : NULL); CK(cudaEventRecord(ee[i], c->stream)); }
        CK(cudaStreamSynchronize(c->stream));
        for (int i : ord) { float ms = 0.f; CK(cudaEventElapsedTime(&ms, eb[i], ee[i])); tt[i].push_back(ms); }
    }
    cudaGetLastError();
    std::vector<float> med(nc, 1e30f);
    for (int i : ord) { std::sort(tt[i].begin(), tt[i].end()); med[i] = tt[i][tt[i].size() / 2]; }
    int best = 0;
    for (int i : ord)
        if (i && med[i] < med[best] * (best ? 1.f : 0.99f)) best = i;   // keep the default unless a candidate is faster by 1%
    for (int i = 0; i < nc; i++) { cudaEventDestroy(eb[i]); cudaEventDestroy(ee[i]); }
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    buf_free(&c->buf[B_TUNE]);
    if (best > 0) { ch.lt = 1; ch.algo = v[best - 1].algo; ch.ws = v[best - 1].ws; }
    if (g_tune_log)
        fprintf(stderr, "armt gemm %c%c M=%d N=%d K=%d batch=%d in=%d/%d out=%d beta=%g align=%d/%d/%d: %d candidates; default %.4f ms, chosen %.4f ms (%s)\n",
                oa == CUBLAS_OP_T ? 'T' : 'N', ob == CUBLAS_OP_T ? 'T' : 'N', M, N, K, batch, (int)ta, (int)tb, (int)tc,
                beta, alA, alB, alC, nc, med[0], med[best], best ? "lt" : "default");
    return ch;
}

static void mm(Ctx* c, int tA, int tB, int M, int N, int K, float alpha,
               const void* A, cudaDataType ta, int lda, long long sA,
               const void* B, cudaDataType tb, int ldb, long long sB, float beta,
               void* C, cudaDataType tc, int ldc, long long sC, int batch) {
    if (M <= 0 || N <= 0 || batch <= 0) return;
    if (K <= 0) { if (beta == 0.f && batch == 1 && ldc == N) CK(cudaMemsetAsync(C, 0, (size_t)M * N * (tc == CUDA_R_32F ? 4 : 2), c->stream)); return; }
    cublasComputeType_t ct = CUBLAS_COMPUTE_32F;
    if (ta == CUDA_R_32F && tb == CUDA_R_32F && c->d.c.allow_tf32) ct = CUBLAS_COMPUTE_32F_FAST_TF32;
    cublasOperation_t oa = tB ? CUBLAS_OP_T : CUBLAS_OP_N, ob = tA ? CUBLAS_OP_T : CUBLAS_OP_N;
    // broadcast operands (stride 0) stay on cublasGemmEx
    if (g_tune && !c->notune && (batch == 1 || (sA && sB && sC))) {
        int alA = ptr_align(A), alB = ptr_align(B), alC = ptr_align(C);
        char kb[256];
        snprintf(kb, sizeof(kb), "%d%d %d %d %d %d %d %d %lld %lld %lld %d %d %d %d %d %d/%d/%d %d", tA, tB, M, N, K, lda, ldb, ldc, sA, sB,
                 sC, batch, (int)ta, (int)tb, (int)tc, (int)ct, alA, alB, alC, beta != 0.f);
        std::string key(kb);
        GemmPlan* pl;
        {
            std::lock_guard<std::mutex> lk(g_plans_mu);
            auto it = g_plans.find(key);
            if (it != g_plans.end()) pl = it->second;
            else {
                pl = plan_desc(oa, ob, M, N, K, ta, lda, sA, tb, ldb, sB, tc, ldc, sC, batch, ct);
                // The file lock is only ever held for a read or an append, never across GPU work: the timing synchronizes
                // the training stream, which (with per-layer backward nodes) may wait on a collective that another rank
                // only launches once it gets past this lock.
                int fd = g_cache_file.empty() ? -1 : open(g_cache_file.c_str(), O_RDWR | O_CREAT, 0644);
                if (fd >= 0) { flock(fd, LOCK_EX); disk_read(fd); flock(fd, LOCK_UN); }   // tuned by another process?
                auto d = g_disk.find(key);
                GemmChoice ch;
                if (d != g_disk.end()) ch = d->second;
                else {
                    GemmChoice mine = tune_gemm(c, pl, oa, ob, M, N, K, alpha, A, ta, lda, sA, B, tb, ldb, sB, beta, tc, ldc,
                                                sC, batch, ct, alA, alB, alC);
                    if (fd >= 0) { flock(fd, LOCK_EX); disk_read(fd); }
                    d = g_disk.find(key);
                    if (d != g_disk.end()) ch = d->second;   // another process tuned it meanwhile: its (first) entry wins
                    else {
                        ch = mine;
                        g_disk.emplace(key, ch);
                        if (fd >= 0) disk_append(fd, key, ch);
                    }
                    if (fd >= 0) flock(fd, LOCK_UN);
                }
                if (fd >= 0) close(fd);
                pl->lt = ch.lt; pl->algo = ch.algo; pl->ws = ch.ws;
                g_plans[key] = pl;
            }
        }
        if (pl->lt) {
            void* ws = ensure(&c->buf[B_LTWS], LT_WS);
            CB(cublasLtMatmul(c->lt, pl->op, &alpha, B, pl->la, A, pl->lb, &beta, C, pl->lc, C, pl->lc, &pl->algo, ws, pl->ws,
                              c->stream));
            return;
        }
    }
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
// fp32 versions (memory projections). They read / write the framework's parameter and gradient tensors directly, whose
// alignment depends on the framework (DeepSpeed's flat buffers: 4 bytes); a tuned choice would then differ between, e.g.,
// DDP and ZeRO runs of the same model, so these small GEMMs always use cublasGemmEx.
static void lin32(Ctx* c, float* y, int ldy, const float* x, int ldx, const float* W, int N, int O, int I, float beta) {
    c->notune++;
    mm(c, 0, 1, N, O, I, 1.f, x, F32, ldx, 0, W, F32, I, 0, beta, y, F32, ldy, 0, 1);
    c->notune--;
}
static void lin32_dx(Ctx* c, float* dx, int lddx, const float* dy, int lddy, const float* W, int N, int O, int I, float beta) {
    c->notune++;
    mm(c, 0, 0, N, I, O, 1.f, dy, F32, lddy, 0, W, F32, I, 0, beta, dx, F32, lddx, 0, 1);
    c->notune--;
}
static void lin32_dw(Ctx* c, float* dW, const float* dy, int lddy, const float* x, int ldx, int N, int O, int I) {
    if (!dW) return;
    c->notune++;
    mm(c, 1, 0, O, I, N, 1.f, dy, F32, lddy, 0, x, F32, ldx, 0, 1.f, dW, F32, I, 0, 1);
    c->notune--;
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

// hybrid: rows of one piece [B, T1, D] <-> the real tokens [B, T, D] (rows start .. start + Tr) and its memory tokens
// [B, nw, Tm, D] (index w). to_x: x = src (memory rows without a source: 0); else dst = x * scale.
__global__ void k_hy_copy(float* x, float* real, float* mem, int T, int start, int Tr, int Tm, int nw, int w, int D, int to_x, float scale,
                          const float* gscale) {
    int row = blockIdx.x, T1 = Tr + Tm, b = row / T1, t = row % T1;
    float* xr = x + (long)row * D;
    float* o = t < Tr ? real + ((long)b * T + start + t) * D : (mem ? mem + (((long)b * nw + w) * Tm + (t - Tr)) * D : NULL);
    if (gscale) scale *= *gscale;
    if (to_x) { for (int d = threadIdx.x; d < D; d += blockDim.x) xr[d] = o ? o[d] : 0.f; }
    else if (o) { for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] = xr[d] * scale; }
}

// ---------------------------------------------------------------- rmsnorm
// xo = xa (+ xb) (+ num/den per memory head); h = xo * rsqrt(mean(xo^2)+eps) * w.  Row r of the output reads source row
// map(r) = (r / rpb_dst) * rpb_src + r % rpb_dst (to normalize only the real rows of a [B, T1] block).
__global__ void k_rmsnorm(float* __restrict__ xo, const float* __restrict__ xa, const floatX* __restrict__ xb, const float* __restrict__ num, const float* __restrict__ den, int Hm,
                          floatX* __restrict__ h, float* __restrict__ rstd, const float* __restrict__ w, int D, float eps, int rpb_dst, int rpb_src,
                          float woff) {
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
    for (int d = threadIdx.x; d < D; d += blockDim.x) stX(hr + d, xv[d] * rs * (w[d] + woff));
}
// dx[map(r)] += rstd*(dxh - xh*mean(dxh*xh)), dxh = dy*w, xh = x*rstd ; dw += sum_r dy*xh. grid-stride rows.
__global__ void k_rmsnorm_bwd(float* __restrict__ dx, const floatX* __restrict__ dy, const float* __restrict__ x, const float* __restrict__ rstd, const float* __restrict__ w, float* __restrict__ dw,
                              int R, int D, int rpb_dst, int rpb_src, float woff) {
    extern __shared__ float sdw[];
    for (int d = threadIdx.x; d < D; d += blockDim.x) sdw[d] = 0.f;
    for (int r = blockIdx.x; r < R; r += gridDim.x) {
        long src = (long)(r / rpb_dst) * rpb_src + r % rpb_dst;
        const float* xr = x + src * D;
        const floatX* dyr = dy + (long)r * D;
        float rs = rstd[r], acc = 0.f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) { float xh = xr[d] * rs; acc += toF(dyr[d]) * (w[d] + woff) * xh; }
        acc = block_sum(acc) / D;
        float* dxr = dx + src * D;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            float g = toF(dyr[d]), xh = xr[d] * rs;
            dxr[d] += rs * (g * (w[d] + woff) - xh * acc);
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
// dz[b,h,:] += sum_m w[row, h] * phi[row, h, :] over the rows of sample b. grid (B*Hm, row blocks of DZ_ROWS): partial sums
// added atomically (a block per (b, h) alone left the GPU nearly idle over 1000+ rows)
#define DZ_ROWS 64
__global__ void k_dz_acc(float* dz, const float* w, const float* phi, int rpb, int dkh, int Hm) {
    int b = blockIdx.x / Hm, h = blockIdx.x % Hm, m0 = blockIdx.y * DZ_ROWS, m1 = min(rpb, m0 + DZ_ROWS);
    for (int f = threadIdx.x; f < dkh; f += blockDim.x) {
        float acc = 0.f;
        for (int m = m0; m < m1; m++) { long row = (long)b * rpb + m; acc += w[row * Hm + h] * phi[row * dkh * Hm + h * dkh + f]; }
        atomicAdd(dz + ((long)b * Hm + h) * dkh + f, acc);
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
// memory rows of the residual output: xm[r] = a[mem row r] + b[r]  (b compact [B*Tm, D])
__global__ void k_resid_mem(float* xm, const float* a, const floatX* b, int Tr, int Tm, int D) {
    int r = blockIdx.x, bi = r / Tm, m = r % Tm;
    const float* s = a + ((long)bi * (Tr + Tm) + Tr + m) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) xm[(long)r * D + d] = s[d] + toF(b[(long)r * D + d]);
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
// SG (Qwen3.5): one beta and one gate input per head, at columns ob + h and oa + h of P (dt_bias per head); beta is both
// the b and the w of the GDN-2 recurrence and the gate is the same for all channels of the head.
template <int KD, bool SG>
__global__ void __launch_bounds__(256, 3) k_prep_fwd(const floatX* __restrict__ P, int NP, const floatX* __restrict__ fb,
                           const float* __restrict__ wq, const float* __restrict__ wk, const float* __restrict__ wv, const float* __restrict__ cs_in,
                           const float* __restrict__ A_log, const float* __restrict__ dtb,
                           floatS* __restrict__ cq, floatS* __restrict__ ck, float* __restrict__ cg, floatS* __restrict__ cb, floatS* __restrict__ cwv,
                           int B, int Tr, int Tm, int H, int nr, int nch, int W, float qscale, int ob, int oa) {
    // one warp per run of PRUN consecutive chunk positions of one (b, h); the conv inputs slide through registers (W = 4).
    // A block's 8 runs cover CH positions, so all its warps share one (b, h) (npos is a multiple of CH): the head's conv
    // weights sit in shared memory as [stream][tap][channel] (each lane's V channels of a tap: one vector load). The three
    // streams are done one after the other, which keeps few registers live (3 blocks / SM).
    constexpr int V = KD / 32, PRUN = 8, CW = 4;
    static_assert(PRUN * 8 == CH, "a block's runs cover one chunk");
    __shared__ __align__(16) float swt[3][CW][KD];
    int lane = threadIdx.x & 31;
    int npos = nch * CH;
    {
        long p0 = (long)blockIdx.x * (blockDim.x >> 5) * PRUN;
        int hb = (int)((p0 / npos) % H);
        for (int e = threadIdx.x; e < 3 * CW * KD; e += blockDim.x) {
            int p = e / (CW * KD), i = (e / KD) % CW, cc = e % KD;
            swt[p][i][cc] = (p == 0 ? wq : (p == 1 ? wk : wv))[(long)(hb * KD + cc) * CW + i];
        }
    }
    __syncthreads();
    long gw = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (gw * PRUN >= (long)B * H * npos) return;
    int pos0 = (gw * PRUN) % npos; long bh = (gw * PRUN) / npos; int h = bh % H, b = bh / H;
    int T1 = Tr + Tm, KDt = H * KD, Cc = 3 * KDt, ch = h * KD + lane * V;
    long rb = (long)b * T1, csb = (long)b * Cc;
    float win[3][CW - 1][V];   // inputs at t-3, t-2, t-1
    int tprev = -1000;
    float ea = EXPF(A_log[h]);
    float dtbv[V];
    if constexpr (!SG) ldv<V>(dtb + ch, dtbv);
    // conv + SiLU of stream p at row `row` into y; slides the window
    auto conv = [&](int p, long row, float* y) {
        float cur[V];
        ldv<V>(P + row * NP + p * KDt + ch, cur);
        float w[V];
        ldv<V>(&swt[p][CW - 1][lane * V], w);
#pragma unroll
        for (int j = 0; j < V; j++) y[j] = w[j] * cur[j];
#pragma unroll
        for (int s = 0; s < CW - 1; s++) {
            ldv<V>(&swt[p][s][lane * V], w);
#pragma unroll
            for (int j = 0; j < V; j++) y[j] += w[j] * win[p][s][j];
        }
#pragma unroll
        for (int j = 0; j < V; j++) y[j] = silu(y[j]);
#pragma unroll
        for (int s = 0; s < CW - 2; s++)
#pragma unroll
            for (int j = 0; j < V; j++) win[p][s][j] = win[p][s + 1][j];
#pragma unroll
        for (int j = 0; j < V; j++) win[p][CW - 2][j] = cur[j];
    };
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
        float y[V];
        conv(0, row, y);             // q: l2norm (x qscale)
        float sq = 0.f;
#pragma unroll
        for (int j = 0; j < V; j++) sq += y[j] * y[j];
        sq = warp_sum(sq);
        float iq = qscale / sqrtf(sq + 1e-6f);
#pragma unroll
        for (int j = 0; j < V; j++) y[j] *= iq;
        stv<V>(cq + o, y);
        conv(1, row, y);             // k: l2norm
        float sk = 0.f;
#pragma unroll
        for (int j = 0; j < V; j++) sk += y[j] * y[j];
        sk = warp_sum(sk);
        float ik = 1.f / sqrtf(sk + 1e-6f);
#pragma unroll
        for (int j = 0; j < V; j++) y[j] *= ik;
        stv<V>(ck + o, y);
        conv(2, row, y);             // v, then the gates
        float bb[V], g[V];
        if constexpr (SG) {
            float bs = sigm(toF(P[row * NP + ob + h])), gs = -ea * softplus(toF(P[row * NP + oa + h]) + dtb[h]);
#pragma unroll
            for (int j = 0; j < V; j++) { bb[j] = bs; y[j] *= bs; g[j] = gs; }
        } else {
            float ww[V], f[V];
            ldv<V>(P + row * NP + 3 * KDt + ch, bb);
            ldv<V>(P + row * NP + 4 * KDt + ch, ww);
            ldv<V>(fb + row * KDt + ch, f);
#pragma unroll
            for (int j = 0; j < V; j++) {
                bb[j] = sigm(bb[j]); y[j] *= sigm(ww[j]);
                g[j] = -ea * softplus(f[j] + dtbv[j]);
            }
        }
        stv<V>(cb + o, bb); stv<V>(cwv + o, y); stv<V>(cg + o, g);
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
template <int KD, bool SG>
__global__ void __launch_bounds__(256, 2) k_prep_bwd1(const floatX* __restrict__ P, int NP, const floatX* __restrict__ fb,
                            const float* __restrict__ wq, const float* __restrict__ wk, const float* __restrict__ wv, const float* __restrict__ cs_in,
                            const float* __restrict__ A_log, const float* __restrict__ dtb,
                            const floatS* __restrict__ dq, const floatS* __restrict__ dk, const floatS* __restrict__ db, const floatS* __restrict__ dg, const floatS* __restrict__ dwv,
                            floatS* __restrict__ dc, floatX* __restrict__ dP, floatX* __restrict__ df, float* __restrict__ dAlog, float* __restrict__ ddtb,
                            int B, int Tr, int Tm, int H, int nr, int nch, int W, float qscale, int rows_per_block, int ob, int oa) {
    // grid (H, row blocks); each warp takes rows_per_block / 8 consecutive rows, the conv inputs sliding through registers
    // (W = 4), the head's conv weights in shared memory as [stream][tap][channel]; the streams one after the other.
    constexpr int V = KD / 32, CW = 4;
    __shared__ __align__(16) float swt[3][CW][KD];
    int h = blockIdx.x, lane = threadIdx.x & 31, wl = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int T1 = Tr + Tm, KDt = H * KD, Cc = 3 * KDt, N = B * T1, ch = h * KD + lane * V;
    for (int e = threadIdx.x; e < 3 * CW * KD; e += blockDim.x) {
        int p = e / (CW * KD), i = (e / KD) % CW, cc = e % KD;
        swt[p][i][cc] = (p == 0 ? wq : (p == 1 ? wk : wv))[(long)(h * KD + cc) * CW + i];
    }
    __syncthreads();
    float acc_dtb[V], acc_A = 0.f;
#pragma unroll
    for (int j = 0; j < V; j++) acc_dtb[j] = 0.f;
    float ea = EXPF(A_log[h]);
    float dtbv[V];
    if constexpr (!SG) ldv<V>(dtb + ch, dtbv);
    int rpw = (rows_per_block + nw - 1) / nw;
    int r0 = blockIdx.y * rows_per_block, r1 = min(N, r0 + rows_per_block);
    int ra = r0 + wl * rpw, rz = min(r1, ra + rpw);
    float win[3][CW - 1][V];   // conv inputs at t-3, t-2, t-1
    int tprev = -1000;
    // pre-activation conv of stream p at (b, t): sum_i w[i] x(t - W + 1 + i), in conv_vec's order; slides the window
    auto conv = [&](int p, long rowi, float* a) {
        float cur[V], w[V];
        ldv<V>(P + rowi * NP + p * KDt + ch, cur);
        ldv<V>(&swt[p][0][lane * V], w);
#pragma unroll
        for (int j = 0; j < V; j++) a[j] = 0.f + w[j] * win[p][0][j];
#pragma unroll
        for (int s2 = 1; s2 < CW - 1; s2++) {
            ldv<V>(&swt[p][s2][lane * V], w);
#pragma unroll
            for (int j = 0; j < V; j++) a[j] += w[j] * win[p][s2][j];
        }
        ldv<V>(&swt[p][CW - 1][lane * V], w);
#pragma unroll
        for (int j = 0; j < V; j++) a[j] += w[j] * cur[j];
#pragma unroll
        for (int s2 = 0; s2 < CW - 2; s2++)
#pragma unroll
            for (int j = 0; j < V; j++) win[p][s2][j] = win[p][s2 + 1][j];
#pragma unroll
        for (int j = 0; j < V; j++) win[p][CW - 2][j] = cur[j];
    };
    for (int row = ra; row < rz; row++) {
        int b = row / T1, t = row % T1;
        long rb = (long)b * T1, csb = (long)b * Cc, ri = row;
        if (t != tprev + 1) {   // (re)load the window
#pragma unroll
            for (int p = 0; p < 3; p++)
#pragma unroll
                for (int s2 = 0; s2 < CW - 1; s2++) {
                    int tt = t - (CW - 1) + s2;
                    if (tt >= 0) ldv<V>(P + (rb + tt) * NP + p * KDt + ch, win[p][s2]);
                    else {
#pragma unroll
                        for (int j = 0; j < V; j++) win[p][s2][j] = cs_in ? cs_in[(csb + p * KDt + ch + j) * CW + CW + tt] : 0.f;
                    }
                }
        }
        tprev = t;
        long o = (((long)b * H + h) * nch * CH + pos_of_tok(t, Tr, nr)) * KD + lane * V;
        float cx[V], y[V], gx[V], o_dc[V];
#pragma unroll
        for (int sgi = 0; sgi < 2; sgi++) {   // q, k: through l2norm (q also x qscale)
            conv(sgi, ri, cx);
            float ss = 0.f;
#pragma unroll
            for (int j = 0; j < V; j++) { y[j] = silu(cx[j]); ss += y[j] * y[j]; }
            ss = warp_sum(ss);
            float nn = sqrtf(ss + 1e-6f), dot = 0.f;
            ldv<V>((sgi == 0 ? dq : dk) + o, gx);
#pragma unroll
            for (int j = 0; j < V; j++) { if (sgi == 0) gx[j] *= qscale; dot += y[j] / nn * gx[j]; }
            dot = warp_sum(dot);
#pragma unroll
            for (int j = 0; j < V; j++) o_dc[j] = (gx[j] - y[j] / nn * dot) / nn * dsilu(cx[j]);
            stv<V>(dc + ri * Cc + sgi * KDt + ch, o_dc);
        }
        conv(2, ri, cx);             // v, then the gates
        float gb[V], gg[V], gwv[V];
        ldv<V>(db + o, gb); ldv<V>(dg + o, gg); ldv<V>(dwv + o, gwv);
        if constexpr (SG) {
            // beta enters as b (with k) and as w (with v); the gate is shared by the head's channels: sum over channels
            float bs = sigm(toF(P[ri * NP + ob + h])), x = toF(P[ri * NP + oa + h]) + dtb[h], sb = 0.f, sg = 0.f;
#pragma unroll
            for (int j = 0; j < V; j++) {
                o_dc[j] = gwv[j] * bs * dsilu(cx[j]);
                sb += gb[j] + gwv[j] * silu(cx[j]);
                sg += gg[j];
            }
            stv<V>(dc + ri * Cc + 2 * KDt + ch, o_dc);
            acc_A += sg * (-ea * softplus(x));
            float dfs = sg * (-ea) * (x > 20.f ? 1.f : sigm(x));
            acc_dtb[0] += dfs;
            sb = warp_sum(sb); dfs = warp_sum(dfs);
            if (lane == 0) { stX(dP + ri * NP + ob + h, sb * bs * (1.f - bs)); stX(dP + ri * NP + oa + h, dfs); }
            continue;
        }
        float bb[V], ww[V], f[V], o_b[V], o_w[V], o_f[V];
        ldv<V>(P + ri * NP + 3 * KDt + ch, bb);
        ldv<V>(P + ri * NP + 4 * KDt + ch, ww);
        ldv<V>(fb + ri * KDt + ch, f);
#pragma unroll
        for (int j = 0; j < V; j++) {
            float bs = sigm(bb[j]), ws = sigm(ww[j]);
            o_dc[j] = gwv[j] * ws * dsilu(cx[j]);
            o_w[j] = gwv[j] * silu(cx[j]) * ws * (1.f - ws);
            o_b[j] = gb[j] * bs * (1.f - bs);
            float x = f[j] + dtbv[j];
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
    if constexpr (SG) {   // dt_bias per head: every lane's acc_dtb[0] is a partial sum (sred[w * KD + lane * V] holds it)
        if (threadIdx.x == 0) {
            float s = 0.f;
            for (int w = 0; w < nw; w++) for (int l = 0; l < 32; l++) s += sred[w * KD + l * V];
            atomicAdd(ddtb + h, s);
        }
        return;
    }
    for (int c = threadIdx.x; c < KD; c += blockDim.x) {
        float s = 0.f;
        for (int w = 0; w < nw; w++) s += sred[w * KD + c];
        atomicAdd(ddtb + h * KD + c, s);
    }
}
// backward 2: conv input grads. grid (Cc / 256, ceil(T1 / P2_TR), B): one thread per (2 channels, range of P2_TR tokens,
// sample) walks its tokens with a sliding window (W = 4). dX(t) = sum_i w[i] dc(t+W-1-i) (+ carried-state grad) ;
// dw[i] += sum_t dc(t) X(t-W+1+i) ; d conv state in (by the first range).
#define P2_TR 128
template <int W>
__global__ void __launch_bounds__(128) k_prep_bwd2(const floatX* __restrict__ P, int NP, const floatS* __restrict__ dc,
                                                   const float* __restrict__ convw, const float* __restrict__ cs_in,
                                                   const float* __restrict__ dcs_out, float* __restrict__ dcs_in, floatX* __restrict__ dP,
                                                   float* __restrict__ dconvw, int B, int T1, int Pc, int Cc) {
    int cc = 2 * (blockIdx.x * blockDim.x + threadIdx.x), b = blockIdx.z;
    int t0 = blockIdx.y * P2_TR, t1 = min(T1, t0 + P2_TR);
    if (cc >= Cc) return;
    float w[2][W], dw[2][W], g[2][W], x[2][W];   // g[k][j] = dc(t + j), x[k][j] = X(t - W + 1 + j) of channel cc + k
#pragma unroll
    for (int k = 0; k < 2; k++)
#pragma unroll
        for (int i = 0; i < W; i++) { w[k][i] = convw[(cc + k) * W + i]; dw[k][i] = 0.f; }
    long rb = (long)b * T1, sb0 = ((long)b * Cc + cc) * W;   // channel cc + k: conv state at sb0 + k * W
    auto DC = [&](int t, float* o) {
        if (t < T1) ldv<2>(dc + (rb + t) * Cc + cc, o);
        else { o[0] = 0.f; o[1] = 0.f; }
    };
    auto XX = [&](int t, float* o) {
        if (t >= 0) ldv<2>(P + (rb + t) * NP + cc, o);
        else {
#pragma unroll
            for (int k = 0; k < 2; k++) o[k] = cs_in ? cs_in[sb0 + k * W + W + t] : 0.f;
        }
    };
    // gradient of the incoming conv state: entry s holds x(s - W)
    if (dcs_in && blockIdx.y == 0) {
#pragma unroll
        for (int s = 0; s < W; s++) {
            int tp = s - W;
            float acc[2] = {0.f, 0.f}, v[2];
#pragma unroll
            for (int i = 0; i < W; i++) {
                int tt = tp + W - 1 - i;
                if (tt >= 0) { DC(tt, v); acc[0] += w[0][i] * v[0]; acc[1] += w[1][i] * v[1]; }
            }
            int j = tp - (Pc - W);
#pragma unroll
            for (int k = 0; k < 2; k++) {
                if (dcs_out && j >= 0 && j < W) acc[k] += dcs_out[sb0 + k * W + j];
                dcs_in[sb0 + k * W + s] = acc[k];
            }
        }
    }
#pragma unroll
    for (int j = 0; j < W; j++) {
        float v[2], u[2];
        DC(t0 + j, v); XX(t0 + j - W + 1, u);
#pragma unroll
        for (int k = 0; k < 2; k++) { g[k][j] = v[k]; x[k][j] = u[k]; }
    }
    for (int t = t0; t < t1; t++) {
        float acc[2];
#pragma unroll
        for (int k = 0; k < 2; k++) {
            acc[k] = 0.f;
#pragma unroll
            for (int i = 0; i < W; i++) acc[k] += w[k][i] * g[k][W - 1 - i];
        }
        int j = t - (Pc - W);
        if (dcs_out && j >= 0 && j < W) { acc[0] += dcs_out[sb0 + j]; acc[1] += dcs_out[sb0 + W + j]; }
        stv<2>(dP + (rb + t) * NP + cc, acc);
        float v[2], u[2];
        DC(t + W, v); XX(t + 1, u);
#pragma unroll
        for (int k = 0; k < 2; k++) {
#pragma unroll
            for (int i = 0; i < W; i++) dw[k][i] += g[k][0] * x[k][i];
#pragma unroll
            for (int i = 0; i < W - 1; i++) { g[k][i] = g[k][i + 1]; x[k][i] = x[k][i + 1]; }
            g[k][W - 1] = v[k];
            x[k][W - 1] = u[k];
        }
    }
#pragma unroll
    for (int k = 0; k < 2; k++)
#pragma unroll
        for (int i = 0; i < W; i++) atomicAdd(dconvw + (cc + k) * W + i, dw[k][i]);
}

// ---------------------------------------------------------------- gated output norm
template <int KD>
__global__ void k_onorm_fwd(const floatS* __restrict__ co, const floatX* __restrict__ gg, const float* __restrict__ g2b, const float* __restrict__ won, floatX* __restrict__ on, float* __restrict__ rstd,
                            int B, int Tr, int Tm, int H, int nr, int nch, float eps, int swish, int ldg) {
    // gg: the gate input, row stride ldg (VDt, or NP when it is read from the packed projection); g2b may be NULL
    constexpr int V = KD / 32;
    long gw = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31, T1 = Tr + Tm;
    if (gw >= (long)B * T1 * H) return;
    int h = gw % H; long row = gw / H; int b = row / T1, t = row % T1;
    long o = (((long)b * H + h) * nch * CH + pos_of_tok(t, Tr, nr)) * KD + lane * V;
    int VDt = H * KD, c0 = lane * V, ch = h * KD + c0;
    float x[V], g[V], y[V], ss = 0.f;
    ldv<V>(co + o, x);
    ldv<V>(gg + row * ldg + ch, g);
#pragma unroll
    for (int j = 0; j < V; j++) ss += x[j] * x[j];
    ss = warp_sum(ss);
    float rs = rsqrtf(ss / KD + eps);
    if (lane == 0) rstd[gw] = rs;
#pragma unroll
    for (int j = 0; j < V; j++) {
        float gv = g[j] + (g2b ? g2b[ch + j] : 0.f);
        float a = swish ? gv * sigm(gv) : sigm(gv);
        y[j] = x[j] * rs * won[c0 + j] * a;
    }
    stv<V>(on + row * VDt + ch, y);
}
// backward: grid (H, blocks over chunk positions B*nch*CH); writes do (zero on padding), dgg; dw_on, dg2b
template <int KD>
__global__ void k_onorm_bwd(const floatS* __restrict__ co, const floatX* __restrict__ gg, const float* __restrict__ g2b, const float* __restrict__ won, const floatX* __restrict__ don,
                            floatS* __restrict__ dco, floatX* __restrict__ dgg, float* __restrict__ dwon, float* __restrict__ dg2b,
                            int B, int Tr, int Tm, int H, int nr, int nch, float eps, int swish, int per_block, int ldg) {
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
        ldv<V>(co + o, x); ldv<V>(gg + row * ldg + ch, g); ldv<V>(don + row * VDt + ch, d);
#pragma unroll
        for (int j = 0; j < V; j++) ss += x[j] * x[j];
        ss = warp_sum(ss);
        float rs = rsqrtf(ss / KD + eps);
#pragma unroll
        for (int j = 0; j < V; j++) {
            float gv = g[j] + (g2b ? g2b[ch + j] : 0.f), s = sigm(gv);
            float a = swish ? gv * s : s, da = swish ? s * (1.f + gv * (1.f - s)) : s * (1.f - s);
            float xh = x[j] * rs;
            dxh[j] = d[j] * won[c0 + j] * a;
            dot += dxh[j] * xh;
            dgo[j] = d[j] * xh * won[c0 + j] * da;
            aw[j] += d[j] * xh * a;
            ab[j] += dgo[j];
        }
        stv<V>(dgg + row * ldg + ch, dgo);
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
#define LDT 20   // row stride (floats) of the forward's tile arrays: 16-byte rows for vector loads
#define K1_TILES (4 * CH * LDT + (int)((2 * 48 * LDO + 96 * LDO) * sizeof(floatS) / 4))
#define K1_POST (CH * LDL + 3 * SB * (SB + 1))
#define K1_SMEM (CH * LDL + (K1_TILES > K1_POST ? K1_TILES : K1_POST))
#ifdef INTRA_TIMING
__device__ long long* intra_timing;
__device__ long long* intra_timing_b;
#endif
template <int KD>
__global__ void __launch_bounds__(256, 2) k_intra_fwd(const floatS* __restrict__ cq, const floatS* __restrict__ ck,
                                                      const float* __restrict__ cg, const floatS* __restrict__ cb,
                                                      float* __restrict__ gc, floatS* __restrict__ y, floatS* __restrict__ Qg, floatS* __restrict__ Kt, float* __restrict__ Am, floatS* __restrict__ AmX, floatS* __restrict__ Mm,
                                                      float* __restrict__ gl) {
    extern __shared__ __align__(16) float sm[];
    float* sL = sm;                       // [CH][LDL]
    float* un = sL + CH * LDL;            // union: d-tiles | sM | (sA + sT)
    float* sq = un; float* sk = sq + CH * LDT; float* sbk = sk + CH * LDT; float* sgc = sbk + CH * LDT;   // [CH][LDT] tiles
    floatS* sQh = (floatS*)(sgc + CH * LDT);        // [48][LDO] rows 16..63: q_i e^{gc_i - gc_r(I)}
    floatS* sBh = sQh + 48 * LDO;                   // [48][LDO] same with b k
    floatS* sKr = sBh + 48 * LDO;                   // [96][LDO] k_j e^{gc_r(I) - gc_j}, rows 8 I (I - 1) + j (I = 1..3, j < 16 I)
    float* sM = un;                                 // [CH][LDL] after the tile loop
    float* sA = un; float* sT = sA + CH * LDL;      // after M is written out
    float* sgl = sL + 256;                          // [KD] gl, in sL during the tile loop (sL is filled only after it)
    const long cid = blockIdx.x, base = cid * CH * KD;
    const int tid = threadIdx.x;
#ifdef INTRA_TIMING   // development: clock64 per section, thread 0 of every block -> intra_timing[block][section]
    long long t_mark = clock64(), t_start = t_mark, t_scan = 0, t_load = 0, t_ops = 0, t_pair = 0;
#endif
    // 2. pairwise sums, over tiles of DT columns
    // diagonal 16x16 blocks, exact: their 4 x 136 (i, j <= i) pairs, 2 or 3 per thread (pairs tid, tid + 256, tid + 512)
    constexpr int NDP = 4 * 136;
    int pi_[3], pj_[3], np_ = 0;
    for (int p = tid; p < NDP; p += 256) {
        int Ib = p / 136, q = p % 136, ii = 0;
        while ((ii + 1) * (ii + 2) / 2 <= q) ii++;
        pi_[np_] = 16 * Ib + ii; pj_[np_] = 16 * Ib + q - ii * (ii + 1) / 2; np_++;
    }
    float dLa[3] = {}, dMa[3] = {};
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
    // a thread's elements of a tile: column tx, rows 4 tseg .. 4 tseg + 3; the inputs are loaded one tile ahead (registers)
    static_assert(CH == 64 && DT == 16, "the tile mapping assumes 64 rows x 16 columns over 256 threads");
    const int tx = tid & 15, tseg = tid >> 4;
    float* sTot = sL;                   // [16][16] row-group totals of the gc scan (in sL during the tile loop)
    float rq[4], rk[4], rb[4], rg[4];
    auto load_tile = [&](int d0) {
#pragma unroll
        for (int r = 0; r < 4; r++) {
            long g_ = base + (long)(4 * tseg + r) * KD + d0 + tx;
            rq[r] = toF(cq[g_]); rk[r] = toF(ck[g_]); rb[r] = toF(cb[g_]); rg[r] = cg[g_];
        }
    };
    load_tile(0);
    for (int d0 = 0; d0 < KD; d0 += DT) {
        // a. gc = cumsum(g) over the chunk rows: within the thread's 4 rows, then over the 16 row groups of the column
        float v[4], run = 0.f;
#pragma unroll
        for (int r = 0; r < 4; r++) {
            int c = 4 * tseg + r;
            run += rg[r]; v[r] = run;
            sq[c * LDT + tx] = rq[r]; sk[c * LDT + tx] = rk[r]; sbk[c * LDT + tx] = rb[r] * rk[r];
        }
        sTot[tseg * 16 + tx] = run;
        __syncthreads();
        float off = 0.f;
        for (int p = 0; p < tseg; p++) off += sTot[p * 16 + tx];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            int c = 4 * tseg + r;
            v[r] += off;
            sgc[c * LDT + tx] = v[r];
            if (gc) gc[base + (long)c * KD + d0 + tx] = v[r];
        }
        if (tseg == 15) { sgl[d0 + tx] = v[3]; gl[cid * KD + d0 + tx] = v[3]; }   // gl = the last row
        __syncthreads();
#ifdef INTRA_TIMING
        if (tid == 0) t_scan += clock64() - t_mark; t_mark = clock64();
#endif
        // b. y = b k e^gc, Qg = q e^gc, Kt = k e^{gl - gc}, from the thread's own registers
        {
            float glx = sgl[d0 + tx];
#pragma unroll
            for (int r = 0; r < 4; r++) {
                long g_ = base + (long)(4 * tseg + r) * KD + d0 + tx;
                float eg = EXPF(v[r]);
                stX(y + g_, rb[r] * rk[r] * eg); stX(Qg + g_, rq[r] * eg); stX(Kt + g_, rk[r] * EXPF(glx - v[r]));
            }
        }
        if (d0 + DT < KD) load_tile(d0 + DT);   // the next tile's loads go out (in flight during c and d)
#ifdef INTRA_TIMING
        if (tid == 0) t_load += clock64() - t_mark; t_mark = clock64();
#endif
        // c. decayed operands of the off-diagonal products (4 consecutive columns per item)
        for (int e = tid; e < 96 * (DT / 4); e += 256) {
            int row = e / (DT / 4), x0 = 4 * (e % (DT / 4));
            int Ir = row < 16 ? 1 : (row < 48 ? 2 : 3), j = row - 8 * Ir * (Ir - 1);
            float gr[4], gj[4], kj[4], o[4];
            ldv<4>(sgc + (16 * Ir - 1) * LDT + x0, gr); ldv<4>(sgc + j * LDT + x0, gj); ldv<4>(sk + j * LDT + x0, kj);
#pragma unroll
            for (int u = 0; u < 4; u++) o[u] = kj[u] * EXPF(gr[u] - gj[u]);
            stv<4>(sKr + row * LDO + x0, o);
            if (row < 48) {
                int ir = 16 + row, Ii = ir >> 4;
                float gi[4], g0[4], qi[4], bi[4], oq[4], ob[4];
                ldv<4>(sgc + ir * LDT + x0, gi); ldv<4>(sgc + (16 * Ii - 1) * LDT + x0, g0);
                ldv<4>(sq + ir * LDT + x0, qi); ldv<4>(sbk + ir * LDT + x0, bi);
#pragma unroll
                for (int u = 0; u < 4; u++) { float ef = EXPF(gi[u] - g0[u]); oq[u] = qi[u] * ef; ob[u] = bi[u] * ef; }
                stv<4>(sQh + row * LDO + x0, oq); stv<4>(sBh + row * LDO + x0, ob);
            }
        }
        __syncthreads();
#ifdef INTRA_TIMING
        if (tid == 0) t_ops += clock64() - t_mark; t_mark = clock64();
#endif
        // d. diagonal blocks (exact pairwise decays) and the off-diagonal products
#pragma unroll
        for (int n = 0; n < 3; n++) {
            if (n < np_) {
                const float* gi = sgc + pi_[n] * LDT; const float* qi = sq + pi_[n] * LDT; const float* bi = sbk + pi_[n] * LDT;
                const float* gj = sgc + pj_[n] * LDT; const float* kj = sk + pj_[n] * LDT;
#pragma unroll
                for (int x0 = 0; x0 < DT; x0 += 4) {
                    float a[4], b[4], k4[4], q4[4], bb[4];
                    ldv<4>(gi + x0, a); ldv<4>(gj + x0, b); ldv<4>(kj + x0, k4); ldv<4>(qi + x0, q4); ldv<4>(bi + x0, bb);
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        float k = k4[u] * EXPF(a[u] - b[u]);
                        dLa[n] += bb[u] * k; dMa[n] += q4[u] * k;
                    }
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
#ifdef INTRA_TIMING
        if (tid == 0) t_pair += clock64() - t_mark; t_mark = clock64();
#endif
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
    for (int n = 0; n < 3; n++) {
        if (n < np_) {
            if (pj_[n] < pi_[n]) sL[pi_[n] * LDL + pj_[n]] = dLa[n];
            sM[pi_[n] * LDL + pj_[n]] = dMa[n];
        }
    }
    __syncthreads();
    floatS* Mo = Mm + cid * CH * CH;
    for (int e = tid; e < CH * CH; e += 256) stX(Mo + e, sM[(e / CH) * LDL + e % CH]);
    __syncthreads();
    for (int e = tid; e < CH * LDL; e += 256) sA[e] = 0.f;
    __syncthreads();
#ifdef INTRA_TIMING
    long long t_asm = clock64() - t_mark; t_mark = clock64();
#endif
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
#ifdef INTRA_TIMING
    long long t_inv1 = clock64() - t_mark; t_mark = clock64();
#endif
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
#ifdef INTRA_TIMING
    long long t_inv2 = clock64() - t_mark; t_mark = clock64();
#endif
    for (int e = tid; e < CH * CH; e += 256) {   // A: fp32 for the backward (Am != NULL), the GEMMs' floatS copy
        float v = sA[(e / CH) * LDL + e % CH];
        if (Am) Am[cid * CH * CH + e] = v;
        if (AmX) stX(AmX + cid * CH * CH + e, v);
    }
#ifdef INTRA_TIMING
    __syncthreads();
    if (tid == 0 && intra_timing) {
        long long* o = intra_timing + cid * 8;
        o[0] = t_scan; o[1] = t_load; o[2] = t_ops; o[3] = t_pair; o[4] = t_asm; o[5] = t_inv1; o[6] = t_inv2; o[7] = clock64() - t_start;
    }
#endif
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
// grid (KD/TVS, B*H): the KD/TVS blocks of one (b, h) are consecutive, so they run together and share the chunk loads in L2.
// block 256 (8 warps), 2 blocks per SM. smem: sS f32 [KD][TVS] | sSb bf16 [KD][TVS+8], aliased by sV f32 [CH][TVS] once the
// product that reads sSb is done | sWK bf16 [CH][KD+8] | sVb bf16 [CH][TVS+8]
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
// bytes of the sSb / sV union
template <int KD> __host__ __device__ constexpr int tc_ub() { return KD * TC_LDV * 2 > CH * TVS * 4 ? KD * TC_LDV * 2 : CH * TVS * 4; }
template <int KD> __host__ __device__ constexpr int tc_smem() { return KD * TVS * 4 + tc_ub<KD>() + CH * (KD + 8) * 2 + CH * TC_LDV * 2 + TC_SPLIT_SMEM(KD); }
static_assert((CH / 16) * (TVS / 16) == 8, "one 16x16 tile of the 64 x TVS products per warp");
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
// register prefetch of a [CH][KD] bf16 chunk (NL uint4 per thread), and its store into smem (row stride KD+8, optionally
// negated): the state passes load chunk n+1 while computing chunk n
template <int KD> static __device__ __forceinline__ void tc_ld_regs(uint4* r, const bf16* src) {
    constexpr int R8 = KD / 8, NL = CH * R8 / 256;
#pragma unroll
    for (int i = 0; i < NL; i++) { int e = threadIdx.x + 256 * i; r[i] = ((const uint4*)(src + (long)(e / R8) * KD))[e % R8]; }
}
template <int KD> static __device__ __forceinline__ void tc_st_regs(bf16* dst, const uint4* r, bool neg) {
    constexpr int R8 = KD / 8, NL = CH * R8 / 256;
#pragma unroll
    for (int i = 0; i < NL; i++) {
        int e = threadIdx.x + 256 * i;
        uint4 v = r[i];
        if (neg) {
            __nv_bfloat162* h = (__nv_bfloat162*)&v;
#pragma unroll
            for (int j = 0; j < 4; j++) h[j] = __hneg2(h[j]);
        }
        *(uint4*)(dst + (e / R8) * (KD + 8) + (e % R8) * 8) = v;
    }
}
template <int KD>
__global__ void __launch_bounds__(256, 2) k_state_fwd(const bf16* __restrict__ u, const bf16* __restrict__ Wm, const bf16* __restrict__ Kt,
                                                      const float* __restrict__ gl, bf16* __restrict__ S0, bf16* __restrict__ S0lo,
                                                      bf16* __restrict__ vn,
                                                      const float* __restrict__ S_in, float* __restrict__ S_out, int ncarry, int nch) {
    // with ARMT_SPLIT_STATE the state enters the products as hi + lo (two bf16 parts, ~fp32 accuracy)
    extern __shared__ __align__(16) unsigned char smraw[];
    constexpr int LDK = KD + 8, VD = KD;
    float* sS = (float*)smraw; bf16* sSb = (bf16*)(sS + KD * TVS); bf16* sWK = (bf16*)((unsigned char*)sSb + tc_ub<KD>());
    float* sV = (float*)sSb; bf16* sVb = sWK + CH * LDK; bf16* sSl = sVb + CH * TC_LDV;
    const long bh = blockIdx.y;
    const int v0 = blockIdx.x * TVS, tid = threadIdx.x, warp = tid >> 5;
    for (int e = tid; e < KD * TVS; e += 256) { int k = e / TVS, vv = e % TVS; sS[e] = S_in ? S_in[bh * KD * VD + (long)k * VD + v0 + vv] : 0.f; }
    constexpr int NL = CH * (KD / 8) / 256, NU = CH * TVS / 256;
    uint4 rW[NL], rK[NL];   // the chunk's Wm, Kt and u, loaded one chunk ahead
    bf16 ru[NU];
    auto ld_u = [&](long cb_) {
#pragma unroll
        for (int j = 0; j < NU; j++) { int e = tid + 256 * j, c = e / TVS, vv = e % TVS; ru[j] = u[cb_ * CH * VD + (long)c * VD + v0 + vv]; }
    };
    if (nch > 0) { tc_ld_regs<KD>(rW, Wm + bh * nch * CH * KD); tc_ld_regs<KD>(rK, Kt + bh * nch * CH * KD); ld_u(bh * nch); }
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
        tc_st_regs<KD>(sWK, rW, true);
        if (n + 1 < nch) tc_ld_regs<KD>(rW, Wm + (cb + 1) * CH * KD);
        __syncthreads();
        {   // vn = u - Wm S: the product into a fragment (one tile per warp), stored over sSb once every warp is done with it
            int m = warp / (TVS / 16), nn = warp % (TVS / 16);
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
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
            __syncthreads();
            wmma::store_matrix_sync(sV + 16 * m * TVS + 16 * nn, acc, TVS, wmma::mem_row_major);
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NU; j++) {
            int e = tid + 256 * j, c = e / TVS, vv = e % TVS;
            bf16 x = __float2bfloat16(sV[e] + __bfloat162float(ru[j]));
            sVb[c * TC_LDV + vv] = x;
            vn[cb * CH * VD + (long)c * VD + v0 + vv] = x;
        }
        if (n + 1 < nch) ld_u(cb + 1);
        for (int e = tid; e < KD * TVS; e += 256) sS[e] *= EXPF(gl[cb * KD + (e / TVS)]);
        tc_st_regs<KD>(sWK, rK, false);
        if (n + 1 < nch) tc_ld_regs<KD>(rK, Kt + (cb + 1) * CH * KD);
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
__global__ void __launch_bounds__(256, 2) k_state_bwd(const bf16* __restrict__ pre_dvn, bf16* dvn, bf16* dSe,
                                                      const bf16* __restrict__ Kt, const bf16* __restrict__ Wm, const float* __restrict__ gl,
                                                      const float* dS_out, int ncarry, float* dS_in, int nch) {
    extern __shared__ __align__(16) unsigned char smraw[];
    constexpr int LDK = KD + 8, VD = KD, PR = KD * TVS / 256;
    float* sS = (float*)smraw; bf16* sSb = (bf16*)(sS + KD * TVS); bf16* sWK = (bf16*)((unsigned char*)sSb + tc_ub<KD>());
    float* sV = (float*)sSb; bf16* sVb = sWK + CH * LDK;
    const long bh = blockIdx.y;
    const int v0 = blockIdx.x * TVS, tid = threadIdx.x, warp = tid >> 5;
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
        __syncthreads();
        {   // dvn = pre_dvn + Kt dS: the product into a fragment, stored over sSb once every warp is done with it
            int m = warp / (TVS / 16), nn = warp % (TVS / 16);
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
#pragma unroll
            for (int kk = 0; kk < KD / 16; kk++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
                wmma::load_matrix_sync(a, sWK + 16 * m * LDK + 16 * kk, LDK);
                wmma::load_matrix_sync(b, sSb + 16 * kk * TC_LDV + 16 * nn, TC_LDV);
                wmma::mma_sync(acc, a, b, acc);
            }
            __syncthreads();
            wmma::store_matrix_sync(sV + 16 * m * TVS + 16 * nn, acc, TVS, wmma::mem_row_major);
        }
        __syncthreads();
        for (int e = tid; e < CH * TVS; e += 256) {
            int c = e / TVS, vv = e % TVS;
            float x = sV[e] + __bfloat162float(pre_dvn[cb * CH * VD + (long)c * VD + v0 + vv]);
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
#define LDB 16   // row stride of the backward's tile arrays: unpadded, every warp access below is row-contiguous
#define B1_SMEM_BYTES (2 * CH * B1_LD * 2 + 2 * 4 * 16 * 17 * 4 + 5 * CH * LDB * 4 + 4 * CH * 20 * 4 + KD_MAX_B1 * 4)
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
    float* sq = sdMd + 4 * 16 * 17; float* sk = sq + CH * LDB; float* sbk = sk + CH * LDB; float* sgc = sbk + CH * LDB;
    float* sdgc = sgc + CH * LDB;
    float* sOut = sdgc + CH * LDB;                 // [4][CH][20]: dqM_off, dbkL_off (rows 16..), dkL_off, dkM_off (rows ..47)
    bf16* sKr = (bf16*)sOut;                            // operands alias sOut (used before the products are stored)
    bf16* sQs = sKr + 96 * LDO; bf16* sBs = sQs + 96 * LDO;
    float* sdgl = sOut + 4 * CH * 20;
    const int VD = KD;
    const long cid = blockIdx.x, base = cid * CH * KD;
    const int tid = threadIdx.x, lane = tid & 31, wl = tid >> 5;
#ifdef INTRA_TIMING
    long long t_mark = clock64(), t_start = t_mark, tb[8] = {};
#define TB(k) if (tid == 0) { long long t_ = clock64(); tb[k] += t_ - t_mark; t_mark = t_; }
#else
#define TB(k)
#endif
    for (int d = wl; d < KD; d += 16) {     // dgl = e^{gl} rowsum(S0 dSe) + colsum(dKt Kt)
        const floatS* s0 = S0 + cid * KD * VD + (long)d * VD;
        const floatS* ds = dSe + cid * KD * VD + (long)d * VD;
        float a = 0.f;
        for (int v = lane * 2; v < VD; v += 64) { float x[2], z[2]; ldv<2>(s0 + v, x); ldv<2>(ds + v, z); a += x[0] * z[0] + x[1] * z[1]; }
        a = warp_sum(a);
        if (lane == 0) sdgl[d] = EXPF(gl[cid * KD + d]) * a;
    }
    __syncthreads();
    TB(0)
    if (tid < KD) {   // (runs alongside the staging below)
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
    TB(1)
    const int warp = wl;
    // jobs (k-steps): 0-3: col J=0 dkL, col J=0 dkM, row I=3 dqM, row I=3 dbkL (3) ; 4-7: J=1 / I=2 (2) ; 8-11: J=2 / I=1 (1)
    for (int d0 = 0; d0 < KD; d0 += DT) {
        {   // tile loads: thread -> (row c, columns x0, x0 + 1)
            static_assert(CH * DT == 2 * 512, "two tile elements per thread");
            int c = tid >> 3, x0 = 2 * (tid & 7);
            long g_ = base + c * KD + d0 + x0;
            float q2[2], k2[2], b2[2], g2[2];
            ldv<2>(cq + g_, q2); ldv<2>(ck + g_, k2); ldv<2>(cb + g_, b2); ldv<2>(gc + g_, g2);
            float bk2[2] = {b2[0] * k2[0], b2[1] * k2[1]};
            stv<2>(sq + c * LDB + x0, q2); stv<2>(sk + c * LDB + x0, k2); stv<2>(sbk + c * LDB + x0, bk2); stv<2>(sgc + c * LDB + x0, g2);
        }
        __syncthreads();
        TB(2)
        for (int e = tid; e < 96 * DT; e += 512) {
            int row = e / DT, x = e % DT;
            int Ir = row < 16 ? 1 : (row < 48 ? 2 : 3), j = row - 8 * Ir * (Ir - 1);
            stX(sKr + row * LDO + x, sk[j * LDB + x] * EXPF(sgc[(16 * Ir - 1) * LDB + x] - sgc[j * LDB + x]));
            int Jc = row < 48 ? 0 : (row < 80 ? 1 : 2), i = 16 * (Jc + 1) + row - (Jc == 0 ? 0 : (Jc == 1 ? 48 : 80));
            float ef = EXPF(sgc[i * LDB + x] - sgc[(16 * Jc + 15) * LDB + x]);
            stX(sQs + row * LDO + x, sq[i * LDB + x] * ef);
            stX(sBs + row * LDO + x, sbk[i * LDB + x] * ef);
        }
        __syncthreads();
        TB(3)
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
        TB(4)
        {   // per-element outputs: thread -> (row c, columns dd0, dd0 + 1); paired loads / stores, the old arithmetic order
            int c = tid >> 3, dd0 = 2 * (tid & 7), B_ = c >> 4, cc = c & 15;
            float gci[2], dqM[2] = {}, dbkL[2] = {}, dkL[2] = {}, dkM[2] = {};
            ldv<2>(sgc + c * LDB + dd0, gci);
            if (B_ > 0) {
                float g0[2], o0[2], o1[2];
                ldv<2>(sgc + (16 * B_ - 1) * LDB + dd0, g0); ldv<2>(sOut + 0 * CH * 20 + c * 20 + dd0, o0); ldv<2>(sOut + 1 * CH * 20 + c * 20 + dd0, o1);
#pragma unroll
                for (int u = 0; u < 2; u++) { float f = EXPF(gci[u] - g0[u]); dqM[u] = o0[u] * f; dbkL[u] = o1[u] * f; }
            }
            if (B_ < 3) {
                float g1[2], o2[2], o3[2];
                ldv<2>(sgc + (16 * B_ + 15) * LDB + dd0, g1); ldv<2>(sOut + 2 * CH * 20 + c * 20 + dd0, o2); ldv<2>(sOut + 3 * CH * 20 + c * 20 + dd0, o3);
#pragma unroll
                for (int u = 0; u < 2; u++) { float f = EXPF(g1[u] - gci[u]); dkL[u] = o2[u] * f; dkM[u] = o3[u] * f; }
            }
            const float* Ld = sdLd + B_ * 272; const float* Md = sdMd + B_ * 272;
            for (int jj = 0; jj <= cc; jj++) {          // diagonal block, row side
                int j = 16 * B_ + jj;
                float kj[2], gj[2];
                ldv<2>(sk + j * LDB + dd0, kj); ldv<2>(sgc + j * LDB + dd0, gj);
                float md = Md[cc * 17 + jj], ld = Ld[cc * 17 + jj];
#pragma unroll
                for (int u = 0; u < 2; u++) { float k = kj[u] * EXPF(gci[u] - gj[u]); dqM[u] += md * k; dbkL[u] += ld * k; }
            }
            for (int ii = cc; ii < 16; ii++) {          // diagonal block, column side
                int i = 16 * B_ + ii;
                float gi[2], bi[2], qi[2];
                ldv<2>(sgc + i * LDB + dd0, gi); ldv<2>(sbk + i * LDB + dd0, bi); ldv<2>(sq + i * LDB + dd0, qi);
                float ld = Ld[ii * 17 + cc], md = Md[ii * 17 + cc];
#pragma unroll
                for (int u = 0; u < 2; u++) { float e2 = EXPF(gi[u] - gci[u]); dkL[u] += ld * bi[u] * e2; dkM[u] += md * qi[u] * e2; }
            }
            long g_ = base + c * KD + d0 + dd0;
            float q2[2], k2[2], bk2[2], b2[2], dQ2[2], dK2[2], dY2[2], Qg2[2], Kt2[2], y2[2], gl2[2];
            ldv<2>(sq + c * LDB + dd0, q2); ldv<2>(sk + c * LDB + dd0, k2); ldv<2>(sbk + c * LDB + dd0, bk2);
            ldv<2>(cb + g_, b2); ldv<2>(dQg + g_, dQ2); ldv<2>(dKt + g_, dK2); ldv<2>(dy + g_, dY2);
            ldv<2>(Qg + g_, Qg2); ldv<2>(Kt + g_, Kt2); ldv<2>(y + g_, y2); ldv<2>(gl + cid * KD + d0 + dd0, gl2);
            float oq[2], ok[2], ob[2], odg[2];
#pragma unroll
            for (int u = 0; u < 2; u++) {
                float eg = EXPF(gci[u]);
                oq[u] = dqM[u] + dQ2[u] * eg;
                ok[u] = dkL[u] + dkM[u] + dK2[u] * EXPF(gl2[u] - gci[u]) + dY2[u] * b2[u] * eg + dbkL[u] * b2[u];
                ob[u] = dY2[u] * k2[u] * eg + dbkL[u] * k2[u];
                odg[u] = bk2[u] * dbkL[u] - k2[u] * dkL[u] + q2[u] * dqM[u] - k2[u] * dkM[u] + dQ2[u] * Qg2[u] - dK2[u] * Kt2[u] + dY2[u] * y2[u];
                if (c == CH - 1) odg[u] += sdgl[d0 + dd0 + u];
            }
            stv<2>(dq + g_, oq); stv<2>(dk + g_, ok); stv<2>(db + g_, ob); stv<2>(sdgc + c * LDB + dd0, odg);
        }
        __syncthreads();
        TB(5)
        if (tid < DT) {   // dg = reverse cumsum of dgc over the chunk rows
            float s_ = 0.f;
            for (int c = CH - 1; c >= 0; c--) { s_ += sdgc[c * LDB + tid]; stX(dg + base + c * KD + d0 + tid, s_); }
        }
        __syncthreads();
        TB(6)
    }
#ifdef INTRA_TIMING
    if (tid == 0 && intra_timing_b) {
        long long* o = intra_timing_b + cid * 8;
        for (int k = 0; k < 7; k++) o[k] = tb[k];
        o[7] = clock64() - t_start;
    }
#endif
#undef TB
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
    CK(cudaFuncSetAttribute(k_state_fwd<KD>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));   // 2 blocks per SM
    CK(cudaFuncSetAttribute(k_state_bwd<KD>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CK(cudaFuncSetAttribute(k_intra_bwd<KD>, cudaFuncAttributeMaxDynamicSharedMemorySize, B1_SMEM_BYTES));
}
#define SC CUDA_S
// chunk-major scan inputs come from k_prep_fwd (B_CQ, B_CK, B_CB, B_CWV: floatS ; B_CG: fp32)
// keep: also produce what only the backward reads (gc, the fp32 A); a forward-only call (training forward, inference) skips it
template <int KD>
static void scan_fwd(Ctx* c, const Step& s, const float* S_in, float* S_out, int ncarry, bool keep) {
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
    prof_mark(c, PR_INTRA);
#ifdef SCAN32
    float* Akeep = Am;              // the GEMMs read the fp32 A
#else
    float* Akeep = keep ? Am : NULL;
#endif
    k_intra_fwd<KD><<<nb, 256, K1_SMEM * 4, c->stream>>>(cq, ck, cg, cb, keep ? gc : NULL, y, Qg, Kt, Akeep, AmW, Mm, gl);
    prof_mark(c, PR_SG_A);
    mm(c, 0, 0, CH, KD, CH, 1.f, AmX, SC, CH, sC, cwv, SC, KD, sK, 0.f, u, SC, KD, sK, nb);     // u = A wv
    mm(c, 0, 0, CH, KD, CH, 1.f, AmX, SC, CH, sC, y, SC, KD, sK, 0.f, Wm, SC, KD, sK, nb);      // Wm = A y
#ifdef SCAN32
    dim3 g((unsigned)(s.B * c->d.H), KD / 32);
#else
    dim3 g(KD / TVS, (unsigned)(s.B * c->d.H));
#endif
#ifdef SCAN32
    k_state_fwd<KD><<<g, 256, ST_SMEM(KD), c->stream>>>(u, Wm, Kt, gl, S0, vn, S_in, S_out, ncarry, s.nch);
    mm(c, 0, 0, CH, KD, KD, 1.f, Qg, SC, KD, sK, S0, SC, KD, sS, 0.f, o, SC, KD, sK, nb);       // o = Qg S0
#elif !defined(ARMT_SPLIT_STATE)
    prof_mark(c, PR_STATE);
    k_state_fwd<KD><<<g, 256, ST_SMEM(KD), c->stream>>>(u, Wm, Kt, gl, S0, NULL, vn, S_in, S_out, ncarry, s.nch);
    prof_mark(c, PR_SG_O);
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
    prof_mark(c, PR_SG_PRE);
    mm(c, 1, 0, CH, KD, CH, 1.f, Mm, SC, CH, sC, dO, SC, KD, sK, 0.f, pre_dvn, SC, KD, sK, nb);  // pre_dvn = M^T do
    mm(c, 1, 0, KD, KD, CH, 1.f, Qg, SC, KD, sK, dO, SC, KD, sK, 0.f, dSe, SC, KD, sS, nb);      // pre_dS = Qg^T do
#ifdef SCAN32
    dim3 g((unsigned)(s.B * c->d.H), KD / 32);
#else
    dim3 g(KD / TVS, (unsigned)(s.B * c->d.H));
#endif
    prof_mark(c, PR_STATE);
    k_state_bwd<KD><<<g, 256, ST_SMEM(KD), c->stream>>>(pre_dvn, dvn, dSe, Kt, Wm, gl, dS_out, ncarry, dS_in, s.nch);
    prof_mark(c, PR_SG_POST);
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
    prof_mark(c, PR_INTRA);
    k_intra_bwd<KD><<<nb, b1_threads, B1_SMEM_BYTES, c->stream>>>(cq, ck, cb, gc, Qg, Kt, y, dL, dM, dQg, dKt, dy, S0, dSe, gl,
                                                                                 dq, dk, db, dg);
}

// =============================================================================================== host: one layer
// forward of layer li over one segment piece. x: [B, T1, D] input (fp32), xo: output. st_in: state before the piece;
// st_out: carried state (may be NULL to discard; may alias st_in for S, Wm, z). write: memory tokens present.
// recomp: the backward's recompute: only what layer_bwd reads is produced (xo is not written; the MLP output is computed
// on the memory-token rows only, for the memory write's scratch; the new memory itself is not formed).
template <int KD>
static void layer_fwd(Ctx* c, int li, const Step& s, const float* x, float* xo, const LState* st_in, const LState* st_out,
                      bool recomp = false) {
    Dims& d = c->d;
    int D = d.D, N = s.N, NP = d.NP, KDt = d.KDt, VDt = d.VDt, Hv = d.Hv, F = d.F;
    float eps = d.c.norm_eps;
    bool mem = d.has_mem[li];
    c->prof_pass = recomp ? 1 : 0;
    float *xr = BF(c, B_XR, float, (long)N * D), *rstd1 = BF(c, B_RSTD1, float, N);
    floatX* h1 = BF(c, B_H1, floatX, (long)N * D);
    // 1. memory read + norm_1
    prof_mark(c, mem ? PR_MEMREAD : PR_NORM1);
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
        prof_mark(c, PR_NORM1);
        k_rmsnorm<<<N, 256, 0, c->stream>>>(xr, x, NULL, numq, denq, d.Hm, h1, rstd1, PP(c, PI(li, L_N1)), D, eps, 1, 1, d.woff);
    } else {
        CK(cudaMemcpyAsync(xr, x, (size_t)N * D * 4, cudaMemcpyDeviceToDevice, c->stream));
        k_rmsnorm<<<N, 256, 0, c->stream>>>(NULL, x, NULL, NULL, NULL, 1, h1, rstd1, PP(c, PI(li, L_N1)), D, eps, 1, 1, d.woff);
    }
    // 2. projections
    floatX *P = BF(c, B_P, floatX, (long)N * NP), *fb = BF(c, B_FB, floatX, (long)N * KDt), *gg = BF(c, B_GG, floatX, (long)N * VDt);
    prof_mark(c, PR_PROJ);
    lin(c, P, NP, h1, D, c->w_in[li], N, NP, D, 0.f);
    int of1 = 3 * KDt + 2 * VDt, og1 = of1 + Hv;
    prof_mark(c, PR_GATES);
    if (!d.sg) {   // GDN-2: low-rank gate input and output gate (Qwen3.5: a and z come straight out of P)
        lin(c, fb, KDt, P + of1, NP, c->w_f2[li], N, KDt, Hv, 0.f);
        lin(c, gg, VDt, P + og1, NP, c->w_g2[li], N, VDt, Hv, 0.f);
    }
    // 3. conv / gates -> chunk-major scan inputs ; carried conv state
    long CSK = s.nchB * CH * KD;
    floatS *cq = BF(c, B_CQ, floatS, CSK), *ck = BF(c, B_CK, floatS, CSK), *cb = BF(c, B_CB, floatS, CSK), *cwv = BF(c, B_CWV, floatS, CSK);
    float* cg = BF(c, B_CG, float, CSK);
    // q, k, v conv weights (GDN-2: separate tensors, contiguous only in the grad layout; Qwen3.5: one tensor)
    const float* wq = PP(c, PI(li, L_QC));
    const float* wk = d.sg ? wq + (long)KDt * d.W : PP(c, PI(li, L_KC));
    const float* wv = d.sg ? wq + 2L * KDt * d.W : PP(c, PI(li, L_VC));
    if (d.W != 4) armt_fail("conv_size must be 4");
    long nwarps = s.nchB * CH / 8;   // runs of 8 positions (CH is a multiple of 8)
    unsigned npb = (unsigned)((nwarps + 7) / 8);
    prof_mark(c, PR_PREP);
    if (d.sg)
        k_prep_fwd<KD, true><<<npb, 256, 0, c->stream>>>(P, NP, NULL, wq, wk, wv, st_in->cs, PP(c, PI(li, L_ALOG)), PP(c, PI(li, L_DTB)),
                                                          cq, ck, cg, cb, cwv, s.B, s.Tr, s.Tm, d.H, s.nr, s.nch, d.W, 1.f / sqrtf((float)KD),
                                                          d.ob, d.oa);
    else
        k_prep_fwd<KD, false><<<npb, 256, 0, c->stream>>>(P, NP, fb, wq, wk, wv, st_in->cs, PP(c, PI(li, L_ALOG)), PP(c, PI(li, L_DTB)),
                                                           cq, ck, cg, cb, cwv, s.B, s.Tr, s.Tm, d.H, s.nr, s.nch, d.W, 1.f / sqrtf((float)KD),
                                                           0, 0);
    int Pc = (s.Tm == 0 || d.c.mem_in_stream) ? s.T1 : s.Tr;
    int ncarry = (s.Tm == 0 || d.c.mem_in_stream) ? s.nch : s.nr;
    prof_mark(c, PR_CONVST);
    if (st_out && st_out->cs) {
        float* tmp = BF(c, B_CSTMP, float, (long)s.B * d.Cc * d.W);
        LAUNCH1D(k_conv_state, (long)s.B * d.Cc * d.W, tmp, st_in->cs, P, NP, s.B, s.T1, Pc, d.Cc, d.W);
        CK(cudaMemcpyAsync(st_out->cs, tmp, (size_t)s.B * d.Cc * d.W * 4, cudaMemcpyDeviceToDevice, c->stream));
    }
    // 4. scan
    scan_fwd<KD>(c, s, st_in->S, st_out ? st_out->S : NULL, ncarry, recomp);
    // 5. gated norm, o_proj, residual + norm_2
    floatX *on = BF(c, B_ON, floatX, (long)N * VDt), *ao = BF(c, B_AO, floatX, (long)N * D);
    float* rstdo = BF(c, B_RSTDO, float, (long)N * d.H);
    long nw2 = (long)N * d.H;
    prof_mark(c, PR_ONORM);
    k_onorm_fwd<KD><<<(unsigned)((nw2 + 7) / 8), 256, 0, c->stream>>>(BF(c, B_O, floatS, CSK), d.sg ? P + d.oz : gg,
                                                                     d.sg ? NULL : PP(c, PI(li, L_G2B)), PP(c, PI(li, L_ON)), on, rstdo,
                                                                     s.B, s.Tr, s.Tm, d.H, s.nr, s.nch, eps, d.c.gate_swish, d.sg ? NP : VDt);
    prof_mark(c, PR_OPROJ);
    lin(c, ao, D, on, VDt, c->w_o[li], N, D, VDt, 0.f);
    float *xmid = BF(c, B_XMID, float, (long)N * D), *rstd2 = BF(c, B_RSTD2, float, N);
    floatX* h2 = BF(c, B_H2, floatX, (long)N * D);
    prof_mark(c, PR_NORM2);
    k_rmsnorm<<<N, 256, 0, c->stream>>>(xmid, xr, ao, NULL, NULL, 1, h2, rstd2, PP(c, PI(li, L_N2)), D, eps, 1, 1, d.woff);
    // 6. MLP
    floatX *a = BF(c, B_A, floatX, (long)N * 2 * F), *m = BF(c, B_MM, floatX, (long)N * F), *mo = BF(c, B_MO, floatX, (long)N * D);
    prof_mark(c, PR_MLP);
    lin(c, a, 2 * F, h2, D, c->w_12[li], N, 2 * F, D, 0.f);
    LAUNCH1D(k_swiglu, (long)N * F, m, a, (long)N, F);
    bool write = mem && s.Tm > 0 && st_out && st_out->Wm;
    if (!recomp) {
        lin(c, mo, D, m, F, c->w_3[li], N, D, F, 0.f);
        LAUNCH1D(k_resid, (long)N * D, xo, xmid, mo, (long)N * D);
    }
    // 7. memory write from the memory tokens' outputs
    prof_mark(c, PR_MEMWRITE);
    if (write) {
        int BM = s.BM, dkh = d.dk / d.Hm, dvh = D / d.Hm;
        float *xm = BF(c, B_XM, float, (long)BM * D), *mk = BF(c, B_MK, float, (long)BM * d.dm), *phik = BF(c, B_PHIK, float, (long)BM * d.dk);
        float *denk = BF(c, B_DENK, float, (long)BM * d.Hm), *nrmk = BF(c, B_NRMK, float, (long)BM * d.Hm), *nmv = BF(c, B_NMV, float, (long)BM * D);
        float *mbp = BF(c, B_MBP, float, (long)BM * d.nmb), *prevn = BF(c, B_PREVN, float, (long)BM * D), *E = BF(c, B_E, float, (long)BM * D);
        float* coef = BF(c, B_COEF, float, (long)BM * d.Hm);
        if (!recomp) k_gather_mem<<<BM, 256, 0, c->stream>>>(xm, xo, s.Tr, s.Tm, D);
        else {   // w3 on the memory rows only (per sample: rows [Tr, T1)), into a compact [BM, D] block
            mm(c, 0, 1, s.Tm, D, F, 1.f, m + (long)s.Tr * F, CUDA_X, F, (long long)s.T1 * F, c->w_3[li], CUDA_X, F, 0, 0.f,
               mo, CUDA_X, D, (long long)s.Tm * D, s.B);
            k_resid_mem<<<BM, 256, 0, c->stream>>>(xm, xmid, mo, s.Tr, s.Tm, D);
        }
        lin32(c, mk, d.dm, xm, D, PP(c, PI(li, L_MK)), BM, d.dm, D, 0.f);
        lin32(c, nmv, D, xm, D, PP(c, PI(li, L_MV)), BM, D, D, 0.f);
        lin32(c, mbp, d.nmb, xm, D, PP(c, PI(li, L_MB)), BM, d.nmb, D, 0.f);
        k_dpfp<<<(BM * d.Hm + 7) / 8, 256, 0, c->stream>>>(mk, d.dm, phik, denk, nrmk, st_in->z, BM, s.Tm, d.dm, d.Hm);
        for (int h = 0; h < d.Hm; h++)
            mm(c, 0, 0, s.Tm, dvh, dkh, 1.f, phik + h * dkh, F32, d.dk, (long long)s.Tm * d.dk, st_in->Wm + (long)h * dkh * dvh, F32, dvh,
               (long long)d.dk * D / d.Hm, 0.f, prevn + h * dvh, F32, D, (long long)s.Tm * D, s.B);
        k_write_E<<<BM, 256, 0, c->stream>>>(E, nmv, prevn, denk, mbp, PP(c, PI(li, L_MBB)), D, d.Hm, d.nmb);
        if (recomp) {   // layer_bwd needs coef (k_write_z) but not the new W
            k_write_z<<<s.B * d.Hm, 256, 0, c->stream>>>(st_in->z, st_out->z, phik, denk, nrmk, coef, s.Tm, dkh, d.Hm);
            return;
        }
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
    c->prof_pass = 2;
    // ---- memory write backward
    prof_mark(c, PR_MEMWRITE);
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
        k_dz_acc<<<dim3(s.B * d.Hm, (s.Tm + DZ_ROWS - 1) / DZ_ROWS), 256, 0, c->stream>>>(dst->z, ddenk, phik, s.Tm, dkh, d.Hm);
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
    prof_mark(c, PR_MLP);
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
    prof_mark(c, PR_NORM2);
    k_rmsnorm_bwd<<<rb_blocks, 256, D * 4, c->stream>>>(dx, dh2, xmid, rstd2, PP(c, PI(li, L_N2)), GP(c, PI(li, L_N2)), N, D, 1, 1, d.woff);
    // ---- attention backward
    prof_mark(c, PR_OPROJ);
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
    prof_mark(c, PR_ONORM);
    {
        int per = 512;
        dim3 g(d.H, (unsigned)(((long)s.B * s.nch * CH + per - 1) / per));
        float* gwon = GP(c, PI(li, L_ON)); float* gg2b = GP(c, PI(li, L_G2B));
        float* junk = BF(c, B_ZERO, float, 2 * (long)d.VDt + 64);
        // Qwen3.5: the gate input is z, columns [oz, oz + VDt) of P: its gradient goes straight into dP
        k_onorm_bwd<KD><<<g, 256, 0, c->stream>>>(BF(c, B_O, floatS, CSK), d.sg ? P + d.oz : gg, d.sg ? NULL : PP(c, PI(li, L_G2B)),
                                                 PP(c, PI(li, L_ON)), don, dO, d.sg ? dP + d.oz : dgg, gwon ? gwon : junk,
                                                 (!d.sg && gg2b) ? gg2b : junk, s.B, s.Tr, s.Tm, d.H, s.nr, s.nch, eps, d.c.gate_swish, per,
                                                 d.sg ? NP : VDt);
    }
    int of1 = 3 * KDt + 2 * VDt, og1 = of1 + Hv;
    prof_mark(c, PR_GATES);
    if (!d.sg) {
        lin_dx(c, dP + og1, NP, dgg, VDt, c->w_g2[li], N, VDt, Hv, 0.f);
        lin_dw(c, GP(c, PI(li, L_G2)), dgg, VDt, P + og1, NP, N, VDt, Hv);
    }
    int Pc = (s.Tm == 0 || d.c.mem_in_stream) ? s.T1 : s.Tr;
    int ncarry = (s.Tm == 0 || d.c.mem_in_stream) ? s.nch : s.nr;
    scan_bwd<KD>(c, s, dst->S, dst->S, ncarry);
    floatS* dc = BF(c, B_DC, floatS, (long)N * d.Cc);
    prof_mark(c, PR_PREP1);
    {
        int rpb = 64;
        dim3 g(d.H, (N + rpb - 1) / rpb);
        float* gA = GP(c, PI(li, L_ALOG)); float* gdt = GP(c, PI(li, L_DTB));
        float* junk = BF(c, B_ZERO, float, 2 * (long)d.VDt + 64);
        const float* wq = PP(c, PI(li, L_QC));
        const float* wk = d.sg ? wq + (long)KDt * d.W : PP(c, PI(li, L_KC));
        const float* wv = d.sg ? wq + 2L * KDt * d.W : PP(c, PI(li, L_VC));
        floatS *gq = BF(c, B_DQ, floatS, CSK), *gk = BF(c, B_DK, floatS, CSK), *gb = BF(c, B_DB, floatS, CSK), *gG = BF(c, B_DG, floatS, CSK);
        floatS* gwv = BF(c, B_DWV, floatS, CSK);
        if (d.sg)
            k_prep_bwd1<KD, true><<<g, 256, 0, c->stream>>>(P, NP, NULL, wq, wk, wv, st_in->cs, PP(c, PI(li, L_ALOG)), PP(c, PI(li, L_DTB)),
                                                             gq, gk, gb, gG, gwv, dc, dP, NULL, gA ? gA : junk, gdt ? gdt : junk,
                                                             s.B, s.Tr, s.Tm, d.H, s.nr, s.nch, d.W, 1.f / sqrtf((float)KD), rpb, d.ob, d.oa);
        else
            k_prep_bwd1<KD, false><<<g, 256, 0, c->stream>>>(P, NP, fb, wq, wk, wv, st_in->cs, PP(c, PI(li, L_ALOG)), PP(c, PI(li, L_DTB)),
                                                              gq, gk, gb, gG, gwv, dc, dP, df, gA ? gA : junk, gdt ? gdt : junk,
                                                              s.B, s.Tr, s.Tm, d.H, s.nr, s.nch, d.W, 1.f / sqrtf((float)KD), rpb, 0, 0);
    }
    prof_mark(c, PR_GATES);
    if (!d.sg) {
        lin_dx(c, dP + of1, NP, df, KDt, c->w_f2[li], N, KDt, Hv, 0.f);
        lin_dw(c, GP(c, PI(li, L_F2)), df, KDt, P + of1, NP, N, KDt, Hv);
    }
    prof_mark(c, PR_PREP2);
    {
        // conv weights: q, k, v live in separate tensors; their grads are contiguous (L_QC, L_KC, L_VC adjacent in the layout).
        // the kernel wants one [Cc][W] weight array: stage a contiguous copy
        const float* wcat = PP(c, PI(li, L_QC));   // Qwen3.5: already one tensor
        if (!d.sg) {
            float* wc = BF(c, B_WCAT, float, (long)d.Cc * d.W);
            CK(cudaMemcpyAsync(wc, PP(c, PI(li, L_QC)), (size_t)KDt * d.W * 4, cudaMemcpyDeviceToDevice, c->stream));
            CK(cudaMemcpyAsync(wc + (long)KDt * d.W, PP(c, PI(li, L_KC)), (size_t)KDt * d.W * 4, cudaMemcpyDeviceToDevice, c->stream));
            CK(cudaMemcpyAsync(wc + 2L * KDt * d.W, PP(c, PI(li, L_VC)), (size_t)VDt * d.W * 4, cudaMemcpyDeviceToDevice, c->stream));
            wcat = wc;
        }
        float* dcs_tmp = BF(c, B_DCSTMP, float, (long)s.B * d.Cc * d.W);
        float* gconv = GPG(c, PI(li, L_QC), PI(li, L_VC));
        float* junk = gconv ? gconv : BF(c, B_CSTMP, float, (long)s.B * d.Cc * d.W + (long)d.Cc * d.W);
        if (d.W != 4) armt_fail("conv_size must be 4");
        dim3 g((d.Cc / 2 + 127) / 128, (s.T1 + P2_TR - 1) / P2_TR, s.B);
        k_prep_bwd2<4><<<g, 128, 0, c->stream>>>(P, NP, dc, wcat, st_in->cs, dst->cs, dcs_tmp, dP, junk, s.B, s.T1, Pc, d.Cc);
        if (dst->cs) CK(cudaMemcpyAsync(dst->cs, dcs_tmp, (size_t)s.B * d.Cc * d.W * 4, cudaMemcpyDeviceToDevice, c->stream));
    }
    prof_mark(c, PR_PROJ);
    lin_dx(c, dh1, D, dP, NP, c->w_in[li], N, NP, D, 0.f);
    lin_dw(c, GPG(c, PI(li, L_Q), PI(li, L_G1)), dP, NP, h1, D, N, NP, D);
    float *xr = BF(c, B_XR, float, (long)N * D), *rstd1 = BF(c, B_RSTD1, float, N);
    prof_mark(c, PR_NORM1);
    k_rmsnorm_bwd<<<rb_blocks, 256, D * 4, c->stream>>>(dx, dh1, xr, rstd1, PP(c, PI(li, L_N1)), GP(c, PI(li, L_N1)), N, D, 1, 1, d.woff);
    // ---- memory read backward
    prof_mark(c, PR_MEMREAD);
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
        k_dz_acc<<<dim3(s.B * d.Hm, (s.T1 + DZ_ROWS - 1) / DZ_ROWS), 256, 0, c->stream>>>(dst->z, dden, phiq, s.T1, dkh, d.Hm);
        k_dpfp_bwd<<<(N * d.Hm + 7) / 8, 256, 8 * dkh * 4, c->stream>>>(mq, d.dm, dphi, dmq, d.dm, N, d.dm, d.Hm);
        lin32_dx(c, dx, D, dmq, d.dm, PP(c, PI(li, L_MQ)), N, d.dm, D, 1.f);
        lin32_dw(c, GP(c, PI(li, L_MQ)), dmq, d.dm, x, D, N, d.dm, D);
    }
    (void)xo;
}

// =============================================================================================== host: weights
static void pack_weights(Ctx* c) {
    Dims& d = c->d;
    // packed floatX buffer: per layer w_in [NP x D] (= the parameters L_Q .. L_G1 in order), f2, g2, o, w12, w3 ; lm_head.
    // Empty slots (Qwen3.5) and attention layers (no parameters here) take no space.
    const int maxn = 32 * MAXL;
    const float* src[maxn]; floatX* dst[maxn]; long n[maxn];
    int cnt = 0;
    long tot = (long)d.V * d.D;
    for (int i = 0; i < d.L; i++)
        for (int j = L_Q; j <= L_W3; j++) if (j != L_QC && j != L_KC && j != L_VC && j != L_ALOG && j != L_DTB && j != L_G2B && j != L_ON && j != L_N2)
            tot += d.numel[PI(i, j)];
    floatX* p = BF(c, B_PACK, floatX, tot);
    auto add = [&](int idx) { if (d.numel[idx]) { src[cnt] = PP(c, idx); dst[cnt] = p; n[cnt] = d.numel[idx]; p += n[cnt]; cnt++; } };
    for (int i = 0; i < d.L; i++) {
        c->w_in[i] = p;
        for (int j = L_Q; j <= L_G1; j++) add(PI(i, j));
        c->w_f2[i] = p; add(PI(i, L_F2));
        c->w_g2[i] = p; add(PI(i, L_G2));
        c->w_o[i] = p; add(PI(i, L_O));
        c->w_12[i] = p; add(PI(i, L_W1)); add(PI(i, L_W2));
        c->w_3[i] = p; add(PI(i, L_W3));
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
    c->prof_pass = 0;
    prof_mark(c, PR_HEAD);
    floatX* hf = BF(c, B_HF, floatX, (long)R * D);
    float* rstdf = BF(c, B_RSTDF, float, R);
    k_rmsnorm<<<R, 256, 0, c->stream>>>(NULL, x, NULL, NULL, NULL, 1, hf, rstdf, PP(c, P_NORMF), D, d.c.norm_eps, s.Tr, s.T1, d.woff);
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
        // rows per logits chunk: 4096, fewer for a large vocabulary (the chunk buffer stays <= 512 MB: 1024 rows at 248k)
        int LC = (int)(((256L << 20) / V) & ~255L);
        LC = LC > 4096 ? 4096 : (LC < 256 ? 256 : LC);
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
        if (dxo) k_rmsnorm_bwd<<<2 * c->sms, 256, D * 4, c->stream>>>(dxo, dhf, x, rstdf, PP(c, P_NORMF), GP(c, P_NORMF), R, D, s.Tr, s.T1, d.woff);
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
    if (cfg->kind == 1 && (2 * cfg->n_heads) % (cfg->head_dim / 32)) armt_fail("Qwen3.5 GDN: n_heads must be even for head_dim 128");
    Ctx* c = (Ctx*)calloc(1, sizeof(Ctx));
    make_dims(&c->d, cfg);
    CB(cublasCreate(&c->blas));
    CB(cublasLtCreate(&c->lt));
    gemm_tune_init();
    int dev; CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&c->sms, cudaDevAttrMultiProcessorCount, dev));
    CK(cudaMallocHost((void**)&c->htab, 32 * MAXL * 24));
    CK(cudaEventCreateWithFlags(&c->pack_done, cudaEventDisableTiming));
    c->gmask = (uint8_t*)calloc(c->d.n_params, 1);
    { const char* pv = getenv("ARMT_PROF"); if (pv && atoi(pv)) c->prof = new Prof(); }
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
    cublasLtDestroy(c->lt);
    cudaFreeHost(c->htab);
    cudaEventDestroy(c->pack_done);
    free(c->tr_Tr); free(c->tr_Tm); free(c->tr_start); free(c->gmask); free(c->hy_w); free(c->hy_args);
    if (c->prof) { for (cudaEvent_t e : c->prof->ev) cudaEventDestroy(e); delete c->prof; }
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
            c->prof_pass = 0; prof_mark(c, PR_EMBED);
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
        c->prof_pass = 0; prof_mark(c, PR_EMBED);
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

// backward of layers [lo, hi) (top-down, each layer's segments right to left), then, if lo == 0, of the embeddings.
// Called for hi = L first and then for consecutive lower ranges; every layer's gradients are final (and scaled by
// *gscale) when its range returns, so the caller can start reducing them while the lower layers run.
template <int KD>
static void backward_impl(Ctx* c, const int64_t* ids, const float* gscale, int lo, int hi) {
    Dims& d = c->d;
    if (!c->tr_grad || !c->buf[B_XS].p) armt_fail("backward without a recorded training forward");
    c->grad = c->tr_grad;
    int B = c->tr_B, T = c->tr_T, np = c->tr_nseg, D = d.D;
    long xs_per = (long)B * c->tr_T1max * D;
    long nS = (long)B * d.H * KD * KD, nCs = (long)B * d.Cc * d.W, nW = (long)B * d.dk * D / d.Hm, nZ = (long)B * d.dk;
    float *XS = (float*)c->buf[B_XS].p, *DXS = (float*)c->buf[B_DXS].p;
    float *SS = (float*)c->buf[B_ST_S].p, *SCS = (float*)c->buf[B_ST_CS].p, *SW = (float*)c->buf[B_ST_W].p, *SZ = (float*)c->buf[B_ST_Z].p;
    float *dS = BF(c, B_DST_S, float, nS), *dCs = BF(c, B_DST_CS, float, nCs), *dW = BF(c, B_DST_W, float, nW), *dZ = BF(c, B_DST_Z, float, nZ);
    for (int i = hi - 1; i >= lo; i--) {
        c->prof_pass = 2; prof_mark(c, PR_COPY);
        zero(c, dS, nS * 4); zero(c, dCs, nCs * 4); zero(c, dW, nW * 4); zero(c, dZ, nZ * 4);
        for (int k = np - 1; k >= 0; k--) {
            Step s = make_step(c, B, c->tr_Tr[k], c->tr_Tm[k]);
            long si = (long)i * np + k;
            float* x0 = XS + xs_per * ((long)k * (d.L + 1));
            LState in = {SS + nS * si, SCS + nCs * si, d.has_mem[i] ? SW + nW * si : NULL, d.has_mem[i] ? SZ + nZ * si : NULL};
            float* xo_tmp = BF(c, B_XOT, float, xs_per);
            LState disc = {NULL, NULL, d.has_mem[i] ? BF(c, B_WDISC, float, nW) : NULL, d.has_mem[i] ? BF(c, B_ZDISC, float, nZ) : NULL};
            layer_fwd<KD>(c, i, s, x0 + xs_per * i, xo_tmp, &in, &disc, true);
            float* dx = DXS + xs_per * k;
            LState dst = {dS, dCs, d.has_mem[i] ? dW : NULL, d.has_mem[i] ? dZ : NULL};
            layer_bwd<KD>(c, i, s, x0 + xs_per * i, x0 + xs_per * (i + 1), &in, dx, &dst);
            if (c->tr_reset && k > 0) {
                LAUNCH1D(k_mask_zero, nS, dS, c->tr_reset + (long)k * B, (long)d.H * KD * KD, B);
                LAUNCH1D(k_mask_zero, nCs, dCs, c->tr_reset + (long)k * B, (long)d.Cc * d.W, B);
            }
        }
    }
    long g0 = d.offset[PI(lo, 0)], g1 = hi == d.L ? d.total : d.offset[PI(hi, 0)];
    if (gscale && g1 > g0) LAUNCH1D(k_scale, g1 - g0, c->grad + g0, g1 - g0, gscale);
    if (lo > 0) return;
    for (int k = 0; k < np; k++) {
        Step s = make_step(c, B, c->tr_Tr[k], c->tr_Tm[k]);
        c->prof_pass = 2; prof_mark(c, PR_EMBED);
        k_embed_bwd<<<s.N, 256, 0, c->stream>>>(DXS + xs_per * k, ids, T, c->tr_start[k], GP(c, P_EMB), GP(c, P_MEMTOK), s.Tr, s.Tm, D);
    }
    if (gscale && g0 > 0) LAUNCH1D(k_scale, g0, c->grad, g0, gscale);   // the global parameters (embeddings, head, final norm)
}

// =============================================================================================== hybrid (layer-major)
// A hybrid backbone (recurrent + attention layers): the attention layers run outside the library (in PyTorch), so a call
// goes layer by layer over all its segment pieces:
//   armt_hy_begin(args)       pieces at absolute segment boundaries, weight packing, embeddings (= input of layer 0)
//   armt_hy_run(lo, hi, ..)   the recurrent layers [lo, hi) over all pieces. The hidden streams go in / out in the format of
//                             the attention layers: real tokens [B, T, D] and memory tokens [B, nw, M, D] (fp32; nw = the
//                             pieces that end a segment and write the memory)
//   armt_hy_head(real)        final norm, logits / token losses / loss (training: also the loss gradient, kept)
// and, after a training call, its backward in the reverse order:
//   armt_hy_head_bwd(gscale, d_real)   the head's parameter grads times *gscale; d_real = *gscale * dloss / dreal
//   armt_hy_run_bwd(lo, hi, ..)        the recurrent layers hi-1 .. lo; lo == 0: also the embeddings, which ends it.
// Training stores every layer's input per piece (as armt_forward); inference keeps two O(T) hidden buffers (ping-pong) and
// carries the recurrent / memory states as armt_forward does.
static ArmtForwardArgs* hya(Ctx* c) { return (ArmtForwardArgs*)c->hy_args; }
static long hy_xs_per(Ctx* c) { return (long)c->tr_B * c->tr_T1max * c->d.D; }
static float* hx(Ctx* c, int k, int i) {   // input of layer i (= output of layer i - 1) of piece k
    long xs_per = hy_xs_per(c);
    if (c->hy_train) return (float*)c->buf[B_XS].p + xs_per * ((long)k * (c->d.L + 1) + i);
    return (float*)c->buf[i % 2 ? B_X1 : B_X0].p + xs_per * k;
}
static void hy_copy(Ctx* c, int k, float* x, const float* real, const float* mem, int to_x, const float* gscale) {
    Step s = make_step(c, c->tr_B, c->tr_Tr[k], c->tr_Tm[k]);
    if (!to_x && !real && !mem) return;
    k_hy_copy<<<s.N, 256, 0, c->stream>>>(x, (float*)real, (float*)mem, c->tr_T, c->tr_start[k], s.Tr, s.Tm, c->hy_nw, c->hy_w[k], c->d.D,
                                          to_x, 1.f, gscale);
}

template <int KD>
static void hy_begin_impl(Ctx* c, const ArmtForwardArgs* a, int* pieces, int cap, int* np_out) {
    Dims& d = c->d;
    int B = a->B, T = a->T, S = d.c.segment_size, D = d.D;
    if (a->train && (a->position != 0 || a->state)) armt_fail("a training forward must start from an empty state");
    std::vector<int> pst, pen, ptm;
    for (int st = 0, pos = a->position; st < T;) {
        int en = st + S - pos < T ? st + S - pos : T;
        int done = pos + en - st == S;
        pst.push_back(st); pen.push_back(en); ptm.push_back(done && d.M > 0 ? d.M : 0);
        pos = done ? 0 : pos + en - st;
        st = en;
    }
    int np = (int)pst.size();
    if (a->train) ptm[np - 1] = 0;   // the last memory write feeds nothing that is trained
    if (np > cap) armt_fail("more pieces than the caller's buffer");
    if (np > c->tr_cap) {
        c->tr_cap = np;
        c->tr_Tr = (int*)realloc(c->tr_Tr, np * sizeof(int)); c->tr_Tm = (int*)realloc(c->tr_Tm, np * sizeof(int));
        c->tr_start = (int*)realloc(c->tr_start, np * sizeof(int)); c->hy_w = (int*)realloc(c->hy_w, np * sizeof(int));
    }
    if (!c->hy_args) c->hy_args = calloc(1, sizeof(ArmtForwardArgs));
    memcpy(c->hy_args, a, sizeof(ArmtForwardArgs));
    int T1max = 0, nw = 0;
    for (int k = 0; k < np; k++) {
        c->tr_Tr[k] = pen[k] - pst[k]; c->tr_Tm[k] = ptm[k]; c->tr_start[k] = pst[k];
        c->hy_w[k] = ptm[k] ? nw++ : -1;
        if (c->tr_Tr[k] + ptm[k] > T1max) T1max = c->tr_Tr[k] + ptm[k];
        pieces[3 * k] = pst[k]; pieces[3 * k + 1] = c->tr_Tr[k]; pieces[3 * k + 2] = ptm[k];
    }
    *np_out = np;
    c->tr_B = B; c->tr_T = T; c->tr_nseg = np; c->tr_T1max = T1max; c->tr_reset = a->reset;
    c->hy_np = np; c->hy_nw = nw; c->hy_train = a->train;
    c->grad = a->train ? a->grad : NULL;
    if (a->train) c->tr_grad = a->grad;
    if (a->repack || !c->w_lm) pack_weights(c);
    long xs_per = hy_xs_per(c);
    long nS = (long)B * d.H * KD * KD, nCs = (long)B * d.Cc * d.W, nW = (long)B * d.dk * D / d.Hm, nZ = (long)B * d.dk;
    if (a->train) {
        BF(c, B_XS, float, xs_per * (d.L + 1) * np);
        BF(c, B_DXS, float, xs_per * np);
        float *SS = BF(c, B_ST_S, float, nS * d.L * np), *SCS = BF(c, B_ST_CS, float, nCs * d.L * np);
        float *SW = BF(c, B_ST_W, float, nW * d.L * np), *SZ = BF(c, B_ST_Z, float, nZ * d.L * np);
        for (int i = 0; i < d.L; i++) {   // initial (k = 0) state of every recurrent layer
            if (d.c.attn[i]) continue;
            long si = (long)i * np;
            zero(c, SS + nS * si, nS * 4); zero(c, SCS + nCs * si, nCs * 4); zero(c, SW + nW * si, nW * 4); zero(c, SZ + nZ * si, nZ * 4);
        }
    } else {
        BF(c, B_X0, float, xs_per * np); BF(c, B_X1, float, xs_per * np);
        long nzb = nS > nW ? (nS > nCs ? nS : nCs) : (nW > nCs ? nW : nCs);
        zero(c, BF(c, B_ZERO, float, nzb), sizeof(float) * nzb);
    }
    for (int k = 0; k < np; k++) {
        Step s = make_step(c, B, c->tr_Tr[k], c->tr_Tm[k]);
        c->prof_pass = 0; prof_mark(c, PR_EMBED);
        k_embed<<<s.N, 256, 0, c->stream>>>(hx(c, k, 0), a->ids, T, pst[k], PP(c, P_EMB), PP(c, P_MEMTOK), s.Tr, s.Tm, D);
    }
    c->hy_active = 1;
}

template <int KD>
static void hy_run_impl(Ctx* c, int lo, int hi, const float* real_in, const float* mem_in, float* real_out, float* mem_out) {
    Dims& d = c->d;
    const ArmtForwardArgs* a = hya(c);
    int B = c->tr_B, np = c->hy_np, D = d.D;
    long nS = (long)B * d.H * KD * KD, nCs = (long)B * d.Cc * d.W, nW = (long)B * d.dk * D / d.Hm, nZ = (long)B * d.dk;
    if (real_in) for (int k = 0; k < np; k++) hy_copy(c, k, hx(c, k, lo), real_in, mem_in, 1, NULL);
    float* zbuf = (float*)c->buf[B_ZERO].p;
    for (int i = lo; i < hi; i++) {
        if (d.c.attn[i]) armt_fail("armt_hy_run: layer " + std::to_string(i) + " is an attention layer");
        for (int k = 0; k < np; k++) {
            Step s = make_step(c, B, c->tr_Tr[k], c->tr_Tm[k]);
            LState in, out;
            if (c->hy_train) {
                float *SS = (float*)c->buf[B_ST_S].p, *SCS = (float*)c->buf[B_ST_CS].p, *SW = (float*)c->buf[B_ST_W].p, *SZ = (float*)c->buf[B_ST_Z].p;
                long si = (long)i * np + k;
                in = {SS + nS * si, SCS + nCs * si, d.has_mem[i] ? SW + nW * si : NULL, d.has_mem[i] ? SZ + nZ * si : NULL};
                if (a->reset && k > 0) {
                    LAUNCH1D(k_mask_zero, nS, in.S, a->reset + (long)k * B, (long)d.H * KD * KD, B);
                    LAUNCH1D(k_mask_zero, nCs, in.cs, a->reset + (long)k * B, (long)d.Cc * d.W, B);
                }
                out = {NULL, NULL, NULL, NULL};
                if (k + 1 < np) out = {SS + nS * (si + 1), SCS + nCs * (si + 1), d.has_mem[i] ? SW + nW * (si + 1) : NULL, d.has_mem[i] ? SZ + nZ * (si + 1) : NULL};
            } else {
                if (a->state) { in.S = a->state[4 * i]; in.cs = a->state[4 * i + 1]; in.Wm = a->state[4 * i + 2]; in.z = a->state[4 * i + 3]; }
                else {   // no state kept: carry within this call through scratch state buffers
                    float* st = BF(c, B_IST_S, float, nS * d.L); float* scs = BF(c, B_IST_CS, float, nCs * d.L);
                    float* sw = BF(c, B_IST_W, float, nW * d.L); float* sz = BF(c, B_IST_Z, float, nZ * d.L);
                    if (k == 0) { zero(c, st + nS * i, nS * 4); zero(c, scs + nCs * i, nCs * 4); zero(c, sw + nW * i, nW * 4); zero(c, sz + nZ * i, nZ * 4); }
                    in.S = st + nS * i; in.cs = scs + nCs * i; in.Wm = sw + nW * i; in.z = sz + nZ * i;
                }
                if (a->reset) {
                    LAUNCH1D(k_mask_zero, nS, in.S, a->reset + (long)k * B, (long)d.H * KD * KD, B);
                    LAUNCH1D(k_mask_zero, nCs, in.cs, a->reset + (long)k * B, (long)d.Cc * d.W, B);
                }
                if (!d.has_mem[i]) { in.Wm = in.z = NULL; }
                out = in;
                if (d.has_mem[i] && !in.Wm) { in.Wm = zbuf; in.z = zbuf; out.Wm = out.z = NULL; }
            }
            layer_fwd<KD>(c, i, s, hx(c, k, i), hx(c, k, i + 1), &in, &out);
        }
    }
    if (real_out || mem_out) for (int k = 0; k < np; k++) hy_copy(c, k, hx(c, k, hi), real_out, mem_out, 0, NULL);
}

static void hy_head_impl(Ctx* c, const float* real_in) {
    Dims& d = c->d;
    const ArmtForwardArgs* a = hya(c);
    int np = c->hy_np;
    long xs_per = hy_xs_per(c);
    for (int k = 0; k < np; k++) {
        Step s = make_step(c, c->tr_B, c->tr_Tr[k], c->tr_Tm[k]);
        float* x = hx(c, k, d.L);
        if (real_in) hy_copy(c, k, x, real_in, NULL, 1, NULL);
        float* dx = NULL;
        if (c->hy_train) { dx = (float*)c->buf[B_DXS].p + xs_per * k; zero(c, dx, (size_t)s.N * d.D * 4); }
        head(c, s, x, c->tr_start[k], c->tr_T, a->labels, a->token_losses, a->logits, a->logits_last, k == np - 1, a->inv_n, dx);
    }
    if (!c->hy_train) c->hy_active = 0;
}

static void hy_head_bwd_impl(Ctx* c, const float* gscale, float* d_real) {
    Dims& d = c->d;
    c->grad = c->tr_grad;
    if (gscale) {   // the head's parameter grads were computed in the forward for a loss gradient of 1
        float* gn = GP(c, P_NORMF);
        if (gn) LAUNCH1D(k_scale, (long)d.D, gn, (long)d.D, gscale);
        float* gl = GP(c, P_LMHEAD);   // the embedding's when tied: it holds only the head's part until armt_hy_run_bwd(0, ..)
        if (gl) LAUNCH1D(k_scale, (long)d.V * d.D, gl, (long)d.V * d.D, gscale);
    }
    long xs_per = hy_xs_per(c);
    for (int k = 0; k < c->hy_np; k++) hy_copy(c, k, (float*)c->buf[B_DXS].p + xs_per * k, d_real, NULL, 0, gscale);
}

template <int KD>
static void hy_run_bwd_impl(Ctx* c, int lo, int hi, const float* d_real_out, const float* d_mem_out, float* d_real_in, float* d_mem_in) {
    Dims& d = c->d;
    c->grad = c->tr_grad;
    int B = c->tr_B, np = c->hy_np, D = d.D;
    long xs_per = hy_xs_per(c);
    long nS = (long)B * d.H * KD * KD, nCs = (long)B * d.Cc * d.W, nW = (long)B * d.dk * D / d.Hm, nZ = (long)B * d.dk;
    float *DXS = (float*)c->buf[B_DXS].p;
    float *SS = (float*)c->buf[B_ST_S].p, *SCS = (float*)c->buf[B_ST_CS].p, *SW = (float*)c->buf[B_ST_W].p, *SZ = (float*)c->buf[B_ST_Z].p;
    if (d_real_out || d_mem_out) for (int k = 0; k < np; k++) hy_copy(c, k, DXS + xs_per * k, d_real_out, d_mem_out, 1, NULL);
    float *dS = BF(c, B_DST_S, float, nS), *dCs = BF(c, B_DST_CS, float, nCs), *dW = BF(c, B_DST_W, float, nW), *dZ = BF(c, B_DST_Z, float, nZ);
    for (int i = hi - 1; i >= lo; i--) {
        if (d.c.attn[i]) armt_fail("armt_hy_run_bwd: layer " + std::to_string(i) + " is an attention layer");
        c->prof_pass = 2; prof_mark(c, PR_COPY);
        zero(c, dS, nS * 4); zero(c, dCs, nCs * 4); zero(c, dW, nW * 4); zero(c, dZ, nZ * 4);
        for (int k = np - 1; k >= 0; k--) {
            Step s = make_step(c, B, c->tr_Tr[k], c->tr_Tm[k]);
            long si = (long)i * np + k;
            LState in = {SS + nS * si, SCS + nCs * si, d.has_mem[i] ? SW + nW * si : NULL, d.has_mem[i] ? SZ + nZ * si : NULL};
            float* xo_tmp = BF(c, B_XOT, float, xs_per);
            LState disc = {NULL, NULL, d.has_mem[i] ? BF(c, B_WDISC, float, nW) : NULL, d.has_mem[i] ? BF(c, B_ZDISC, float, nZ) : NULL};
            layer_fwd<KD>(c, i, s, hx(c, k, i), xo_tmp, &in, &disc, true);
            LState dst = {dS, dCs, d.has_mem[i] ? dW : NULL, d.has_mem[i] ? dZ : NULL};
            layer_bwd<KD>(c, i, s, hx(c, k, i), hx(c, k, i + 1), &in, DXS + xs_per * k, &dst);
            if (c->tr_reset && k > 0) {
                LAUNCH1D(k_mask_zero, nS, dS, c->tr_reset + (long)k * B, (long)d.H * KD * KD, B);
                LAUNCH1D(k_mask_zero, nCs, dCs, c->tr_reset + (long)k * B, (long)d.Cc * d.W, B);
            }
        }
    }
    if (lo > 0) {
        if (d_real_in || d_mem_in) for (int k = 0; k < np; k++) hy_copy(c, k, DXS + xs_per * k, d_real_in, d_mem_in, 0, NULL);
        return;
    }
    const ArmtForwardArgs* a = hya(c);
    for (int k = 0; k < np; k++) {
        Step s = make_step(c, B, c->tr_Tr[k], c->tr_Tm[k]);
        c->prof_pass = 2; prof_mark(c, PR_EMBED);
        k_embed_bwd<<<s.N, 256, 0, c->stream>>>(DXS + xs_per * k, a->ids, c->tr_T, c->tr_start[k], GP(c, P_EMB), GP(c, P_MEMTOK), s.Tr, s.Tm, D);
    }
    c->hy_active = 0;
}

API int armt_hy_begin(void* p, const float** params, const ArmtForwardArgs* a, int* pieces, int cap, int* np_out) {
    API_TRY
    Ctx* c = (Ctx*)p;
    c->params = params;
    set_stream(c, a->stream);
    if (c->d.KD == 64) hy_begin_impl<64>(c, a, pieces, cap, np_out); else hy_begin_impl<128>(c, a, pieces, cap, np_out);
    CK(cudaGetLastError());
    prof_flush(c);
    return 0;
    API_CATCH(-1)
}
API int armt_hy_run(void* p, const float** params, int lo, int hi, const float* real_in, const float* mem_in, float* real_out, float* mem_out,
                    void* stream) {
    API_TRY
    Ctx* c = (Ctx*)p;
    if (!c->hy_active) armt_fail("armt_hy_run without armt_hy_begin");
    if (lo < 0 || hi > c->d.L || lo > hi) armt_fail("layer range out of bounds");
    c->params = params;
    set_stream(c, stream);
    if (c->d.KD == 64) hy_run_impl<64>(c, lo, hi, real_in, mem_in, real_out, mem_out);
    else hy_run_impl<128>(c, lo, hi, real_in, mem_in, real_out, mem_out);
    CK(cudaGetLastError());
    prof_flush(c);
    return 0;
    API_CATCH(-1)
}
API int armt_hy_head(void* p, const float** params, const float* real_in, void* stream) {
    API_TRY
    Ctx* c = (Ctx*)p;
    if (!c->hy_active) armt_fail("armt_hy_head without armt_hy_begin");
    c->params = params;
    set_stream(c, stream);
    hy_head_impl(c, real_in);
    CK(cudaGetLastError());
    prof_flush(c);
    return 0;
    API_CATCH(-1)
}
API int armt_hy_head_bwd(void* p, const float** params, const float* gscale, float* d_real, void* stream) {
    API_TRY
    Ctx* c = (Ctx*)p;
    if (!c->hy_active || !c->hy_train || !c->tr_grad) armt_fail("armt_hy_head_bwd without a recorded training forward");
    c->params = params;
    set_stream(c, stream);
    hy_head_bwd_impl(c, gscale, d_real);
    CK(cudaGetLastError());
    prof_flush(c);
    return 0;
    API_CATCH(-1)
}
API int armt_hy_run_bwd(void* p, const float** params, int lo, int hi, const float* d_real_out, const float* d_mem_out, float* d_real_in,
                        float* d_mem_in, void* stream) {
    API_TRY
    Ctx* c = (Ctx*)p;
    if (!c->hy_active || !c->hy_train || !c->tr_grad) armt_fail("armt_hy_run_bwd without a recorded training forward");
    if (lo < 0 || hi > c->d.L || lo > hi) armt_fail("layer range out of bounds");
    c->params = params;
    set_stream(c, stream);
    if (c->d.KD == 64) hy_run_bwd_impl<64>(c, lo, hi, d_real_out, d_mem_out, d_real_in, d_mem_in);
    else hy_run_bwd_impl<128>(c, lo, hi, d_real_out, d_mem_out, d_real_in, d_mem_in);
    CK(cudaGetLastError());
    prof_flush(c);
    if (lo == 0) c->tr_grad = NULL;
    return 0;
    API_CATCH(-1)
}

// forward. params: n_params device pointers in canonical order.
API int armt_forward(void* p, const float** params, const ArmtForwardArgs* a) {
    API_TRY
    Ctx* c = (Ctx*)p;
    for (int i = 0; i < c->d.L; i++) if (c->d.c.attn[i]) armt_fail("a hybrid backbone (attention layers) runs through armt_hy_*");
    c->params = params;
    set_stream(c, a->stream);
    if (c->d.KD == 64) forward_impl<64>(c, a); else forward_impl<128>(c, a);
    CK(cudaGetLastError());
    prof_flush(c);
    return 0;
    API_CATCH(-1)
}
// backward of the last training forward: grads (+)= into the flat buffer given to the forward, then scaled by *gscale.
// armt_backward_range: layers [lo, hi) only (and the embeddings if lo == 0); call it for hi = n_layers first, then for
// consecutive lower ranges down to lo = 0, which ends the backward. armt_backward = the whole range.
API int armt_backward_range(void* p, const float** params, const int64_t* ids, const float* gscale, void* stream, int lo, int hi) {
    API_TRY
    Ctx* c = (Ctx*)p;
    if (lo < 0 || hi > c->d.L || lo > hi) armt_fail("backward range out of bounds");
    c->params = params;
    set_stream(c, stream);
    if (c->d.KD == 64) backward_impl<64>(c, ids, gscale, lo, hi); else backward_impl<128>(c, ids, gscale, lo, hi);
    CK(cudaGetLastError());
    prof_flush(c);
    if (lo == 0) c->tr_grad = NULL;
    return 0;
    API_CATCH(-1)
}
API int armt_backward(void* p, const float** params, const int64_t* ids, const float* gscale, void* stream) {
    return armt_backward_range(p, params, ids, gscale, stream, 0, ((Ctx*)p)->d.L);
}


// phase profile (ARMT_PROF=1): a text table of GPU ms per phase (forward / recompute / backward) since the last reset, divided
// by nsteps
API int armt_prof_report(void* p, char* buf, int len, int reset, int nsteps) {
    Ctx* c = (Ctx*)p;
    if (!c->prof) { snprintf(buf, len, "phase profiler off (set ARMT_PROF=1 before the context is created)\n"); return 0; }
    Prof& f = *c->prof;
    double div = nsteps > 0 ? nsteps : 1, tot = 0.0, col[3] = {};
    for (int q = 0; q < 3; q++) for (int i = 0; i < PR_COUNT; i++) f.ms[q][i] /= div;
    for (int q = 0; q < 3; q++) for (int i = 1; i < PR_COUNT; i++) { tot += f.ms[q][i]; col[q] += f.ms[q][i]; }
    int n = snprintf(buf, len, "%-38s %10s %10s %10s %10s %6s\n", "phase", "forward", "recompute", "backward", "total ms", "share");
    int order[PR_COUNT], m = 0;
    for (int i = 1; i < PR_COUNT; i++) order[m++] = i;
    for (int a = 0; a < m; a++) for (int b = a + 1; b < m; b++) {
        double ta = f.ms[0][order[a]] + f.ms[1][order[a]] + f.ms[2][order[a]], tb = f.ms[0][order[b]] + f.ms[1][order[b]] + f.ms[2][order[b]];
        if (tb > ta) { int t = order[a]; order[a] = order[b]; order[b] = t; }
    }
    for (int a = 0; a < m && n < len; a++) {
        int i = order[a];
        double t = f.ms[0][i] + f.ms[1][i] + f.ms[2][i];
        if (t <= 0.0) continue;
        n += snprintf(buf + n, len - n, "%-38s %10.2f %10.2f %10.2f %10.2f %5.1f%%\n", PR_NAME[i], f.ms[0][i], f.ms[1][i], f.ms[2][i], t,
                      tot > 0 ? 100.0 * t / tot : 0.0);
    }
    if (n < len) n += snprintf(buf + n, len - n, "%-38s %10.2f %10.2f %10.2f %10.2f\n", "all phases", col[0], col[1], col[2], tot);
    for (int q = 0; q < 3; q++) for (int i = 0; i < PR_COUNT; i++) f.ms[q][i] = reset ? 0.0 : f.ms[q][i] * div;
    return 0;
}
