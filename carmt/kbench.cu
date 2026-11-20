// kbench.cu -- development benchmark: GPU time per phase of one GDN-2 ARMT layer step (forward, the backward's
// recompute, backward) at the real sizes, through the library's own code paths and its phase profiler. No Python.
//
//   nvcc -O3 -arch=sm_89 -o kbench carmt/kbench.cu -lcublas -lcublasLt     (bf16 build; add -DARMT_FP32 for fp32)
//   ./kbench [reps] [B] [dump_file] [seed]          (-DKBENCH_NO_PROF to build against a library without the phase profiler)
//
// Inputs are pseudo-random with a fixed seed, so two builds see the same data; with a dump file the step's outputs
// (layer output, carried states, input grads, parameter grads) are written for comparing builds (kbench_cmp.py).
#include "armt.cu"
#include <chrono>
#include <functional>

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static float frand() {   // uniform [-1, 1)
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (float)((rng_state >> 40) & 0xFFFFFF) / 8388608.f - 1.f;
}
static float* dev_fill(long n, float scale, float offset = 0.f, bool positive = false) {
    std::vector<float> h(n);
    for (long i = 0; i < n; i++) { float v = frand() * scale; h[i] = (positive ? fabsf(v) : v) + offset; }
    float* d = NULL;
    CK(cudaMalloc(&d, n * sizeof(float)));
    CK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}
static void dump(FILE* f, const char* name, const float* d, long n) {
    std::vector<float> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    int len = (int)strlen(name);
    fwrite(&len, 4, 1, f); fwrite(name, 1, len, f); fwrite(&n, 8, 1, f); fwrite(h.data(), 4, n, f);
}

int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 10, B = argc > 2 ? atoi(argv[2]) : 8;
    const char* dumpf = argc > 3 ? argv[3] : NULL;
    if (argc > 4) rng_state ^= 0xD1B54A32D192ED03ull * (uint64_t)atoll(argv[4]);   // input seed
    ArmtConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.vocab = 32000; cfg.d_model = 1024; cfg.n_layers = 1; cfg.n_heads = 16; cfg.head_dim = 128; cfg.d_ff = 2048; cfg.conv_size = 4;
    cfg.gate_swish = 1; cfg.norm_eps = 1e-5f; cfg.num_mem = 32; cfg.d_mem = 64; cfg.mem_heads = 1; cfg.segment_size = 1024;
    cfg.mem_in_stream = 0; cfg.gating = 0; cfg.tie_emb = 0; cfg.allow_tf32 = 1; cfg.wrap[0] = 1;
    Ctx* c = (Ctx*)armt_create(&cfg);
    if (!c) { fprintf(stderr, "create: %s\n", armt_last_error()); return 1; }
    Dims& d = c->d;
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    set_stream(c, st);
    // parameters: small random weights, norms ~1, decay / gate parameters in their usual ranges
    std::vector<const float*> params(d.n_params, nullptr);
    for (long i = 0; i < d.n_params; i++) {
        long n = d.numel[i];
        if (!n) continue;
        int j = i >= P_GLOBAL ? (int)((i - P_GLOBAL) % L_COUNT) : -1;
        float* p;
        if (j == L_N1 || j == L_N2 || j == L_ON || i == P_NORMF) p = dev_fill(n, 0.1f, 1.f);
        else if (j == L_ALOG) p = dev_fill(n, 1.f, 1.5f);           // A = e^{A_log} in ~[1.6, 12]
        else if (j == L_DTB) p = dev_fill(n, 1.f, -2.f);
        else if (j == L_QC || j == L_KC || j == L_VC) p = dev_fill(n, 0.4f);
        else if (j == L_MV) p = dev_fill(n, 0.02f);
        else p = dev_fill(n, 0.03f);
        params[i] = p;
    }
    c->params = params.data();
    pack_weights(c);
    float* grad = NULL;
    CK(cudaMalloc(&grad, d.total * sizeof(float)));
    // one segment piece with memory tokens
    Step s = make_step(c, B, 1024, 32);
    int D = d.D, N = s.N;
    long nS = (long)B * d.H * d.KD * d.KD, nCs = (long)B * d.Cc * d.W, nW = (long)B * d.dk * D / d.Hm, nZ = (long)B * d.dk;
    float* x = dev_fill((long)N * D, 1.f);
    float* xo = dev_fill((long)N * D, 0.f);
    float* xo2 = dev_fill((long)N * D, 0.f);
    float* dx0 = dev_fill((long)N * D, 1e-3f);
    float* dx = dev_fill((long)N * D, 0.f);
    LState in = {dev_fill(nS, 0.02f), dev_fill(nCs, 0.5f), dev_fill(nW, 0.05f), dev_fill(nZ, 1.f, 0.5f, true)};
    LState out = {dev_fill(nS, 0.f), dev_fill(nCs, 0.f), dev_fill(nW, 0.f), dev_fill(nZ, 0.f)};
    LState disc = {NULL, NULL, dev_fill(nW, 0.f), dev_fill(nZ, 0.f)};
    LState dst = {dev_fill(nS, 0.f), dev_fill(nCs, 0.f), dev_fill(nW, 0.f), dev_fill(nZ, 0.f)};
    float* dS0 = dev_fill(nS, 1e-3f); float* dCs0 = dev_fill(nCs, 1e-3f); float* dW0 = dev_fill(nW, 1e-3f); float* dZ0 = dev_fill(nZ, 1e-3f);
    c->grad = grad;
    auto one = [&](bool fwd, bool bwd) {
        if (fwd) layer_fwd<128>(c, 0, s, x, xo, &in, &out);
        if (bwd) {
            CK(cudaMemcpyAsync(dx, dx0, (size_t)N * D * 4, cudaMemcpyDeviceToDevice, st));
            CK(cudaMemcpyAsync(dst.S, dS0, nS * 4, cudaMemcpyDeviceToDevice, st));
            CK(cudaMemcpyAsync(dst.cs, dCs0, nCs * 4, cudaMemcpyDeviceToDevice, st));
            CK(cudaMemcpyAsync(dst.Wm, dW0, nW * 4, cudaMemcpyDeviceToDevice, st));
            CK(cudaMemcpyAsync(dst.z, dZ0, nZ * 4, cudaMemcpyDeviceToDevice, st));
            layer_fwd<128>(c, 0, s, x, xo2, &in, &disc, true);
            layer_bwd<128>(c, 0, s, x, xo, &in, dx, &dst);
        }
    };
    try {
        CK(cudaMemsetAsync(grad, 0, d.total * sizeof(float), st));
        for (int w = 0; w < 3; w++) one(true, true);
        CK(cudaStreamSynchronize(st));
        if (dumpf) {   // outputs of one clean step
            CK(cudaMemsetAsync(grad, 0, d.total * sizeof(float), st));
            one(true, true);
            CK(cudaStreamSynchronize(st));
            FILE* f = fopen(dumpf, "wb");
            dump(f, "xo", xo, (long)N * D); dump(f, "S_out", out.S, nS); dump(f, "cs_out", out.cs, nCs); dump(f, "W_out", out.Wm, nW);
            dump(f, "z_out", out.z, nZ); dump(f, "dx", dx, (long)N * D); dump(f, "dS", dst.S, nS); dump(f, "dcs", dst.cs, nCs);
            dump(f, "dW", dst.Wm, nW); dump(f, "dz", dst.z, nZ);
            for (int j = 0; j < L_COUNT; j++) {
                long i = PI(0, j);
                if (d.numel[i]) { char nm[32]; snprintf(nm, sizeof nm, "grad_%d", j); dump(f, nm, grad + d.offset[i], d.numel[i]); }
            }
            fclose(f);
        }
#ifdef INTRA_TIMING
        {   // where a k_intra_fwd block spends its time (cycles, averaged over the blocks of one forward call)
            long long* tb = NULL;
            CK(cudaMalloc(&tb, s.nchB * 8 * sizeof(long long)));
            CK(cudaMemcpyToSymbol(intra_timing, &tb, sizeof(tb)));
            one(true, false);
            CK(cudaStreamSynchronize(st));
            std::vector<long long> h(s.nchB * 8);
            CK(cudaMemcpy(h.data(), tb, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
            const char* nm[8] = {"gc scan (8 tiles)", "loads + y/Qg/Kt (8 tiles)", "decayed operands (8 tiles)", "pair sums (8 tiles)",
                                 "assemble L, M; write M", "inverse: diagonal blocks", "inverse: off-diagonal", "total incl. A write"};
            for (int k = 0; k < 8; k++) {
                double a = 0;
                for (long i = 0; i < s.nchB; i++) a += h[i * 8 + k];
                printf("intra_fwd %-28s %9.0f cycles\n", nm[k], a / s.nchB);
            }
            CK(cudaMemcpyToSymbol(intra_timing_b, &tb, sizeof(tb)));
            one(false, true);
            CK(cudaStreamSynchronize(st));
            CK(cudaMemcpy(h.data(), tb, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
            const char* nb[8] = {"dgl: S0 . dSe row sums", "dKt.Kt; stage dL, dM", "tile loads (8)", "decayed operands (8)",
                                 "tensor-core products (8)", "per-element outputs (8)", "dg reverse cumsum (8)", "total"};
            for (int k = 0; k < 8; k++) {
                double a = 0;
                for (long i = 0; i < s.nchB; i++) a += h[i * 8 + k];
                printf("intra_bwd %-28s %9.0f cycles\n", nb[k], a / s.nchB);
            }
        }
#endif
        if (getenv("KBENCH_GEMMS")) {   // each GEMM of the scan's backward on its own (the buffers of the last step)
            const int KD = 128;
            long CSK = s.nchB * CH * KD, CSC = s.nchB * CH * CH;
            int nb = (int)s.nchB;
            long long sK = CH * KD, sC = CH * CH, sS = (long long)KD * KD;
            floatS *Qg = BF(c, B_QG, floatS, CSK), *vn = BF(c, B_VN, floatS, CSK), *Mm = BF(c, B_MMAT, floatS, CSC), *S0 = BF(c, B_S0, floatS, s.nchB * KD * KD);
            floatS *AmX = BF(c, B_AMX, floatS, CSC), *cwv = BF(c, B_CWV, floatS, CSK), *y = BF(c, B_Y, floatS, CSK);
            float* Am = BF(c, B_AM, float, CSC);
            floatS *dO = BF(c, B_DO, floatS, CSK), *dSe = BF(c, B_DSE, floatS, s.nchB * KD * KD), *dvn = BF(c, B_DVN, floatS, CSK);
            floatS *pre_dvn = BF(c, B_PREDVN, floatS, CSK), *dWm = BF(c, B_DWM, floatS, CSK), *dQg = BF(c, B_DQG, floatS, CSK);
            floatS *dKt = BF(c, B_DKT, floatS, CSK), *dy = BF(c, B_DY, floatS, CSK), *dwv = BF(c, B_DWV, floatS, CSK);
            float *dM = BF(c, B_DMMAT, float, CSC), *dA = BF(c, B_DAM, float, CSC), *T1 = BF(c, B_T1, float, CSC), *dL = BF(c, B_DL, float, CSC);
            auto timeit = [&](const char* nm, std::function<void()> f) {
                cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1));
                f(); CK(cudaEventRecord(a0, st));
                for (int r = 0; r < 20; r++) f();
                CK(cudaEventRecord(a1, st)); CK(cudaEventSynchronize(a1));
                float ms; CK(cudaEventElapsedTime(&ms, a0, a1)); printf("gemm %-26s %.3f ms\n", nm, ms / 20);
            };
#define SC CUDA_S
            timeit("pre_dvn = M^T do", [&] { mm(c, 1, 0, CH, KD, CH, 1.f, Mm, SC, CH, sC, dO, SC, KD, sK, 0.f, pre_dvn, SC, KD, sK, nb); });
            timeit("pre_dS = Qg^T do", [&] { mm(c, 1, 0, KD, KD, CH, 1.f, Qg, SC, KD, sK, dO, SC, KD, sK, 0.f, dSe, SC, KD, sS, nb); });
            timeit("dQg = do S0^T", [&] { mm(c, 0, 1, CH, KD, KD, 1.f, dO, SC, KD, sK, S0, SC, KD, sS, 0.f, dQg, SC, KD, sK, nb); });
            timeit("dKt = vn dSe^T", [&] { mm(c, 0, 1, CH, KD, KD, 1.f, vn, SC, KD, sK, dSe, SC, KD, sS, 0.f, dKt, SC, KD, sK, nb); });
            timeit("dWm = -dvn S0^T", [&] { mm(c, 0, 1, CH, KD, KD, -1.f, dvn, SC, KD, sK, S0, SC, KD, sS, 0.f, dWm, SC, KD, sK, nb); });
            timeit("dM = do vn^T (f32)", [&] { mm(c, 0, 1, CH, CH, KD, 1.f, dO, SC, KD, sK, vn, SC, KD, sK, 0.f, dM, F32, CH, sC, nb); });
            timeit("dwv = A^T du", [&] { mm(c, 1, 0, CH, KD, CH, 1.f, AmX, SC, CH, sC, dvn, SC, KD, sK, 0.f, dwv, SC, KD, sK, nb); });
            timeit("dy = A^T dWm", [&] { mm(c, 1, 0, CH, KD, CH, 1.f, AmX, SC, CH, sC, dWm, SC, KD, sK, 0.f, dy, SC, KD, sK, nb); });
            timeit("dA = du wv^T (f32)", [&] { mm(c, 0, 1, CH, CH, KD, 1.f, dvn, SC, KD, sK, cwv, SC, KD, sK, 0.f, dA, F32, CH, sC, nb); });
            timeit("dA += dWm y^T (f32)", [&] { mm(c, 0, 1, CH, CH, KD, 1.f, dWm, SC, KD, sK, y, SC, KD, sK, 1.f, dA, F32, CH, sC, nb); });
            timeit("T1 = dA A^T (f32 x f32)", [&] { mm(c, 0, 1, CH, CH, CH, 1.f, dA, F32, CH, sC, Am, F32, CH, sC, 0.f, T1, F32, CH, sC, nb); });
            timeit("dL = -A^T T1 (f32 x f32)", [&] { mm(c, 1, 0, CH, CH, CH, -1.f, Am, F32, CH, sC, T1, F32, CH, sC, 0.f, dL, F32, CH, sC, nb); });
            timeit("o = Qg S0 (fwd)", [&] { mm(c, 0, 0, CH, KD, KD, 1.f, Qg, SC, KD, sK, S0, SC, KD, sS, 0.f, dQg, SC, KD, sK, nb); });
            timeit("o += M vn (fwd)", [&] { mm(c, 0, 0, CH, KD, CH, 1.f, Mm, SC, CH, sC, vn, SC, KD, sK, 1.f, dQg, SC, KD, sK, nb); });
            timeit("u = A wv (fwd)", [&] { mm(c, 0, 0, CH, KD, CH, 1.f, AmX, SC, CH, sC, cwv, SC, KD, sK, 0.f, dwv, SC, KD, sK, nb); });
#undef SC
            // the layer's large linears (bf16, fp32 accumulation): TFLOPS
            int D = d.D, NP = d.NP, F = d.F, Nn = s.N;
            floatX *h1 = BF(c, B_H1, floatX, (long)Nn * D), *P = BF(c, B_P, floatX, (long)Nn * NP), *dP = BF(c, B_DP, floatX, (long)Nn * NP);
            floatX *dh1 = BF(c, B_DH1, floatX, (long)Nn * D), *a2 = BF(c, B_A, floatX, (long)Nn * 2 * F), *da = BF(c, B_DA, floatX, (long)Nn * 2 * F);
            floatX *h2 = BF(c, B_H2, floatX, (long)Nn * D);
            float* gW = grad + d.offset[PI(0, L_Q)];
            float* gW12 = grad + d.offset[PI(0, L_W1)];
            auto tf = [&](const char* nm, double flop, std::function<void()> f) {
                cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1));
                f(); CK(cudaEventRecord(a0, st));
                for (int r = 0; r < 10; r++) f();
                CK(cudaEventRecord(a1, st)); CK(cudaEventSynchronize(a1));
                float ms; CK(cudaEventElapsedTime(&ms, a0, a1)); ms /= 10;
                printf("linear %-34s %.3f ms  %6.1f TFLOPS\n", nm, ms, flop / ms / 1e9);
            };
            double fP = 2.0 * Nn * NP * D, f12 = 2.0 * Nn * 2 * F * D;
            tf("P = h1 W_in^T (fwd)", fP, [&] { lin(c, P, NP, h1, D, c->w_in[0], Nn, NP, D, 0.f); });
            tf("dh1 = dP W_in (bwd dx)", fP, [&] { lin_dx(c, dh1, D, dP, NP, c->w_in[0], Nn, NP, D, 0.f); });
            tf("dW_in += dP^T h1 (bwd dW, fp32 acc)", fP, [&] { lin_dw(c, gW, dP, NP, h1, D, Nn, NP, D); });
            tf("a = h2 W12^T (MLP fwd)", f12, [&] { lin(c, a2, 2 * F, h2, D, c->w_12[0], Nn, 2 * F, D, 0.f); });
            tf("dh2 = da W12 (MLP dx)", f12, [&] { lin_dx(c, dh1, D, da, 2 * F, c->w_12[0], Nn, 2 * F, D, 0.f); });
            tf("dW12 += da^T h2 (MLP dW)", f12, [&] { lin_dw(c, gW12, da, 2 * F, h2, D, Nn, 2 * F, D); });
            // the same with the gradient 4-byte aligned (an odd float offset), as for every other layer when a layer's
            // parameter count is odd
            tf("dW_in, grad at an odd offset", fP, [&] { lin_dw(c, gW + 1, dP, NP, h1, D, Nn, NP, D); });
            tf("dW12, grad at an odd offset", f12, [&] { lin_dw(c, gW12 + 1, da, 2 * F, h2, D, Nn, 2 * F, D); });
            tf("dW_o, grad aligned", 2.0 * Nn * D * d.VDt, [&] { lin_dw(c, grad + d.offset[PI(0, L_O)], BF(c, B_DXX, floatX, (long)Nn * D), D, BF(c, B_ON, floatX, (long)Nn * d.VDt), d.VDt, Nn, D, d.VDt); });
            tf("dW_o, grad at an odd offset", 2.0 * Nn * D * d.VDt, [&] { lin_dw(c, grad + d.offset[PI(0, L_O)] + 1, BF(c, B_DXX, floatX, (long)Nn * D), D, BF(c, B_ON, floatX, (long)Nn * d.VDt), d.VDt, Nn, D, d.VDt); });
        }
        // timing: whole forward and recompute+backward with the profiler, then without it (wall clock of the stream)
#ifndef KBENCH_NO_PROF   // (-DKBENCH_NO_PROF for library versions without the phase profiler)
        c->prof = new Prof();
        for (int r = 0; r < reps; r++) { one(true, true); prof_flush(c); }
        static char buf[1 << 14];
        armt_prof_report(c, buf, sizeof buf, 1, reps);
        printf("one layer x one segment piece (B=%d, %d + %d tokens), ms per step:\n%s", B, s.Tr, s.Tm, buf);
        Prof* pf = c->prof; c->prof = NULL;
        for (cudaEvent_t e : pf->ev) cudaEventDestroy(e);
        delete pf;
#endif
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        float msf = 0.f, msb = 0.f;
        CK(cudaEventRecord(e0, st));
        for (int r = 0; r < reps; r++) one(true, false);
        CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&msf, e0, e1));
        CK(cudaEventRecord(e0, st));
        for (int r = 0; r < reps; r++) one(false, true);
        CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&msb, e0, e1));
        printf("unprofiled: forward %.3f ms, recompute + backward %.3f ms, total %.3f ms per layer-segment step\n", msf / reps, msb / reps,
               (msf + msb) / reps);
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
