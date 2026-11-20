// gemm_search.cu -- development tool: for the large GEMMs of a GDN-2 ARMT training step (370M sizes), time the library's
// tuned choice (mm(), with its ARMT_GEMM_CACHE), cublasGemmEx's default and every cuBLASLt heuristic candidate, under
// steady clocks (a warm-up first, then interleaved rounds; the best round per candidate counts).
//
//   nvcc -O3 -arch=sm_89 -std=c++17 -o gemm_search carmt/gemm_search.cu -lcublas -lcublasLt
//   ./gemm_search [B] [rounds] [reps] [filter]
#include "armt.cu"
#include <functional>
#include <algorithm>

int main(int argc, char** argv) {
    int Bt = argc > 1 ? atoi(argv[1]) : 8, rounds = argc > 2 ? atoi(argv[2]) : 3, reps = argc > 3 ? atoi(argv[3]) : 10;
    const char* filt = argc > 4 ? argv[4] : NULL;
    ArmtConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.vocab = 32000; cfg.d_model = 1024; cfg.n_layers = 1; cfg.n_heads = 16; cfg.head_dim = 128; cfg.d_ff = 2048; cfg.conv_size = 4;
    cfg.gate_swish = 1; cfg.norm_eps = 1e-5f; cfg.num_mem = 32; cfg.d_mem = 64; cfg.mem_heads = 1; cfg.segment_size = 1024;
    cfg.allow_tf32 = 1; cfg.wrap[0] = 1;
    Ctx* c = (Ctx*)armt_create(&cfg);
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    set_stream(c, st);
    Dims& d = c->d;
    int Nt = Bt * (1024 + 32), Nh = 4096, D = d.D, NP = d.NP, KDt = d.KDt, VDt = d.VDt, F = d.F, V = d.V;
    struct Shape { const char* name; int tA, tB, M, N, K, lda, ldb, ldc; bool f32out; float beta; int coff; };
    std::vector<Shape> shapes = {
        {"in_proj fwd", 0, 1, Nt, NP, D, D, D, NP, false, 0.f, 0},
        {"in_proj dx", 0, 0, Nt, D, NP, NP, D, D, false, 0.f, 0},
        {"in_proj dW", 1, 0, NP, D, Nt, NP, D, D, true, 1.f, 0},
        {"in_proj dW (grad 4B-aligned)", 1, 0, NP, D, Nt, NP, D, D, true, 1.f, 1},
        {"out_proj fwd", 0, 1, Nt, D, VDt, VDt, VDt, D, false, 0.f, 0},
        {"out_proj dx", 0, 0, Nt, VDt, D, D, VDt, VDt, false, 0.f, 0},
        {"out_proj dW", 1, 0, D, VDt, Nt, D, VDt, VDt, true, 1.f, 0},
        {"out_proj dW (grad 4B-aligned)", 1, 0, D, VDt, Nt, D, VDt, VDt, true, 1.f, 1},
        {"w12 fwd", 0, 1, Nt, 2 * F, D, D, D, 2 * F, false, 0.f, 0},
        {"w12 dx", 0, 0, Nt, D, 2 * F, 2 * F, D, D, false, 0.f, 0},
        {"w12 dW", 1, 0, 2 * F, D, Nt, 2 * F, D, D, true, 1.f, 0},
        {"w12 dW (grad 4B-aligned)", 1, 0, 2 * F, D, Nt, 2 * F, D, D, true, 1.f, 1},
        {"w3 fwd", 0, 1, Nt, D, F, F, F, D, false, 0.f, 0},
        {"w3 dx", 0, 0, Nt, F, D, D, F, F, false, 0.f, 0},
        {"w3 dW", 1, 0, D, F, Nt, D, F, F, true, 1.f, 0},
        {"w3 dW (grad 4B-aligned)", 1, 0, D, F, Nt, D, F, F, true, 1.f, 1},
        {"f2 fwd (K=128)", 0, 1, Nt, KDt, d.Hv, NP, d.Hv, KDt, false, 0.f, 0},
        {"head fwd", 0, 1, Nh, V, D, D, D, V, false, 0.f, 0},
        {"head dx", 0, 0, Nh, D, V, V, D, D, false, 0.f, 0},
        {"head dW", 1, 0, V, D, Nh, V, D, D, true, 1.f, 0},
    };
    size_t maxe = 0;
    for (auto& s : shapes) {
        maxe = std::max(maxe, (size_t)(s.tA ? s.K : s.M) * s.lda);
        maxe = std::max(maxe, (size_t)(s.tB ? s.N : s.K) * s.ldb);
        maxe = std::max(maxe, (size_t)s.M * s.ldc + 64);
    }
    floatX *A = NULL, *Bm = NULL;
    float* C = NULL;
    CK(cudaMalloc(&A, maxe * 2)); CK(cudaMalloc(&Bm, maxe * 2)); CK(cudaMalloc(&C, maxe * 4));
    {   // small random operands (no denormals / infinities)
        std::vector<floatX> h(maxe);
        for (size_t i = 0; i < maxe; i++) h[i] = (floatX)(((float)((i * 2654435761u) % 2001) - 1000.f) * 1e-4f);
        CK(cudaMemcpy(A, h.data(), maxe * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(Bm, h.data(), maxe * 2, cudaMemcpyHostToDevice));
        CK(cudaMemset(C, 0, maxe * 4));
    }
    void* ws = ensure(&c->buf[B_LTWS], LT_WS);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto timeit = [&](const std::function<bool()>& f) -> float {
        if (!f()) return 1e30f;
        CK(cudaEventRecord(e0, st));
        for (int r = 0; r < reps; r++) f();
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        return ms / reps;
    };
    for (auto& s : shapes) {
        if (filt && !strstr(s.name, filt)) continue;
        cudaDataType tc = s.f32out ? CUDA_R_32F : CUDA_X;
        void* Cp = s.f32out ? (void*)(C + s.coff) : (void*)C;
        cublasOperation_t oa = s.tB ? CUBLAS_OP_T : CUBLAS_OP_N, ob = s.tA ? CUBLAS_OP_T : CUBLAS_OP_N;
        GemmPlan* p = plan_desc(oa, ob, s.M, s.N, s.K, CUDA_X, s.lda, 0, CUDA_X, s.ldb, 0, tc, s.ldc, 0, 1, CUBLAS_COMPUTE_32F);
        int alC = ptr_align(Cp);
        cublasLtMatmulPreference_t pref;
        CB(cublasLtMatmulPreferenceCreate(&pref));
        uint64_t wsmax = LT_WS;
        uint32_t a16 = 16, ac = (uint32_t)alC;
        CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsmax, sizeof(wsmax)));
        CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &a16, 4));
        CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &a16, 4));
        CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &ac, 4));
        CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &ac, 4));
        const int NC = 64;
        cublasLtMatmulHeuristicResult_t res[NC];
        int n = 0;
        cublasLtMatmulAlgoGetHeuristic(c->lt, p->op, p->la, p->lb, p->lc, p->lc, pref, NC, res, &n);
        // the same with the library's rule (one alignment for all operands: the minimum)
        cublasLtMatmulPreference_t pref2;
        CB(cublasLtMatmulPreferenceCreate(&pref2));
        CB(cublasLtMatmulPreferenceSetAttribute(pref2, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsmax, sizeof(wsmax)));
        for (auto at : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
            CB(cublasLtMatmulPreferenceSetAttribute(pref2, at, &ac, 4));
        cublasLtMatmulHeuristicResult_t res2[LT_CAND];
        int n2 = 0;
        cublasLtMatmulAlgoGetHeuristic(c->lt, p->op, p->la, p->lb, p->lc, p->lc, pref2, LT_CAND, res2, &n2);
        float alpha = 1.f, beta = s.beta;
        struct Cand { std::string nm; std::function<bool()> f; float best = 1e30f; };
        std::vector<Cand> cands;
        cands.push_back({"library mm() (tuned / cached)", [&] {
            mm(c, s.tA, s.tB, s.M, s.N, s.K, 1.f, A, CUDA_X, s.lda, 0, Bm, CUDA_X, s.ldb, 0, beta, Cp, tc, s.ldc, 0, 1); return true; }});
        cands.push_back({"cublasGemmEx default", [&] {
            return cublasGemmEx(c->blas, oa, ob, s.N, s.M, s.K, &alpha, Bm, CUDA_X, s.ldb, A, CUDA_X, s.lda, &beta, Cp, tc, s.ldc,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS; }});
        auto desc = [](const cublasLtMatmulAlgo_t& al) {
            int tile = 0, stages = 0, splitk = 0, red = 0, swz = 0, cust = 0, id = 0; size_t w;
            cublasLtMatmulAlgoConfigGetAttribute(&al, CUBLASLT_ALGO_CONFIG_ID, &id, 4, &w);
            cublasLtMatmulAlgoConfigGetAttribute(&al, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, 4, &w);
            cublasLtMatmulAlgoConfigGetAttribute(&al, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, 4, &w);
            cublasLtMatmulAlgoConfigGetAttribute(&al, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, 4, &w);
            cublasLtMatmulAlgoConfigGetAttribute(&al, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, 4, &w);
            cublasLtMatmulAlgoConfigGetAttribute(&al, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, 4, &w);
            cublasLtMatmulAlgoConfigGetAttribute(&al, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &cust, 4, &w);
            char b[128];
            snprintf(b, sizeof b, "algo %d tile %d stages %d splitk %d red %d swz %d cust %d", id, tile, stages, splitk, red, swz, cust);
            return std::string(b);
        };
        for (int i = 0; i < n; i++) {
            if (res[i].state != CUBLAS_STATUS_SUCCESS || res[i].workspaceSize > LT_WS) continue;
            bool inlib = false;
            for (int j = 0; j < n2; j++) inlib |= !memcmp(&res2[j].algo, &res[i].algo, sizeof(cublasLtMatmulAlgo_t));
            char nm[48];
            snprintf(nm, sizeof nm, "heur #%d%s", i, inlib ? " (lib cand)" : "");
            cublasLtMatmulHeuristicResult_t r = res[i];
            cands.push_back({std::string(nm) + " " + desc(r.algo), [&, r] {
                return cublasLtMatmul(c->lt, p->op, &alpha, Bm, p->la, A, p->lb, &beta, Cp, p->lc, Cp, p->lc, &r.algo, ws,
                                      r.workspaceSize, st) == CUBLAS_STATUS_SUCCESS; }});
        }
        // lib-rule candidates not in the 16-byte list (e.g. the align2 kernels)
        for (int j = 0; j < n2; j++) {
            if (res2[j].state != CUBLAS_STATUS_SUCCESS || res2[j].workspaceSize > LT_WS) continue;
            bool dup = false;
            for (int i = 0; i < n; i++) dup |= !memcmp(&res2[j].algo, &res[i].algo, sizeof(cublasLtMatmulAlgo_t));
            if (dup) continue;
            cublasLtMatmulHeuristicResult_t r = res2[j];
            char nm[48];
            snprintf(nm, sizeof nm, "lib-rule #%d", j);
            cands.push_back({std::string(nm) + " " + desc(r.algo), [&, r] {
                return cublasLtMatmul(c->lt, p->op, &alpha, Bm, p->la, A, p->lb, &beta, Cp, p->lc, Cp, p->lc, &r.algo, ws,
                                      r.workspaceSize, st) == CUBLAS_STATUS_SUCCESS; }});
        }
        // relaxed alignment (4 bytes everywhere): kernels the 16-byte query does not return can still be the fastest
        {
            cublasLtMatmulPreference_t pref3;
            CB(cublasLtMatmulPreferenceCreate(&pref3));
            CB(cublasLtMatmulPreferenceSetAttribute(pref3, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsmax, sizeof(wsmax)));
            uint32_t a4 = 4;
            for (auto at : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
                CB(cublasLtMatmulPreferenceSetAttribute(pref3, at, &a4, 4));
            cublasLtMatmulHeuristicResult_t res3[64];
            int n3 = 0;
            cublasLtMatmulAlgoGetHeuristic(c->lt, p->op, p->la, p->lb, p->lc, p->lc, pref3, 64, res3, &n3);
            cublasLtMatmulPreferenceDestroy(pref3);
            for (int j = 0; j < n3; j++) {
                if (res3[j].state != CUBLAS_STATUS_SUCCESS || res3[j].workspaceSize > LT_WS) continue;
                cublasLtMatmulHeuristicResult_t r = res3[j];
                cands.push_back({"heur4B #" + std::to_string(j) + " " + desc(r.algo), [&, r] {
                    return cublasLtMatmul(c->lt, p->op, &alpha, Bm, p->la, A, p->lb, &beta, Cp, p->lc, Cp, p->lc, &r.algo, ws,
                                          r.workspaceSize, st) == CUBLAS_STATUS_SUCCESS; }});
            }
        }
        // exhaustive: every algo id x tile x stages x split-k x reduction x swizzle x custom option that AlgoCheck accepts
        std::vector<cublasLtMatmulAlgo_t> xalgos;
        std::vector<size_t> xws;
        if (getenv("GS_EXHAUSTIVE")) {
            int ids[512], nids = 0;
            cublasLtMatmulAlgoGetIds(c->lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_X, CUDA_X, tc, tc, 512, ids, &nids);
            for (int ii = 0; ii < nids; ii++) {
                cublasLtMatmulAlgo_t al;
                if (cublasLtMatmulAlgoInit(c->lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_X, CUDA_X, tc, tc, ids[ii], &al) != CUBLAS_STATUS_SUCCESS) continue;
                size_t sz = 0;
                std::vector<int> tiles, stages;
                cublasLtMatmulAlgoCapGetAttribute(&al, CUBLASLT_ALGO_CAP_TILE_IDS, NULL, 0, &sz);
                tiles.resize(sz / 4);
                if (sz) cublasLtMatmulAlgoCapGetAttribute(&al, CUBLASLT_ALGO_CAP_TILE_IDS, tiles.data(), sz, &sz);
                if (tiles.empty()) tiles.push_back(CUBLASLT_MATMUL_TILE_UNDEFINED);
                cublasLtMatmulAlgoCapGetAttribute(&al, CUBLASLT_ALGO_CAP_STAGES_IDS, NULL, 0, &sz);
                stages.resize(sz / 4);
                if (sz) cublasLtMatmulAlgoCapGetAttribute(&al, CUBLASLT_ALGO_CAP_STAGES_IDS, stages.data(), sz, &sz);
                if (stages.empty()) stages.push_back(CUBLASLT_MATMUL_STAGES_UNDEFINED);
                int splitk = 0, redmask = 0, swzmax = 0, custmax = 0;
                cublasLtMatmulAlgoCapGetAttribute(&al, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitk, 4, &sz);
                cublasLtMatmulAlgoCapGetAttribute(&al, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redmask, 4, &sz);
                cublasLtMatmulAlgoCapGetAttribute(&al, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swzmax, 4, &sz);
                cublasLtMatmulAlgoCapGetAttribute(&al, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &custmax, 4, &sz);
                std::vector<int> sks = {1};
                if (splitk) for (int k : {2, 3, 4, 5, 6, 8}) sks.push_back(k);
                for (int t : tiles) for (int sg : stages) for (int cu = 0; cu <= custmax; cu++) for (int sw = 0; sw <= swzmax; sw++)
                    for (int sk : sks) for (int rs = 0; rs <= (sk > 1 ? 4 : 0); rs++) {
                        int red = sk > 1 ? (1 << rs) >> 1 : 0;   // 0 (none) only for sk 1; bits 1, 2, 4 for split-k
                        if (sk > 1 && (red == 0 || !(redmask & red))) continue;
                        cublasLtMatmulAlgo_t a2 = al;
                        cublasLtMatmulAlgoConfigSetAttribute(&a2, CUBLASLT_ALGO_CONFIG_TILE_ID, &t, 4);
                        cublasLtMatmulAlgoConfigSetAttribute(&a2, CUBLASLT_ALGO_CONFIG_STAGES_ID, &sg, 4);
                        cublasLtMatmulAlgoConfigSetAttribute(&a2, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &cu, 4);
                        cublasLtMatmulAlgoConfigSetAttribute(&a2, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &sw, 4);
                        cublasLtMatmulAlgoConfigSetAttribute(&a2, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, 4);
                        cublasLtMatmulAlgoConfigSetAttribute(&a2, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, 4);
                        cublasLtMatmulHeuristicResult_t hr;
                        if (cublasLtMatmulAlgoCheck(c->lt, p->op, p->la, p->lb, p->lc, p->lc, &a2, &hr) != CUBLAS_STATUS_SUCCESS) continue;
                        if (hr.workspaceSize > LT_WS) continue;
                        xalgos.push_back(a2); xws.push_back(hr.workspaceSize);
                    }
            }
            // quick pass: 2 reps each; keep the 16 fastest
            std::vector<std::pair<float, int>> qt;
            auto qtime = [&](int i) {
                auto f = [&] { return cublasLtMatmul(c->lt, p->op, &alpha, Bm, p->la, A, p->lb, &beta, Cp, p->lc, Cp, p->lc, &xalgos[i], ws, xws[i], st) == CUBLAS_STATUS_SUCCESS; };
                if (!f()) return 1e30f;
                CK(cudaEventRecord(e0, st)); f(); f(); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return ms / 2;
            };
            for (int i = 0; i < (int)xalgos.size(); i++) qt.push_back({qtime(i), i});
            cudaGetLastError();
            std::sort(qt.begin(), qt.end());
            printf("   (exhaustive: %zu configurations)\n", xalgos.size());
            for (int q = 0; q < (int)qt.size() && q < 16; q++) {
                int i = qt[q].second;
                cublasLtMatmulAlgo_t r = xalgos[i];
                size_t w = xws[i];
                cands.push_back({"exh " + desc(r), [&, r, w] {
                    return cublasLtMatmul(c->lt, p->op, &alpha, Bm, p->la, A, p->lb, &beta, Cp, p->lc, Cp, p->lc, &r, ws, w, st) == CUBLAS_STATUS_SUCCESS; }});
            }
        }
        // warm-up: ~300 ms of the default kernel (steady clocks)
        {
            auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.3) { cands[1].f(); CK(cudaStreamSynchronize(st)); }
        }
        // careful pass: rounds of single calls, the candidates in a fresh random order each round, so that all of them see
        // the same mix of clock states (the GPU runs at its power cap here); a candidate's time is its median over rounds
        {
            int nc = (int)cands.size();
            std::vector<cudaEvent_t> eb(nc), ee(nc);
            for (int i = 0; i < nc; i++) { CK(cudaEventCreate(&eb[i])); CK(cudaEventCreate(&ee[i])); }
            std::vector<std::vector<float>> tt(nc);
            std::vector<int> order(nc);
            for (int i = 0; i < nc; i++) order[i] = i;
            uint64_t rs = 12345;
            for (int i = 0; i < nc; i++) if (!cands[i].f()) order[i] = -1;   // launch failures drop out
            cudaGetLastError();
            for (int r = 0; r < rounds * reps; r++) {
                for (int i = nc - 1; i > 0; i--) { rs = rs * 6364136223846793005ull + 1442695040888963407ull; std::swap(order[i], order[(rs >> 33) % (i + 1)]); }
                for (int i : order) { if (i < 0) continue; CK(cudaEventRecord(eb[i], st)); cands[i].f(); CK(cudaEventRecord(ee[i], st)); }
                CK(cudaStreamSynchronize(st));
                for (int i : order) { if (i < 0) continue; float ms; CK(cudaEventElapsedTime(&ms, eb[i], ee[i])); tt[i].push_back(ms); }
            }
            for (int i = 0; i < nc; i++) {
                if (tt[i].empty()) continue;
                std::sort(tt[i].begin(), tt[i].end());
                cands[i].best = tt[i][tt[i].size() / 2];
                cudaEventDestroy(eb[i]); cudaEventDestroy(ee[i]);
            }
        }
        cudaGetLastError();
        double flop = 2.0 * s.M * s.N * s.K;
        printf("== %s: M=%d N=%d K=%d, C %s %d-byte aligned; %d heuristic candidates (lib rule: %d)\n", s.name, s.M, s.N, s.K,
               s.f32out ? "fp32" : "bf16", alC, n, n2);
        std::vector<int> ord(cands.size());
        for (size_t i = 0; i < ord.size(); i++) ord[i] = (int)i;
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return cands[a].best < cands[b].best; });
        int shown = 0;
        for (int i : ord) {
            bool key = i < 2;
            if (!key && shown >= 8) continue;
            if (!key) shown++;
            printf("  %8.3f ms %6.1f TFLOPS  %s\n", cands[i].best, flop / cands[i].best / 1e9, cands[i].nm.c_str());
        }
        cublasLtMatmulPreferenceDestroy(pref); cublasLtMatmulPreferenceDestroy(pref2);
    }
    return 0;
}
