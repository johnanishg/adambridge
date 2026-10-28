/* Standalone checks of the native kernels.
 *
 * Includes the kernel source so internal pieces (FP8 codec) can be tested
 * directly. Build and run:
 *   cc -O2 -pthread -o selftest tests/native/selftest.c -lm && ./selftest
 *
 * 1. every SIMD path matches the scalar path (all dtype combinations, odd
 *    lengths, split points);
 * 2. FP8 encode is exact round-to-nearest-even (checked against a brute-force
 *    search over all codes) and stochastic rounding is unbiased;
 * 3. multi-tensor updates are bit-identical for any thread count, FP8 included.
 */
#include "../../adamx/csrc/adamx_kernels.c"

#include <stdio.h>

static uint32_t rs = 12345;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) / 16777216.0f) * 2.0f - 1.0f; }
static uint16_t to_bf16(float f) { uint32_t b; memcpy(&b, &f, 4); return (uint16_t)((b + 0x7fff + ((b >> 16) & 1)) >> 16); }
static float from_bf16(uint16_t h) { uint32_t b = (uint32_t)h << 16; float f; memcpy(&f, &b, 4); return f; }

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++failures; printf("FAIL "); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* ------------------------------------------------------------------------- */
/* 1. SIMD vs scalar, single-call API                                        */
/* ------------------------------------------------------------------------- */

typedef struct { float* p; void* g; void* m; void* v; void* out; } bufs_t;

static void* alloc_dt(int dt, int64_t n) { return dt == 0 ? NULL : malloc((size_t)n * (dt == 1 ? 4 : 2)); }

static void fill(bufs_t* b, int gt, int mvt, int64_t n)
{
    rs = 777;
    for (int64_t i = 0; i < n; ++i) {
        float p = frand(), g = frand() * 0.1f, m = frand() * 0.01f, v = fabsf(frand()) * 1e-3f;
        b->p[i] = p;
        if (gt == 1) ((float*)b->g)[i] = g; else ((uint16_t*)b->g)[i] = to_bf16(g);
        if (mvt == 1) { ((float*)b->m)[i] = m; ((float*)b->v)[i] = v; }
        else { ((uint16_t*)b->m)[i] = to_bf16(m); ((uint16_t*)b->v)[i] = to_bf16(v); }
    }
}

static float get(const void* base, int dt, int64_t i) { return dt == 1 ? ((const float*)base)[i] : from_bf16(((const uint16_t*)base)[i]); }

/* Mixed tolerance: values that cancel to ~0 are compared against the
 * array's magnitude rather than their own. */
static double max_rel(const void* a, const void* b, int dt, int64_t n)
{
    double worst = 0, scale = 0;
    for (int64_t i = 0; i < n; ++i) scale = fmax(scale, fabs(get(b, dt, i)));
    for (int64_t i = 0; i < n; ++i) {
        double x = get(a, dt, i), y = get(b, dt, i);
        double d = fabs(x - y) / (fabs(y) + 1e-2 * scale + 1e-30);
        if (d > worst) worst = d;
    }
    return worst;
}

static adamx_hparams test_hparams(int sr)
{
    adamx_hparams h = { 0.9f, 0.999f, 0.1f, 0.001f, 1.0f - 1e-3f * 0.01f, 1e-3f / 0.1f, 1.0f / sqrtf(0.001f), 1e-8f, 0xC0FFEEu, sr };
    return h;
}

static void simd_case(int level, int gt, int mvt, int ot, int sr, int64_t n, int split)
{
    bufs_t ref = { malloc(n * 4), alloc_dt(gt, n), alloc_dt(mvt, n), alloc_dt(mvt, n), alloc_dt(ot, n) };
    bufs_t tst = { malloc(n * 4), alloc_dt(gt, n), alloc_dt(mvt, n), alloc_dt(mvt, n), alloc_dt(ot, n) };
    fill(&ref, gt, mvt, n);
    fill(&tst, gt, mvt, n);
    adamx_hparams h = test_hparams(sr);

    adamx_set_simd_level(0);
    adamx_fused_adamw(ref.p, ref.g, gt, ref.m, ref.v, mvt, ref.out, ot, n, 0, &h);
    int actual = adamx_set_simd_level(level);
    const int64_t a = split ? (n / 3) & ~(int64_t)15 : n;
    size_t ms = mvt == 1 ? 4 : 2, gs = gt == 1 ? 4 : 2, os = ot == 1 ? 4 : 2;
    adamx_fused_adamw(tst.p, tst.g, gt, tst.m, tst.v, mvt, tst.out, ot, a, 0, &h);
    if (a < n)
        adamx_fused_adamw(tst.p + a, (char*)tst.g + a * gs, gt, (char*)tst.m + a * ms, (char*)tst.v + a * ms, mvt,
                          tst.out ? (char*)tst.out + a * os : NULL, ot, n - a, (uint64_t)a, &h);

    /* FMA vs separate mul/add may differ by a few ulp; BF16 storage may then
     * round to a neighbouring value (one bf16 ulp = 2^-8 relative). */
    const double tol32 = 1e-5, tol16 = 1.0 / 128;
    double ep = max_rel(tst.p, ref.p, 1, n);
    double em = max_rel(tst.m, ref.m, mvt, n);
    double ev = max_rel(tst.v, ref.v, mvt, n);
    double eo = ot ? max_rel(tst.out, ref.out, ot, n) : 0;
    CHECK(ep < tol32 && em < (mvt == 1 ? tol32 : tol16) && ev < (mvt == 1 ? tol32 : tol16) && eo < (ot == 1 ? tol32 : tol16),
          "simd level=%d g=%d mv=%d out=%d sr=%d n=%lld split=%d  p=%.2e m=%.2e v=%.2e out=%.2e",
          actual, gt, mvt, ot, sr, (long long)n, split, ep, em, ev, eo);
    free(ref.p); free(ref.g); free(ref.m); free(ref.v); free(ref.out);
    free(tst.p); free(tst.g); free(tst.m); free(tst.v); free(tst.out);
}

/* ------------------------------------------------------------------------- */
/* 2. FP8 codec                                                              */
/* ------------------------------------------------------------------------- */

typedef struct { int mbits, bias; float fmax; uint32_t nan_code; const float* lut; const char* name; } fmt_t;

/* Reference RNE: the finite code with the same sign whose value is nearest,
 * ties to the even code (codes of one sign are ordered by magnitude). */
static uint32_t brute_rne(const fmt_t* f, float x)
{
    const uint32_t sign = x < 0 || (x == 0 && signbit(x)) ? 0x80u : 0u;
    float a = fabsf(x);
    if (a > f->fmax) a = f->fmax;
    uint32_t best = 0;
    double best_d = INFINITY;
    for (uint32_t c = 0; c < 128; ++c) {
        const float v = f->lut[c];
        if (!isfinite(v)) continue;
        const double d = fabs((double)v - (double)a);
        if (d < best_d || (d == best_d && (c & 1u) == 0)) { best_d = d; best = c; }
    }
    return sign | best;
}

static void codec_checks(void)
{
    const fmt_t fmts[2] = {
        { 3, 7, 448.0f, 0x7fu, g_lut_e4m3, "e4m3" },
        { 2, 15, 57344.0f, 0x7eu, g_lut_e5m2, "e5m2" },
    };
    for (int fi = 0; fi < 2; ++fi) {
        const fmt_t* f = &fmts[fi];
        int bad = 0;
        for (uint32_t c = 0; c < 256; ++c) {          /* every finite code round-trips */
            const float v = f->lut[c];
            if (!isfinite(v)) continue;
            if (fp8_encode(v, 0, 0, f->mbits, f->bias, f->fmax, f->nan_code) != c) ++bad;
        }
        CHECK(bad == 0, "%s: %d codes do not round-trip", f->name, bad);
        bad = 0;
        rs = 99;
        for (int i = 0; i < 200000; ++i) {            /* RNE matches brute force across the whole range */
            const float e = (float)(i % 40) - 24.0f;  /* exponents spanning subnormals .. max */
            const float x = frand() * ldexpf(1.0f, (int)e);
            if (fp8_encode(x, 0, 0, f->mbits, f->bias, f->fmax, f->nan_code) != brute_rne(f, x)) ++bad;
        }
        CHECK(bad == 0, "%s: %d of 200000 values not rounded to nearest-even", f->name, bad);
        CHECK(fp8_encode(NAN, 0, 0, f->mbits, f->bias, f->fmax, f->nan_code) == f->nan_code, "%s: NaN code", f->name);
        CHECK(f->lut[fp8_encode(1e30f, 0, 0, f->mbits, f->bias, f->fmax, f->nan_code)] == f->fmax, "%s: saturation", f->name);

        /* Stochastic rounding is unbiased between two neighbouring codes. */
        const float lo_v = f->lut[0x30], hi_v = f->lut[0x31];
        const float x = lo_v + 0.3f * (hi_v - lo_v);
        double sum = 0;
        const int draws = 400000;
        for (int i = 0; i < draws; ++i) sum += f->lut[fp8_encode(x, 1, hash32((uint32_t)i * 2654435761u), f->mbits, f->bias, f->fmax, f->nan_code)];
        const double mean = sum / draws;
        CHECK(fabs(mean - x) < 0.01 * (hi_v - lo_v), "%s: SR mean %.6g for %.6g", f->name, mean, (double)x);
    }
}

/* ------------------------------------------------------------------------- */
/* 3. Multi-tensor updates: thread-count invariance                          */
/* ------------------------------------------------------------------------- */

typedef struct {
    adamx_segment seg;
    int64_t nblocks;
} tseg_t;

static void make_segment(tseg_t* t, int64_t n, int gt, int mvt, int ot, uint32_t seed)
{
    adamx_segment* s = &t->seg;
    memset(s, 0, sizeof(*s));
    s->n = n;
    s->g_dtype = gt;
    s->mv_dtype = mvt;
    s->out_dtype = ot;
    t->nblocks = (n + FP8_BLOCK - 1) / FP8_BLOCK;
    s->p = (float*)malloc((size_t)n * 4);
    s->g = malloc((size_t)n * (gt == DT_F32 ? 4 : 2));
    const size_t mb = mvt == DT_F32 ? 4 : (mvt == DT_BF16 ? 2 : 1);
    const int64_t mlen = mvt == DT_FP8 ? t->nblocks * FP8_BLOCK : n;
    s->m = calloc((size_t)mlen, mb);
    s->v = calloc((size_t)mlen, mb);
    if (mvt == DT_FP8) {
        s->m_scale = (float*)malloc((size_t)t->nblocks * 4);
        s->v_scale = (float*)malloc((size_t)t->nblocks * 4);
        for (int64_t b = 0; b < t->nblocks; ++b) { s->m_scale[b] = 1.0f; s->v_scale[b] = 1.0f; }
    }
    s->out = ot ? malloc((size_t)n * (ot == DT_F32 ? 4 : 2)) : NULL;
    rs = seed;
    for (int64_t i = 0; i < n; ++i) {
        s->p[i] = frand();
        const float g = frand() * 0.1f;
        if (gt == DT_F32) ((float*)s->g)[i] = g; else ((uint16_t*)s->g)[i] = to_bf16(g);
    }
}

static void free_segment(tseg_t* t)
{
    adamx_segment* s = &t->seg;
    free(s->p); free((void*)s->g); free(s->m); free(s->v); free(s->m_scale); free(s->v_scale); free(s->out);
}

static int same_bytes(const void* a, const void* b, size_t n) { return memcmp(a, b, n) == 0; }

static void multi_case(int threads)
{
    /* A mix of sizes (tiny, unaligned, larger than one piece) and dtypes. */
    const int64_t sizes[] = { 1, 37, 128, 1000, 70001, 300000 };
    const int mvts[] = { DT_F32, DT_BF16, DT_FP8 };
    enum { NS = 6 * 3 };
    tseg_t a[NS], b[NS];
    adamx_segment* pa[NS];
    adamx_segment* pb[NS];
    adamx_hparams hps[NS];
    int k = 0;
    for (int mi = 0; mi < 3; ++mi)
        for (int si = 0; si < 6; ++si, ++k) {
            const int gt = (k % 2) ? DT_BF16 : DT_F32;
            const int ot = (k % 3) == 0 ? DT_NONE : ((k % 3) == 1 ? DT_F32 : DT_BF16);
            make_segment(&a[k], sizes[si], gt, mvts[mi], ot, 1000u + (uint32_t)k);
            make_segment(&b[k], sizes[si], gt, mvts[mi], ot, 1000u + (uint32_t)k);
            pa[k] = &a[k].seg;
            pb[k] = &b[k].seg;
            hps[k] = test_hparams(1);
            hps[k].seed = 77u + (uint32_t)k;
        }
    void* pool = adamx_pool_create(threads, NULL, 0);
    for (int step = 0; step < 3; ++step) {           /* several steps: FP8 scales evolve */
        CHECK(adamx_multi_apply(NULL, pa, hps, NS, 1) == 0, "multi_apply single");
        CHECK(adamx_multi_apply(pool, pb, hps, NS, threads) == 0, "multi_apply pool");
    }
    adamx_pool_destroy(pool);
    int diff = 0;
    for (int i = 0; i < NS; ++i) {
        const adamx_segment* x = pa[i];
        const adamx_segment* y = pb[i];
        const size_t n = (size_t)x->n;
        const size_t mb = x->mv_dtype == DT_F32 ? 4 : (x->mv_dtype == DT_BF16 ? 2 : 1);
        const size_t mlen = x->mv_dtype == DT_FP8 ? (size_t)a[i].nblocks * FP8_BLOCK : n;
        int ok = same_bytes(x->p, y->p, n * 4) && same_bytes(x->m, y->m, mlen * mb) && same_bytes(x->v, y->v, mlen * mb);
        if (x->mv_dtype == DT_FP8)
            ok = ok && same_bytes(x->m_scale, y->m_scale, (size_t)a[i].nblocks * 4) && same_bytes(x->v_scale, y->v_scale, (size_t)a[i].nblocks * 4);
        if (x->out_dtype) ok = ok && same_bytes(x->out, y->out, n * (x->out_dtype == DT_F32 ? 4 : 2));
        if (!ok) { ++diff; printf("  segment %d (n=%lld, mv=%d) differs\n", i, (long long)n, x->mv_dtype); }
    }
    CHECK(diff == 0, "threads=%d: %d segments differ from the single-threaded result", threads, diff);
    for (int i = 0; i < NS; ++i) { free_segment(&a[i]); free_segment(&b[i]); }
}

/* A job posted right after the pool is created, before its helpers have
 * started: every helper must still take part (or the caller waits forever). */
static void fresh_pool_case(void)
{
    tseg_t t;
    make_segment(&t, 200000, DT_F32, DT_F32, DT_NONE, 3);   /* several work items */
    adamx_segment* ps = &t.seg;
    adamx_hparams h = test_hparams(0);
    for (int i = 0; i < 300; ++i) {
        void* pool = adamx_pool_create(2 + i % 7, NULL, 0);
        CHECK(adamx_multi_apply(pool, &ps, &h, 1, 2 + i % 7) == 0, "fresh pool %d", i);
        adamx_pool_destroy(pool);
    }
    free_segment(&t);
}

/* One tensor updated whole vs as chunks (segments with an element offset):
 * bit-identical, including stochastic rounding and FP8 block scales. */
static void chunk_case(int mvt)
{
    const int64_t n = 300001;
    const int64_t cuts[] = { 0, 65536, 131072 + 4096, n };   /* 128-aligned, uneven */
    tseg_t a, b;
    make_segment(&a, n, DT_BF16, mvt, DT_BF16, 9);
    make_segment(&b, n, DT_BF16, mvt, DT_BF16, 9);
    adamx_segment* pa = &a.seg;
    adamx_segment chunks[3];
    adamx_segment* pc[3];
    const size_t mb = mvt == DT_F32 ? 4 : (mvt == DT_BF16 ? 2 : 1);
    for (int c = 0; c < 3; ++c) {
        const int64_t lo = cuts[c];
        chunks[c] = b.seg;
        chunks[c].p = b.seg.p + lo;
        chunks[c].g = (const uint16_t*)b.seg.g + lo;
        chunks[c].m = (uint8_t*)b.seg.m + lo * (int64_t)mb;
        chunks[c].v = (uint8_t*)b.seg.v + lo * (int64_t)mb;
        chunks[c].out = (uint16_t*)b.seg.out + lo;
        if (mvt == DT_FP8) {
            chunks[c].m_scale = b.seg.m_scale + lo / FP8_BLOCK;
            chunks[c].v_scale = b.seg.v_scale + lo / FP8_BLOCK;
        }
        chunks[c].n = cuts[c + 1] - lo;
        chunks[c].offset = lo;
        pc[c] = &chunks[c];
    }
    adamx_hparams h = test_hparams(1);
    adamx_hparams hs[3] = { h, h, h };
    for (int step = 0; step < 3; ++step) {
        CHECK(adamx_multi_apply(NULL, &pa, &h, 1, 1) == 0, "chunk_case whole");
        CHECK(adamx_multi_apply(NULL, pc, hs, 3, 1) == 0, "chunk_case chunks");
    }
    const size_t mlen = mvt == DT_FP8 ? (size_t)a.nblocks * FP8_BLOCK : (size_t)n;
    int ok = same_bytes(a.seg.p, b.seg.p, (size_t)n * 4) && same_bytes(a.seg.out, b.seg.out, (size_t)n * 2)
             && same_bytes(a.seg.m, b.seg.m, mlen * mb) && same_bytes(a.seg.v, b.seg.v, mlen * mb);
    if (mvt == DT_FP8)
        ok = ok && same_bytes(a.seg.m_scale, b.seg.m_scale, (size_t)a.nblocks * 4)
                && same_bytes(a.seg.v_scale, b.seg.v_scale, (size_t)a.nblocks * 4);
    CHECK(ok, "moments %d: chunked update differs from the whole-tensor update", mvt);
    chunks[1].offset = mvt == DT_FP8 ? 100 : -1;            /* misaligned / negative: rejected */
    CHECK(adamx_multi_apply(NULL, &pc[1], hs, 1, 1) == -11, "moments %d: bad offset accepted", mvt);
    free_segment(&a);
    free_segment(&b);
}

/* FP8 through each SIMD level vs scalar: same codes for the overwhelming
 * majority (FMA can move a pre-rounding value across a rounding boundary). */
static void fp8_simd_case(int level)
{
    tseg_t a, b;
    const int64_t n = 50000;
    make_segment(&a, n, DT_BF16, DT_FP8, DT_BF16, 5);
    make_segment(&b, n, DT_BF16, DT_FP8, DT_BF16, 5);
    adamx_segment* pa = &a.seg;
    adamx_segment* pb = &b.seg;
    adamx_hparams h = test_hparams(1);
    adamx_set_simd_level(0);
    for (int step = 0; step < 3; ++step) adamx_multi_apply(NULL, &pa, &h, 1, 1);
    const int actual = adamx_set_simd_level(level);
    for (int step = 0; step < 3; ++step) adamx_multi_apply(NULL, &pb, &h, 1, 1);
    adamx_set_simd_level(-1);
    int64_t same_m = 0, same_v = 0;
    for (int64_t i = 0; i < n; ++i) {
        same_m += ((uint8_t*)pa->m)[i] == ((uint8_t*)pb->m)[i];
        same_v += ((uint8_t*)pa->v)[i] == ((uint8_t*)pb->v)[i];
    }
    double ep = max_rel(pb->p, pa->p, 1, n);
    CHECK(ep < 1e-5 && same_m > n * 0.99 && same_v > n * 0.99,
          "fp8 level=%d: p=%.2e, m codes equal %.4f, v codes equal %.4f", actual, ep, (double)same_m / n, (double)same_v / n);
    free_segment(&a);
    free_segment(&b);
}

int main(void)
{
    const int det = adamx_detected_simd_level();
    printf("detected SIMD level: %d\n", det);
    int levels[3] = { 0, 1, 2 };
    int nlev = det == 2 ? 3 : (det == 1 ? 2 : 1);
    if (det == 3) { levels[1] = 3; nlev = 2; }

    int cases = 0;
    const int64_t sizes[3] = { 1, 37, 100003 };
    for (int li = 0; li < nlev; ++li)
        for (int gt = 1; gt <= 2; ++gt)
            for (int mvt = 1; mvt <= 2; ++mvt)
                for (int ot = 0; ot <= 2; ++ot)
                    for (int sr = 0; sr <= (mvt == 2); ++sr)
                        for (int si = 0; si < 3; ++si)
                            for (int split = 0; split <= 1; ++split, ++cases)
                                simd_case(levels[li], gt, mvt, ot, sr, sizes[si], split);
    adamx_set_simd_level(-1);
    printf("simd vs scalar: %d cases\n", cases);

    codec_checks();
    printf("fp8 codec: round-trip, nearest-even vs brute force, stochastic rounding\n");

    for (int li = 1; li < nlev; ++li) fp8_simd_case(levels[li]);
    printf("fp8 simd vs scalar: %d levels\n", nlev - 1);

    const int thread_counts[] = { 2, 3, 8 };
    for (int i = 0; i < 3; ++i) multi_case(thread_counts[i]);
    printf("multi-tensor thread invariance: 2, 3, 8 threads\n");

    fresh_pool_case();
    printf("pools used right after creation: 300\n");

    for (int mvt = DT_F32; mvt <= DT_FP8; ++mvt) chunk_case(mvt);
    printf("chunked updates (segment offsets): fp32, bf16, fp8 moments\n");

    printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures != 0;
}
