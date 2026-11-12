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
#include "../../adambridge/csrc/adambridge_kernels.c"

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

static adambridge_hparams test_hparams(int sr)
{
    adambridge_hparams h = { 0.9f, 0.999f, 0.1f, 0.001f, 1.0f - 1e-3f * 0.01f, 1e-3f / 0.1f, 1.0f / sqrtf(0.001f), 1e-8f, 0xC0FFEEu, sr };
    return h;
}

static void simd_case(int level, int gt, int mvt, int ot, int sr, int64_t n, int split)
{
    bufs_t ref = { malloc(n * 4), alloc_dt(gt, n), alloc_dt(mvt, n), alloc_dt(mvt, n), alloc_dt(ot, n) };
    bufs_t tst = { malloc(n * 4), alloc_dt(gt, n), alloc_dt(mvt, n), alloc_dt(mvt, n), alloc_dt(ot, n) };
    fill(&ref, gt, mvt, n);
    fill(&tst, gt, mvt, n);
    adambridge_hparams h = test_hparams(sr);

    adambridge_set_simd_level(0);
    adambridge_fused_adamw(ref.p, ref.g, gt, ref.m, ref.v, mvt, ref.out, ot, n, 0, &h);
    int actual = adambridge_set_simd_level(level);
    const int64_t a = split ? (n / 3) & ~(int64_t)15 : n;
    size_t ms = mvt == 1 ? 4 : 2, gs = gt == 1 ? 4 : 2, os = ot == 1 ? 4 : 2;
    adambridge_fused_adamw(tst.p, tst.g, gt, tst.m, tst.v, mvt, tst.out, ot, a, 0, &h);
    if (a < n)
        adambridge_fused_adamw(tst.p + a, (char*)tst.g + a * gs, gt, (char*)tst.m + a * ms, (char*)tst.v + a * ms, mvt,
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
    adambridge_segment seg;
    int64_t nblocks;
} tseg_t;

static void make_segment_pt(tseg_t* t, int64_t n, int gt, int mvt, int ot, uint32_t seed, int pt)
{
    adambridge_segment* s = &t->seg;
    memset(s, 0, sizeof(*s));
    s->n = n;
    s->g_dtype = gt;
    s->mv_dtype = mvt;
    s->out_dtype = ot;
    s->p_dtype = pt;
    t->nblocks = (n + FP8_BLOCK - 1) / FP8_BLOCK;
    s->p = malloc((size_t)n * (pt == DT_F24 ? 2 : 4));
    s->p_lo = pt == DT_F24 ? (uint8_t*)malloc((size_t)n) : NULL;
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
        store_master(s->p, s->p_lo, pt, i, frand(), 0, 0);   /* FP24: rounded to nearest even */
        const float g = frand() * 0.1f;
        if (gt == DT_F32) ((float*)s->g)[i] = g; else ((uint16_t*)s->g)[i] = to_bf16(g);
    }
}

static void make_segment(tseg_t* t, int64_t n, int gt, int mvt, int ot, uint32_t seed)
{
    make_segment_pt(t, n, gt, mvt, ot, seed, DT_F32);
}

static void free_segment(tseg_t* t)
{
    adambridge_segment* s = &t->seg;
    free(s->p); free(s->p_lo); free((void*)s->g); free(s->m); free(s->v); free(s->m_scale); free(s->v_scale); free(s->out);
}

static int same_bytes(const void* a, const void* b, size_t n) { return memcmp(a, b, n) == 0; }

static int master_same(const adambridge_segment* x, const adambridge_segment* y)
{
    const size_t n = (size_t)x->n;
    if (x->p_dtype == DT_F24) return same_bytes(x->p, y->p, n * 2) && same_bytes(x->p_lo, y->p_lo, n);
    return same_bytes(x->p, y->p, n * 4);
}

/* A chunk of segment s starting at element lo (its buffers advanced, offset set). */
static adambridge_segment chunk_of(const adambridge_segment* s, int64_t lo, int64_t hi)
{
    adambridge_segment c = *s;
    const size_t mb = s->mv_dtype == DT_F32 ? 4 : (s->mv_dtype == DT_BF16 ? 2 : 1);
    c.p = (char*)s->p + lo * (s->p_dtype == DT_F24 ? 2 : 4);
    c.p_lo = s->p_lo ? s->p_lo + lo : NULL;
    c.g = (const char*)s->g + lo * (s->g_dtype == DT_F32 ? 4 : 2);
    c.m = (uint8_t*)s->m + lo * (int64_t)mb;
    c.v = s->v ? (uint8_t*)s->v + lo * (int64_t)mb : NULL;
    c.out = s->out ? (char*)s->out + lo * (s->out_dtype == DT_F32 ? 4 : 2) : NULL;
    if (s->mv_dtype == DT_FP8) {
        c.m_scale = s->m_scale + lo / FP8_BLOCK;
        c.v_scale = s->v_scale ? s->v_scale + lo / FP8_BLOCK : NULL;
    }
    c.n = hi - lo;
    c.offset = lo;
    return c;
}

static void multi_case(int threads)
{
    /* A mix of sizes (tiny, unaligned, larger than one piece) and dtypes. */
    const int64_t sizes[] = { 1, 37, 128, 1000, 70001, 300000 };
    const int mvts[] = { DT_F32, DT_BF16, DT_FP8 };
    enum { NS = 6 * 3 };
    tseg_t a[NS], b[NS];
    adambridge_segment* pa[NS];
    adambridge_segment* pb[NS];
    adambridge_hparams hps[NS];
    int k = 0;
    for (int mi = 0; mi < 3; ++mi)
        for (int si = 0; si < 6; ++si, ++k) {
            const int gt = (k % 2) ? DT_BF16 : DT_F32;
            const int ot = (k % 3) == 0 ? DT_NONE : ((k % 3) == 1 ? DT_F32 : DT_BF16);
            const int pt = ((k / 2) % 2) ? DT_F24 : DT_F32;
            make_segment_pt(&a[k], sizes[si], gt, mvts[mi], ot, 1000u + (uint32_t)k, pt);
            make_segment_pt(&b[k], sizes[si], gt, mvts[mi], ot, 1000u + (uint32_t)k, pt);
            pa[k] = &a[k].seg;
            pb[k] = &b[k].seg;
            hps[k] = test_hparams(1);
            hps[k].seed = 77u + (uint32_t)k;
        }
    void* pool = adambridge_pool_create(threads, NULL, 0);
    for (int step = 0; step < 3; ++step) {           /* several steps: FP8 scales evolve */
        CHECK(adambridge_multi_apply(NULL, pa, hps, NS, 1) == 0, "multi_apply single");
        CHECK(adambridge_multi_apply(pool, pb, hps, NS, threads) == 0, "multi_apply pool");
    }
    adambridge_pool_destroy(pool);
    int diff = 0;
    for (int i = 0; i < NS; ++i) {
        const adambridge_segment* x = pa[i];
        const adambridge_segment* y = pb[i];
        const size_t n = (size_t)x->n;
        const size_t mb = x->mv_dtype == DT_F32 ? 4 : (x->mv_dtype == DT_BF16 ? 2 : 1);
        const size_t mlen = x->mv_dtype == DT_FP8 ? (size_t)a[i].nblocks * FP8_BLOCK : n;
        int ok = master_same(x, y) && same_bytes(x->m, y->m, mlen * mb) && same_bytes(x->v, y->v, mlen * mb);
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
    adambridge_segment* ps = &t.seg;
    adambridge_hparams h = test_hparams(0);
    for (int i = 0; i < 300; ++i) {
        void* pool = adambridge_pool_create(2 + i % 7, NULL, 0);
        CHECK(adambridge_multi_apply(pool, &ps, &h, 1, 2 + i % 7) == 0, "fresh pool %d", i);
        adambridge_pool_destroy(pool);
    }
    free_segment(&t);
}

/* One tensor updated whole vs as chunks (segments with an element offset):
 * bit-identical, including stochastic rounding and FP8 block scales. */
static void chunk_case(int mvt, int pt)
{
    const int64_t n = 300001;
    const int64_t cuts[] = { 0, 65536, 131072 + 4096, n };   /* 128-aligned, uneven */
    tseg_t a, b;
    make_segment_pt(&a, n, DT_BF16, mvt, DT_BF16, 9, pt);
    make_segment_pt(&b, n, DT_BF16, mvt, DT_BF16, 9, pt);
    adambridge_segment* pa = &a.seg;
    adambridge_segment chunks[3];
    adambridge_segment* pc[3];
    const size_t mb = mvt == DT_F32 ? 4 : (mvt == DT_BF16 ? 2 : 1);
    for (int c = 0; c < 3; ++c) {
        chunks[c] = chunk_of(&b.seg, cuts[c], cuts[c + 1]);
        pc[c] = &chunks[c];
    }
    adambridge_hparams h = test_hparams(1);
    adambridge_hparams hs[3] = { h, h, h };
    for (int step = 0; step < 3; ++step) {
        CHECK(adambridge_multi_apply(NULL, &pa, &h, 1, 1) == 0, "chunk_case whole");
        CHECK(adambridge_multi_apply(NULL, pc, hs, 3, 1) == 0, "chunk_case chunks");
    }
    const size_t mlen = mvt == DT_FP8 ? (size_t)a.nblocks * FP8_BLOCK : (size_t)n;
    int ok = master_same(&a.seg, &b.seg) && same_bytes(a.seg.out, b.seg.out, (size_t)n * 2)
             && same_bytes(a.seg.m, b.seg.m, mlen * mb) && same_bytes(a.seg.v, b.seg.v, mlen * mb);
    if (mvt == DT_FP8)
        ok = ok && same_bytes(a.seg.m_scale, b.seg.m_scale, (size_t)a.nblocks * 4)
                && same_bytes(a.seg.v_scale, b.seg.v_scale, (size_t)a.nblocks * 4);
    CHECK(ok, "moments %d master %d: chunked update differs from the whole-tensor update", mvt, pt);
    chunks[1].offset = mvt == DT_FP8 ? 100 : -1;            /* misaligned / negative: rejected */
    CHECK(adambridge_multi_apply(NULL, &pc[1], hs, 1, 1) == -11, "moments %d: bad offset accepted", mvt);
    free_segment(&a);
    free_segment(&b);
}

/* ------------------------------------------------------------------------- */
/* FP24 master weights                                                       */
/* ------------------------------------------------------------------------- */

/* (a) the rounding helper: nearest-even matches a brute-force choice between the two 24-bit neighbours, and
 * stochastic rounding is unbiased; (b) one AdamW step with an FP24 master lands within 2^-14 of the FP32 result
 * and the accelerator copy is the stored value rounded to BF16; (c) each SIMD level matches the scalar path;
 * (d) bad segments are rejected. */
static void f24_case(int nlev, const int* levels)
{
    int bad = 0;
    rs = 2024;
    for (int i = 0; i < 200000; ++i) {
        const float x = frand() * ldexpf(1.0f, i % 50 - 25);
        const uint32_t b = f32_bits(x);
        const uint32_t down = b & ~0xffU, up = down + 0x100U;
        const double dd = fabs((double)x - (double)bits_f32(down)), du = fabs((double)bits_f32(up) - (double)x);
        const uint32_t want = dd < du ? down : (du < dd ? up : ((down & 0x100U) ? up : down));
        bad += f24_round_bits(b, 0, 0) != want;
    }
    CHECK(bad == 0, "fp24 nearest-even: %d of 200000 wrong", bad);
    CHECK(f24_round_bits(f32_bits(NAN), 0, 0) != 0 && isnan(bits_f32(f24_round_bits(f32_bits(NAN), 1, 0x80))), "fp24 NaN");
    {
        const uint32_t b = f32_bits(1.5f) | 0x40U;            /* a quarter of the way to the next 24-bit value */
        const double x = (double)bits_f32(b), spacing = (double)bits_f32((b & ~0xffU) + 0x100U) - (double)bits_f32(b & ~0xffU);
        double sum = 0;
        const int draws = 400000;
        for (int i = 0; i < draws; ++i) sum += bits_f32(f24_round_bits(b, 1, hash32((uint32_t)i * 2654435761u)));
        CHECK(fabs(sum / draws - x) < 0.01 * spacing, "fp24 SR mean %.9g for %.9g", sum / draws, x);
    }

    const int64_t n = 100003;
    for (int sr = 0; sr <= 1; ++sr) {
        tseg_t f32, f24;
        make_segment(&f32, n, DT_BF16, DT_BF16, DT_BF16, 11);
        make_segment_pt(&f24, n, DT_BF16, DT_BF16, DT_BF16, 11, DT_F24);
        adambridge_segment* p32 = &f32.seg;
        adambridge_segment* p24 = &f24.seg;
        adambridge_hparams h = test_hparams(sr);
        CHECK(adambridge_multi_apply(NULL, &p32, &h, 1, 1) == 0 && adambridge_multi_apply(NULL, &p24, &h, 1, 1) == 0,
              "fp24 step rejected");
        double worst = 0;
        int out_bad = 0;
        for (int64_t i = 0; i < n; ++i) {
            const float stored = load_master(f24.seg.p, f24.seg.p_lo, DT_F24, i);
            const float ref = ((const float*)f32.seg.p)[i];
            const double rel = fabs((double)stored - (double)ref) / (fabs((double)ref) + 1e-30);
            if (rel > worst) worst = rel;
            out_bad += ((const uint16_t*)f24.seg.out)[i] != to_bf16(stored);
        }
        CHECK(worst < ldexp(1.0, -14), "fp24 sr=%d: master differs from FP32 by %.3g relative", sr, worst);
        CHECK(out_bad == 0, "fp24 sr=%d: %d accelerator copies are not the stored value rounded", sr, out_bad);
        free_segment(&f32);
        free_segment(&f24);
    }

    for (int li = 1; li < nlev; ++li) {
        tseg_t a, b;
        make_segment_pt(&a, n, DT_BF16, DT_FP8, DT_BF16, 13, DT_F24);
        make_segment_pt(&b, n, DT_BF16, DT_FP8, DT_BF16, 13, DT_F24);
        adambridge_segment* pa = &a.seg;
        adambridge_segment* pb = &b.seg;
        adambridge_hparams h = test_hparams(0);
        adambridge_set_simd_level(0);
        for (int step = 0; step < 3; ++step) adambridge_multi_apply(NULL, &pa, &h, 1, 1);
        const int actual = adambridge_set_simd_level(levels[li]);
        for (int step = 0; step < 3; ++step) adambridge_multi_apply(NULL, &pb, &h, 1, 1);
        adambridge_set_simd_level(-1);
        int64_t same = 0;
        double worst = 0;
        for (int64_t i = 0; i < n; ++i) {
            const float x = load_master(a.seg.p, a.seg.p_lo, DT_F24, i), y = load_master(b.seg.p, b.seg.p_lo, DT_F24, i);
            same += f32_bits(x) == f32_bits(y);
            const double rel = fabs((double)x - (double)y) / (fabs((double)x) + 1e-30);
            if (rel > worst) worst = rel;
        }
        /* FMA can move a value across a 24-bit rounding boundary: a one-ulp (2^-15) difference, rarely */
        CHECK(same > n * 0.99 && worst < ldexp(1.0, -13), "fp24 level=%d: %.4f of masters identical, worst %.3g",
              actual, (double)same / n, worst);
        free_segment(&a);
        free_segment(&b);
    }

    tseg_t bad_seg;
    make_segment_pt(&bad_seg, 10, DT_F32, DT_F32, DT_NONE, 1, DT_F24);
    adambridge_segment* pbad = &bad_seg.seg;
    adambridge_hparams h = test_hparams(0);
    uint8_t* lo = bad_seg.seg.p_lo;
    bad_seg.seg.p_lo = NULL;
    CHECK(adambridge_multi_apply(NULL, &pbad, &h, 1, 1) == -2, "fp24 segment without a lo plane accepted");
    bad_seg.seg.p_lo = lo;
    bad_seg.seg.p_dtype = 7;
    CHECK(adambridge_multi_apply(NULL, &pbad, &h, 1, 1) == -13, "unknown master dtype accepted");
    free_segment(&bad_seg);
}

/* The AVX2 FP8 block steps (decode, sqrt + maxima, encode) against the generic
 * loops, bit for bit: every code, specials (NaN, inf, -0, subnormals, values
 * past the format's max), both rounding modes and partial blocks. */
#if defined(ADAMBRIDGE_X86)
static int same_bits(float a, float b) { return f32_bits(a) == f32_bits(b) || (isnan(a) && isnan(b)); }

static int fp8_block_avx2_checks(void)
{
    static const float special[] = { 0.0f, -0.0f, 1.0f, -1.0f, 448.0f, 449.0f, 57344.0f, 1e30f, -1e30f,
                                     1e-45f, -1e-45f, 1e-40f, 3e-38f, 1.5e-3f };
    const int sizes[] = { FP8_BLOCK, FP8_BLOCK - 1, 37, 8, 1 };
    int bad = 0, blocks = 0;
    rs = 4242;
    for (int round = 0; round < 400; ++round) {
        const int nb = sizes[round % 5];
        AX_ALIGN64 float mt[FP8_BLOCK], vt[FP8_BLOCK], mt2[FP8_BLOCK], vt2[FP8_BLOCK];
        uint8_t mq[FP8_BLOCK], vq[FP8_BLOCK], mq2[FP8_BLOCK], vq2[FP8_BLOCK];
        for (int k = 0; k < FP8_BLOCK; ++k) {
            mq[k] = (uint8_t)(round * 131 + k * 7);   /* every code, NaN codes included */
            vq[k] = (uint8_t)(round * 17 + k * 29);
        }
        const float sm = ldexpf(1.0f + 0.5f * frand(), round % 30 - 20), sv = ldexpf(1.0f + 0.5f * frand(), round % 20 - 12);
        fp8_decode_block(mq, vq, sm, sv, mt, vt, nb);
        fp8_decode_block_avx2(mq, vq, sm, sv, mt2, vt2, nb);
        for (int k = 0; k < nb; ++k) bad += !same_bits(mt[k], mt2[k]) || !same_bits(vt[k], vt2[k]);

        for (int k = 0; k < nb; ++k) {                /* fresh moments, as the update leaves them */
            const int e = (int)(frand() * 40.0f);
            mt[k] = (k % 11 == 3) ? special[(round + k) % 14] : frand() * ldexpf(1.0f, e - 20);
            vt[k] = (k % 13 == 5) ? (k % 2 ? NAN : -1.0f) : fabsf(frand()) * ldexpf(1.0f, e - 30);
            if (round % 7 == 0 && k == 1) mt[k] = NAN;
            mt2[k] = mt[k];
            vt2[k] = vt[k];
        }
        float am, au, am2, au2;
        fp8_absmax_block(mt, vt, nb, &am, &au);
        fp8_absmax_block_avx2(mt2, vt2, nb, &am2, &au2);
        bad += !same_bits(am, am2) || !same_bits(au, au2);
        for (int k = 0; k < nb; ++k) bad += !same_bits(vt[k], vt2[k]);

        const float scm = am > 0.0f ? am / 448.0f : 1.0f, scu = au > 0.0f ? au / 57344.0f : 1.0f;
        for (int sr = 0; sr <= 1; ++sr) {
            const uint32_t idx0 = (uint32_t)round * 1000003u, s_m = 0x1234567u + (uint32_t)round, s_v = seed_v_of(s_m);
            fp8_encode_block(mt, vt, scm, scu, sr, idx0, s_m, s_v, mq, vq, nb);
            fp8_encode_block_avx2(mt, vt, scm, scu, sr, idx0, s_m, s_v, mq2, vq2, nb);
            for (int k = 0; k < nb; ++k) bad += mq[k] != mq2[k] || vq[k] != vq2[k];
            /* odd scales too: values far past the format's max and into its subnormals */
            fp8_encode_block(mt, vt, scm * 1e-3f, scu * 1e3f, sr, idx0, s_m, s_v, mq, vq, nb);
            fp8_encode_block_avx2(mt, vt, scm * 1e-3f, scu * 1e3f, sr, idx0, s_m, s_v, mq2, vq2, nb);
            for (int k = 0; k < nb; ++k) bad += mq[k] != mq2[k] || vq[k] != vq2[k];
        }
        ++blocks;
    }
    CHECK(bad == 0, "fp8 AVX2 block steps: %d mismatches vs the generic loops", bad);
    return blocks;
}
#endif

/* FP8 through each SIMD level vs scalar: same codes for the overwhelming
 * majority (FMA can move a pre-rounding value across a rounding boundary). */
static void fp8_simd_case(int level)
{
    tseg_t a, b;
    const int64_t n = 50000;
    make_segment(&a, n, DT_BF16, DT_FP8, DT_BF16, 5);
    make_segment(&b, n, DT_BF16, DT_FP8, DT_BF16, 5);
    adambridge_segment* pa = &a.seg;
    adambridge_segment* pb = &b.seg;
    adambridge_hparams h = test_hparams(1);
    adambridge_set_simd_level(0);
    for (int step = 0; step < 3; ++step) adambridge_multi_apply(NULL, &pa, &h, 1, 1);
    const int actual = adambridge_set_simd_level(level);
    for (int step = 0; step < 3; ++step) adambridge_multi_apply(NULL, &pb, &h, 1, 1);
    adambridge_set_simd_level(-1);
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

/* ------------------------------------------------------------------------- */
/* 4. Lion                                                                   */
/* ------------------------------------------------------------------------- */

static void lion_segment(tseg_t* t, int64_t n, int gt, int mvt, int ot, uint32_t seed)
{
    make_segment(t, n, gt, mvt, ot, seed);
    free(t->seg.v); free(t->seg.v_scale);
    t->seg.v = NULL;
    t->seg.v_scale = NULL;
    t->seg.algorithm = ALGO_LION;
}

static int lion_same(const tseg_t* a, const tseg_t* b)
{
    const adambridge_segment* x = &a->seg;
    const adambridge_segment* y = &b->seg;
    const size_t n = (size_t)x->n;
    const size_t mb = x->mv_dtype == DT_F32 ? 4 : (x->mv_dtype == DT_BF16 ? 2 : 1);
    const size_t mlen = x->mv_dtype == DT_FP8 ? (size_t)a->nblocks * FP8_BLOCK : n;
    int ok = master_same(x, y) && same_bytes(x->m, y->m, mlen * mb);
    if (x->mv_dtype == DT_FP8) ok = ok && same_bytes(x->m_scale, y->m_scale, (size_t)a->nblocks * 4);
    if (x->out_dtype) ok = ok && same_bytes(x->out, y->out, n * (x->out_dtype == DT_F32 ? 4 : 2));
    return ok;
}

/* The update itself, checked against the formula on the first step (m = 0,
 * so c = (1 - b1) g and the step is exactly -lr * sign(g) after decay). */
static void lion_formula_case(void)
{
    tseg_t t;
    const int64_t n = 1001;
    lion_segment(&t, n, DT_F32, DT_F32, DT_F32, 21);
    float* p0 = (float*)malloc((size_t)n * 4);
    memcpy(p0, t.seg.p, (size_t)n * 4);
    adambridge_segment* ps = &t.seg;
    adambridge_hparams h = test_hparams(0);
    h.step_size = 1e-3f;
    CHECK(adambridge_multi_apply(NULL, &ps, &h, 1, 1) == 0, "lion formula: rejected");
    int bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float g = ((const float*)t.seg.g)[i];
        const double want = (double)p0[i] * h.decay - (g > 0 ? 1e-3 : (g < 0 ? -1e-3 : 0.0));
        const double m_want = (1.0 - 0.999) * g;
        const float pi = ((const float*)t.seg.p)[i];
        if (fabs(pi - want) > 1e-6 || fabs(((float*)t.seg.m)[i] - m_want) > 1e-9 || ((float*)t.seg.out)[i] != pi) ++bad;
    }
    CHECK(bad == 0, "lion formula: %d of %lld elements wrong", bad, (long long)n);
    free(p0);
    free_segment(&t);
}

/* Every SIMD level is bit-identical to scalar (all multiply-adds are fused
 * everywhere), for every dtype combination, several steps, odd lengths. */
static int lion_simd_case(int level)
{
    const int64_t sizes[] = { 1, 37, 100003 };
    const int pts[] = { DT_F32, DT_F24 };
    int cases = 0;
    for (int gt = DT_F32; gt <= DT_BF16; ++gt)
        for (int mvt = DT_F32; mvt <= DT_FP8; ++mvt)
            for (int ot = DT_NONE; ot <= DT_BF16; ++ot)
                for (int pi = 0; pi < 2; ++pi)
                    for (int sr = 0; sr <= (mvt != DT_F32 || pts[pi] == DT_F24); ++sr)
                        for (int si = 0; si < 3; ++si, ++cases) {
                            tseg_t a, b;
                            lion_segment(&a, sizes[si], gt, mvt, ot, 31);
                            lion_segment(&b, sizes[si], gt, mvt, ot, 31);
                            if (pts[pi] == DT_F24) {   /* the same values as an FP24 master */
                                free_segment(&a); free_segment(&b);
                                make_segment_pt(&a, sizes[si], gt, mvt, ot, 31, DT_F24);
                                make_segment_pt(&b, sizes[si], gt, mvt, ot, 31, DT_F24);
                                free(a.seg.v); free(a.seg.v_scale); a.seg.v = NULL; a.seg.v_scale = NULL; a.seg.algorithm = ALGO_LION;
                                free(b.seg.v); free(b.seg.v_scale); b.seg.v = NULL; b.seg.v_scale = NULL; b.seg.algorithm = ALGO_LION;
                            }
                            adambridge_segment* pa = &a.seg;
                            adambridge_segment* pb = &b.seg;
                            adambridge_hparams h = test_hparams(sr);
                            adambridge_set_simd_level(0);
                            for (int step = 0; step < 3; ++step) adambridge_multi_apply(NULL, &pa, &h, 1, 1);
                            const int actual = adambridge_set_simd_level(level);
                            for (int step = 0; step < 3; ++step) adambridge_multi_apply(NULL, &pb, &h, 1, 1);
                            adambridge_set_simd_level(-1);
                            CHECK(lion_same(&a, &b), "lion level=%d g=%d m=%d out=%d master=%d sr=%d n=%lld differs from scalar",
                                  actual, gt, mvt, ot, pts[pi], sr, (long long)sizes[si]);
                            free_segment(&a);
                            free_segment(&b);
                        }
    return cases;
}

/* Thread-count invariance and whole-vs-chunked updates, mixed with AdamW
 * segments in the same call. */
static void lion_multi_case(int threads)
{
    enum { NS = 6 };
    const int64_t sizes[NS] = { 1, 129, 70001, 300000, 1000, 300001 };
    tseg_t a[NS], b[NS];
    adambridge_segment* pa[NS];
    adambridge_segment* pb[NS];
    adambridge_hparams hps[NS];
    for (int k = 0; k < NS; ++k) {
        const int mvt = DT_F32 + k % 3;
        if (k == 4) {   /* an AdamW segment in the same batch */
            make_segment(&a[k], sizes[k], DT_BF16, mvt, DT_BF16, 50u + (uint32_t)k);
            make_segment(&b[k], sizes[k], DT_BF16, mvt, DT_BF16, 50u + (uint32_t)k);
        } else {
            lion_segment(&a[k], sizes[k], DT_BF16, mvt, DT_BF16, 50u + (uint32_t)k);
            lion_segment(&b[k], sizes[k], DT_BF16, mvt, DT_BF16, 50u + (uint32_t)k);
        }
        pa[k] = &a[k].seg;
        pb[k] = &b[k].seg;
        hps[k] = test_hparams(1);
        hps[k].seed = 91u + (uint32_t)k;
    }
    void* pool = adambridge_pool_create(threads, NULL, 0);
    for (int step = 0; step < 3; ++step) {
        CHECK(adambridge_multi_apply(NULL, pa, hps, NS, 1) == 0, "lion multi single");
        CHECK(adambridge_multi_apply(pool, pb, hps, NS, threads) == 0, "lion multi pool");
    }
    adambridge_pool_destroy(pool);
    int diff = 0;
    for (int k = 0; k < NS; ++k) {
        const size_t vbytes = (size_t)(a[k].seg.mv_dtype == DT_FP8 ? a[k].nblocks * FP8_BLOCK : a[k].seg.n)
                              * (a[k].seg.mv_dtype == DT_F32 ? 4 : (a[k].seg.mv_dtype == DT_BF16 ? 2 : 1));
        diff += !(lion_same(&a[k], &b[k]) && (a[k].seg.v == NULL || same_bytes(a[k].seg.v, b[k].seg.v, vbytes)));
    }
    CHECK(diff == 0, "lion threads=%d: %d segments differ from the single-threaded result", threads, diff);
    for (int k = 0; k < NS; ++k) { free_segment(&a[k]); free_segment(&b[k]); }

    /* chunks of one tensor at 128-aligned offsets == the whole tensor */
    for (int mvt = DT_F32; mvt <= DT_FP8; ++mvt) {
        tseg_t w, c;
        const int64_t n = 300001, cuts[] = { 0, 65536, 131072 + 4096, n };
        lion_segment(&w, n, DT_BF16, mvt, DT_BF16, 7);
        lion_segment(&c, n, DT_BF16, mvt, DT_BF16, 7);
        adambridge_segment chunks[3];
        adambridge_segment* pc[3];
        for (int i = 0; i < 3; ++i) {
            chunks[i] = chunk_of(&c.seg, cuts[i], cuts[i + 1]);
            pc[i] = &chunks[i];
        }
        adambridge_segment* pw = &w.seg;
        adambridge_hparams h = test_hparams(1), hs[3] = { h, h, h };
        for (int step = 0; step < 3; ++step) {
            adambridge_multi_apply(NULL, &pw, &h, 1, 1);
            adambridge_multi_apply(NULL, pc, hs, 3, 1);
        }
        CHECK(lion_same(&w, &c), "lion moments %d: chunked update differs from the whole-tensor update", mvt);
        free_segment(&w);
        free_segment(&c);
    }

    tseg_t bad;
    lion_segment(&bad, 10, DT_F32, DT_F32, DT_NONE, 1);
    adambridge_segment* pbad = &bad.seg;
    adambridge_hparams h = test_hparams(0);
    bad.seg.algorithm = 7;
    CHECK(adambridge_multi_apply(NULL, &pbad, &h, 1, 1) == -12, "unknown algorithm accepted");
    bad.seg.algorithm = ALGO_ADAMW;   /* AdamW needs v */
    CHECK(adambridge_multi_apply(NULL, &pbad, &h, 1, 1) == -2, "AdamW segment without v accepted");
    free_segment(&bad);
}

int main(void)
{
    const int det = adambridge_detected_simd_level();
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
    adambridge_set_simd_level(-1);
    printf("simd vs scalar: %d cases\n", cases);

    codec_checks();
    printf("fp8 codec: round-trip, nearest-even vs brute force, stochastic rounding\n");

#if defined(ADAMBRIDGE_X86)
    if (det == SIMD_AVX2 || det == SIMD_AVX512)
        printf("fp8 AVX2 block steps == generic loops, bitwise: %d blocks\n", fp8_block_avx2_checks());
#endif
    for (int li = 1; li < nlev; ++li) fp8_simd_case(levels[li]);
    printf("fp8 simd vs scalar: %d SIMD level%s\n", nlev - 1, nlev - 1 == 1 ? "" : "s");

    const int thread_counts[] = { 2, 3, 8 };
    for (int i = 0; i < 3; ++i) multi_case(thread_counts[i]);
    printf("multi-tensor thread invariance: 2, 3, 8 threads\n");

    fresh_pool_case();
    printf("pools used right after creation: 300\n");

    for (int mvt = DT_F32; mvt <= DT_FP8; ++mvt) { chunk_case(mvt, DT_F32); chunk_case(mvt, DT_F24); }
    printf("chunked updates (segment offsets): fp32, bf16, fp8 moments, with fp32 and fp24 masters\n");

    f24_case(nlev, levels);
    printf("fp24 master: nearest-even vs brute force, SR unbiased, step vs fp32 master, simd vs scalar, rejects\n");

    lion_formula_case();
    int lion_cases = 0;
    for (int li = 1; li < nlev; ++li) lion_cases += lion_simd_case(levels[li]);
    for (int i = 0; i < 3; ++i) lion_multi_case(thread_counts[i]);
    printf("lion: formula, simd == scalar bitwise (%d cases), threads 2/3/8, chunks, fp32/bf16/fp8 m\n", lion_cases);

    printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures != 0;
}
