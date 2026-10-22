/* Standalone check of every available SIMD path against the scalar path.
 * Build: cc -O2 -o selftest tests/native/selftest.c adamx/csrc/adamx_kernels.c -lm */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    float beta1, beta2, one_minus_beta1, one_minus_beta2, decay, step_size, inv_sqrt_bc2, eps;
    uint32_t seed;
    int32_t stochastic;
} hp_t;

int adamx_fused_adamw(float*, const void*, int, void*, void*, int, void*, int, int64_t, uint64_t, const hp_t*);
int adamx_set_simd_level(int);
int adamx_detected_simd_level(void);

static uint32_t rs = 12345;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) / 16777216.0f) * 2.0f - 1.0f; }
static uint16_t to_bf16(float f) { uint32_t b; memcpy(&b, &f, 4); return (uint16_t)((b + 0x7fff + ((b >> 16) & 1)) >> 16); }
static float from_bf16(uint16_t h) { uint32_t b = (uint32_t)h << 16; float f; memcpy(&f, &b, 4); return f; }

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

static double max_rel(const void* a, const void* b, int dt, int64_t n)
{
    /* Mixed tolerance: values that cancel to ~0 are compared against the
     * array's magnitude rather than their own. */
    double worst = 0, scale = 0;
    for (int64_t i = 0; i < n; ++i) scale = fmax(scale, fabs(get(b, dt, i)));
    for (int64_t i = 0; i < n; ++i) {
        double x = get(a, dt, i), y = get(b, dt, i);
        double d = fabs(x - y) / (fabs(y) + 1e-2 * scale + 1e-30);
        if (d > worst) worst = d;
    }
    return worst;
}

static int run_case(int level, int gt, int mvt, int ot, int sr, int64_t n, int split)
{
    bufs_t ref = { malloc(n * 4), alloc_dt(gt, n), alloc_dt(mvt, n), alloc_dt(mvt, n), alloc_dt(ot, n) };
    bufs_t tst = { malloc(n * 4), alloc_dt(gt, n), alloc_dt(mvt, n), alloc_dt(mvt, n), alloc_dt(ot, n) };
    fill(&ref, gt, mvt, n);
    fill(&tst, gt, mvt, n);
    hp_t h = { 0.9f, 0.999f, 0.1f, 0.001f, 1.0f - 1e-3f * 0.01f, 1e-3f / 0.1f, 1.0f / sqrtf(0.001f), 1e-8f, 0xC0FFEEu, sr };

    adamx_set_simd_level(0);
    adamx_fused_adamw(ref.p, ref.g, gt, ref.m, ref.v, mvt, ref.out, ot, n, 0, &h);
    int actual = adamx_set_simd_level(level);
    /* Process in two pieces to prove results do not depend on the split. */
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
    int ok = ep < tol32 && em < (mvt == 1 ? tol32 : tol16) && ev < (mvt == 1 ? tol32 : tol16) && eo < (ot == 1 ? tol32 : tol16);
    printf("%s level=%d g=%d mv=%d out=%d sr=%d n=%lld split=%d  p=%.2e m=%.2e v=%.2e out=%.2e\n",
           ok ? "ok  " : "FAIL", actual, gt, mvt, ot, sr, (long long)n, split, ep, em, ev, eo);
    free(ref.p); free(ref.g); free(ref.m); free(ref.v); free(ref.out);
    free(tst.p); free(tst.g); free(tst.m); free(tst.v); free(tst.out);
    return ok;
}

int main(void)
{
    int det = adamx_detected_simd_level();
    printf("detected SIMD level: %d\n", det);
    int levels[3] = { 0, 1, 2 };
    int nlev = det == 2 ? 3 : (det == 1 ? 2 : 1);
    if (det == 3) { levels[1] = 3; nlev = 2; }
    int failures = 0;
    int64_t sizes[3] = { 1, 37, 100003 };
    for (int li = 0; li < nlev; ++li)
        for (int gt = 1; gt <= 2; ++gt)
            for (int mvt = 1; mvt <= 2; ++mvt)
                for (int ot = 0; ot <= 2; ++ot)
                    for (int sr = 0; sr <= (mvt == 2); ++sr)
                        for (int si = 0; si < 3; ++si)
                            for (int split = 0; split <= 1; ++split)
                                failures += !run_case(levels[li], gt, mvt, ot, sr, sizes[si], split);
    printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures != 0;
}
