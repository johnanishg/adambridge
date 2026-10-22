/*
 * adamx native kernels: single-pass fused AdamW update for host-offloaded
 * optimizer state.
 *
 * Plain C ABI so the library can be built by any C compiler (gcc, clang,
 * zig cc, MSVC) and loaded through ctypes without linking against libtorch.
 * ISA selection happens at runtime: AVX-512F, AVX2+FMA, NEON or scalar.
 *
 * One call reads p, g, m, v exactly once and writes p, m, v and (optionally)
 * a low-precision copy of p for the accelerator, all from registers.
 *
 *   m   = b1 * m + (1 - b1) * g
 *   v   = b2 * v + (1 - b2) * g^2
 *   p   = p * (1 - lr * wd) - (lr / bc1) * m / (sqrt(v) / sqrt(bc2) + eps)
 *
 * This is the exact operation order of torch.optim.AdamW (decoupled weight
 * decay applied before the moment step, eps added after bias correction).
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#if defined(_WIN32)
#  define ADAMX_API __declspec(dllexport)
#else
#  define ADAMX_API __attribute__((visibility("default")))
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define ADAMX_X86 1
#  include <immintrin.h>
#  if defined(_MSC_VER) && !defined(__clang__)
#    include <intrin.h>
#    define ADAMX_MSVC 1
#    define ADAMX_TARGET(isa)
#  else
#    include <cpuid.h>
#    define ADAMX_TARGET(isa) __attribute__((target(isa)))
#  endif
/* clang >= 18 gates 512-bit registers behind a separate "evex512" feature. */
#  if defined(__clang__) && __clang_major__ >= 18
#    define ADAMX_AVX512 "avx512f,evex512"
#  else
#    define ADAMX_AVX512 "avx512f"
#  endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#  define ADAMX_NEON 1
#  include <arm_neon.h>
#endif

#define ADAMX_ABI_VERSION 1

enum { DT_NONE = 0, DT_F32 = 1, DT_BF16 = 2 };
enum { SIMD_SCALAR = 0, SIMD_AVX2 = 1, SIMD_AVX512 = 2, SIMD_NEON = 3 };

typedef struct adamx_hparams {
    float beta1;
    float beta2;
    float one_minus_beta1;
    float one_minus_beta2;
    float decay;         /* 1 - lr * weight_decay */
    float step_size;     /* lr / (1 - beta1^t) */
    float inv_sqrt_bc2;  /* 1 / sqrt(1 - beta2^t) */
    float eps;
    uint32_t seed;       /* stochastic rounding stream for this (param, step) */
    int32_t stochastic;  /* nonzero: stochastic rounding when storing BF16 moments */
} adamx_hparams;

/* ------------------------------------------------------------------------- */
/* Scalar helpers (also used for vector tails, so results do not depend on   */
/* where a buffer is split between threads).                                 */
/* ------------------------------------------------------------------------- */

static inline uint32_t f32_bits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
static inline float bits_f32(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }

/* Counter-based hash (lowbias32): a stateless RNG, so any element can be
 * rounded independently by any thread and results are reproducible. */
static inline uint32_t hash32(uint32_t x)
{
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static inline uint32_t bf16_round_bits(uint32_t b)
{
    if ((b & 0x7fffffffU) > 0x7f800000U) return b | 0x00400000U; /* quiet NaN */
    return b + 0x7fffU + ((b >> 16) & 1U);                         /* nearest-even */
}

static inline float load_scalar(const void* base, int dt, int64_t i)
{
    if (dt == DT_BF16) return bits_f32((uint32_t)((const uint16_t*)base)[i] << 16);
    return ((const float*)base)[i];
}

static inline void store_moment_scalar(void* base, int dt, int64_t i, float x, int sr, uint32_t r)
{
    if (dt == DT_F32) { ((float*)base)[i] = x; return; }
    uint32_t b = f32_bits(x);
    b = sr ? b + (r & 0xffffU) : bf16_round_bits(b);
    ((uint16_t*)base)[i] = (uint16_t)(b >> 16);
}

static inline size_t dt_size(int dt) { return dt == DT_F32 ? 4 : (dt == DT_BF16 ? 2 : 0); }
static inline void* adv(void* p, int dt, int64_t i) { return p ? (char*)p + (size_t)i * dt_size(dt) : p; }
static inline const void* cadv(const void* p, int dt, int64_t i) { return (const char*)p + (size_t)i * dt_size(dt); }

static uint32_t seed_v_of(uint32_t seed) { return hash32(seed ^ 0xA511E9B3U); }

static void fused_scalar(float* p, const void* g, int gt, void* m, void* v, int mvt,
                         void* out, int ot, int64_t n, uint64_t off, const adamx_hparams* h)
{
    const uint32_t s_m = h->seed, s_v = seed_v_of(h->seed);
    for (int64_t i = 0; i < n; ++i) {
        const float gi = load_scalar(g, gt, i);
        float mi = load_scalar(m, mvt, i);
        float vi = load_scalar(v, mvt, i);
        float pi = p[i] * h->decay;
        mi = h->beta1 * mi + h->one_minus_beta1 * gi;
        vi = h->beta2 * vi + h->one_minus_beta2 * (gi * gi);
        pi -= h->step_size * (mi / (sqrtf(vi) * h->inv_sqrt_bc2 + h->eps));
        p[i] = pi;

        const uint32_t idx = (uint32_t)(off + (uint64_t)i);
        store_moment_scalar(m, mvt, i, mi, h->stochastic, hash32(idx ^ s_m));
        store_moment_scalar(v, mvt, i, vi, h->stochastic, hash32(idx ^ s_v));
        if (ot == DT_F32) ((float*)out)[i] = pi;
        else if (ot == DT_BF16) ((uint16_t*)out)[i] = (uint16_t)(bf16_round_bits(f32_bits(pi)) >> 16);
    }
}

/* ------------------------------------------------------------------------- */
/* AVX2 + FMA: 8 lanes                                                       */
/* ------------------------------------------------------------------------- */
#if defined(ADAMX_X86)

ADAMX_TARGET("avx2,fma")
static inline __m256 avx2_load(const void* base, int dt, int64_t i)
{
    if (dt == DT_BF16) {
        __m128i h = _mm_loadu_si128((const __m128i*)((const uint16_t*)base + i));
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
    }
    return _mm256_loadu_ps((const float*)base + i);
}

ADAMX_TARGET("avx2,fma")
static inline __m256i avx2_hash(__m256i x)
{
    x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 16));
    x = _mm256_mullo_epi32(x, _mm256_set1_epi32(0x7feb352d));
    x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 15));
    x = _mm256_mullo_epi32(x, _mm256_set1_epi32((int)0x846ca68bU));
    return _mm256_xor_si256(x, _mm256_srli_epi32(x, 16));
}

ADAMX_TARGET("avx2,fma")
static inline __m256i avx2_round_rne(__m256 x)
{
    const __m256i bits = _mm256_castps_si256(x);
    const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
    const __m256i rounded = _mm256_add_epi32(bits, _mm256_add_epi32(lsb, _mm256_set1_epi32(0x7fff)));
    const __m256i qnan = _mm256_or_si256(bits, _mm256_set1_epi32(0x00400000));
    const __m256 is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(rounded),
                                                _mm256_castsi256_ps(qnan), is_nan));
}

/* Take the high 16 bits of 8 x u32 into 8 x u16. */
ADAMX_TARGET("avx2,fma")
static inline __m128i avx2_pack_hi16(__m256i bits)
{
    __m256i hi = _mm256_srli_epi32(bits, 16);
    __m256i packed = _mm256_packus_epi32(hi, hi);     /* [a0-3 a0-3 | a4-7 a4-7] */
    packed = _mm256_permute4x64_epi64(packed, 0x08);  /* qwords 0, 2 -> low half */
    return _mm256_castsi256_si128(packed);
}

ADAMX_TARGET("avx2,fma")
static inline void avx2_store_moment(void* base, int dt, int64_t i, __m256 x, int sr, __m256i idx, __m256i seed)
{
    if (dt == DT_F32) { _mm256_storeu_ps((float*)base + i, x); return; }
    __m256i bits;
    if (sr) {
        __m256i r = _mm256_and_si256(avx2_hash(_mm256_xor_si256(idx, seed)), _mm256_set1_epi32(0xffff));
        bits = _mm256_add_epi32(_mm256_castps_si256(x), r);
    } else {
        bits = avx2_round_rne(x);
    }
    _mm_storeu_si128((__m128i*)((uint16_t*)base + i), avx2_pack_hi16(bits));
}

ADAMX_TARGET("avx2,fma")
static void fused_avx2(float* p, const void* g, int gt, void* m, void* v, int mvt,
                       void* out, int ot, int64_t n, uint64_t off, const adamx_hparams* h)
{
    const __m256 b1 = _mm256_set1_ps(h->beta1), b2 = _mm256_set1_ps(h->beta2);
    const __m256 omb1 = _mm256_set1_ps(h->one_minus_beta1), omb2 = _mm256_set1_ps(h->one_minus_beta2);
    const __m256 decay = _mm256_set1_ps(h->decay), step = _mm256_set1_ps(h->step_size);
    const __m256 isbc2 = _mm256_set1_ps(h->inv_sqrt_bc2), eps = _mm256_set1_ps(h->eps);
    const __m256i s_m = _mm256_set1_epi32((int)h->seed), s_v = _mm256_set1_epi32((int)seed_v_of(h->seed));
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const int sr = h->stochastic;

    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 vg = avx2_load(g, gt, i);
        __m256 vm = avx2_load(m, mvt, i);
        __m256 vv = avx2_load(v, mvt, i);
        __m256 vp = _mm256_mul_ps(_mm256_loadu_ps(p + i), decay);

        vm = _mm256_fmadd_ps(b1, vm, _mm256_mul_ps(omb1, vg));
        vv = _mm256_fmadd_ps(b2, vv, _mm256_mul_ps(omb2, _mm256_mul_ps(vg, vg)));
        const __m256 denom = _mm256_fmadd_ps(_mm256_sqrt_ps(vv), isbc2, eps);
        vp = _mm256_fnmadd_ps(step, _mm256_div_ps(vm, denom), vp);
        _mm256_storeu_ps(p + i, vp);

        const __m256i idx = _mm256_add_epi32(_mm256_set1_epi32((int)(uint32_t)(off + (uint64_t)i)), lane);
        avx2_store_moment(m, mvt, i, vm, sr, idx, s_m);
        avx2_store_moment(v, mvt, i, vv, sr, idx, s_v);
        if (ot == DT_F32) _mm256_storeu_ps((float*)out + i, vp);
        else if (ot == DT_BF16) _mm_storeu_si128((__m128i*)((uint16_t*)out + i), avx2_pack_hi16(avx2_round_rne(vp)));
    }
    if (i < n)
        fused_scalar(p + i, cadv(g, gt, i), gt, adv(m, mvt, i), adv(v, mvt, i), mvt,
                     adv(out, ot, i), ot, n - i, off + (uint64_t)i, h);
}

/* ------------------------------------------------------------------------- */
/* AVX-512F: 16 lanes                                                        */
/* ------------------------------------------------------------------------- */

ADAMX_TARGET(ADAMX_AVX512)
static inline __m512 avx512_load(const void* base, int dt, int64_t i)
{
    if (dt == DT_BF16) {
        __m256i h = _mm256_loadu_si256((const __m256i*)((const uint16_t*)base + i));
        return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
    }
    return _mm512_loadu_ps((const float*)base + i);
}

ADAMX_TARGET(ADAMX_AVX512)
static inline __m512i avx512_hash(__m512i x)
{
    x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 16));
    x = _mm512_mullo_epi32(x, _mm512_set1_epi32(0x7feb352d));
    x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 15));
    x = _mm512_mullo_epi32(x, _mm512_set1_epi32((int)0x846ca68bU));
    return _mm512_xor_si512(x, _mm512_srli_epi32(x, 16));
}

ADAMX_TARGET(ADAMX_AVX512)
static inline __m512i avx512_round_rne(__m512 x)
{
    const __m512i bits = _mm512_castps_si512(x);
    const __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
    const __m512i rounded = _mm512_add_epi32(bits, _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7fff)));
    const __mmask16 is_nan = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
    return _mm512_mask_mov_epi32(rounded, is_nan, _mm512_or_si512(bits, _mm512_set1_epi32(0x00400000)));
}

ADAMX_TARGET(ADAMX_AVX512)
static inline void avx512_store_bf16(void* base, int64_t i, __m512i bits)
{
    _mm256_storeu_si256((__m256i*)((uint16_t*)base + i), _mm512_cvtepi32_epi16(_mm512_srli_epi32(bits, 16)));
}

ADAMX_TARGET(ADAMX_AVX512)
static inline void avx512_store_moment(void* base, int dt, int64_t i, __m512 x, int sr, __m512i idx, __m512i seed)
{
    if (dt == DT_F32) { _mm512_storeu_ps((float*)base + i, x); return; }
    __m512i bits;
    if (sr) {
        __m512i r = _mm512_and_si512(avx512_hash(_mm512_xor_si512(idx, seed)), _mm512_set1_epi32(0xffff));
        bits = _mm512_add_epi32(_mm512_castps_si512(x), r);
    } else {
        bits = avx512_round_rne(x);
    }
    avx512_store_bf16(base, i, bits);
}

ADAMX_TARGET(ADAMX_AVX512)
static void fused_avx512(float* p, const void* g, int gt, void* m, void* v, int mvt,
                         void* out, int ot, int64_t n, uint64_t off, const adamx_hparams* h)
{
    const __m512 b1 = _mm512_set1_ps(h->beta1), b2 = _mm512_set1_ps(h->beta2);
    const __m512 omb1 = _mm512_set1_ps(h->one_minus_beta1), omb2 = _mm512_set1_ps(h->one_minus_beta2);
    const __m512 decay = _mm512_set1_ps(h->decay), step = _mm512_set1_ps(h->step_size);
    const __m512 isbc2 = _mm512_set1_ps(h->inv_sqrt_bc2), eps = _mm512_set1_ps(h->eps);
    const __m512i s_m = _mm512_set1_epi32((int)h->seed), s_v = _mm512_set1_epi32((int)seed_v_of(h->seed));
    const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const int sr = h->stochastic;

    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 vg = avx512_load(g, gt, i);
        __m512 vm = avx512_load(m, mvt, i);
        __m512 vv = avx512_load(v, mvt, i);
        __m512 vp = _mm512_mul_ps(_mm512_loadu_ps(p + i), decay);

        vm = _mm512_fmadd_ps(b1, vm, _mm512_mul_ps(omb1, vg));
        vv = _mm512_fmadd_ps(b2, vv, _mm512_mul_ps(omb2, _mm512_mul_ps(vg, vg)));
        const __m512 denom = _mm512_fmadd_ps(_mm512_sqrt_ps(vv), isbc2, eps);
        vp = _mm512_fnmadd_ps(step, _mm512_div_ps(vm, denom), vp);
        _mm512_storeu_ps(p + i, vp);

        const __m512i idx = _mm512_add_epi32(_mm512_set1_epi32((int)(uint32_t)(off + (uint64_t)i)), lane);
        avx512_store_moment(m, mvt, i, vm, sr, idx, s_m);
        avx512_store_moment(v, mvt, i, vv, sr, idx, s_v);
        if (ot == DT_F32) _mm512_storeu_ps((float*)out + i, vp);
        else if (ot == DT_BF16) avx512_store_bf16(out, i, avx512_round_rne(vp));
    }
    if (i < n)
        fused_scalar(p + i, cadv(g, gt, i), gt, adv(m, mvt, i), adv(v, mvt, i), mvt,
                     adv(out, ot, i), ot, n - i, off + (uint64_t)i, h);
}

static unsigned long long read_xcr0(void)
{
#  if defined(ADAMX_MSVC)
    return _xgetbv(0);
#  else
    unsigned int lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
#  endif
}

static void cpuid_count(unsigned leaf, unsigned sub, unsigned r[4])
{
#  if defined(ADAMX_MSVC)
    int x[4];
    __cpuidex(x, (int)leaf, (int)sub);
    r[0] = (unsigned)x[0]; r[1] = (unsigned)x[1]; r[2] = (unsigned)x[2]; r[3] = (unsigned)x[3];
#  else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#  endif
}

#endif /* ADAMX_X86 */

/* ------------------------------------------------------------------------- */
/* NEON (AArch64): 4 lanes                                                   */
/* ------------------------------------------------------------------------- */
#if defined(ADAMX_NEON)

static inline float32x4_t neon_load(const void* base, int dt, int64_t i)
{
    if (dt == DT_BF16) return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16((const uint16_t*)base + i), 16));
    return vld1q_f32((const float*)base + i);
}

static inline uint32x4_t neon_hash(uint32x4_t x)
{
    x = veorq_u32(x, vshrq_n_u32(x, 16));
    x = vmulq_u32(x, vdupq_n_u32(0x7feb352dU));
    x = veorq_u32(x, vshrq_n_u32(x, 15));
    x = vmulq_u32(x, vdupq_n_u32(0x846ca68bU));
    return veorq_u32(x, vshrq_n_u32(x, 16));
}

static inline uint32x4_t neon_round_rne(float32x4_t x)
{
    const uint32x4_t bits = vreinterpretq_u32_f32(x);
    const uint32x4_t lsb = vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1));
    const uint32x4_t rounded = vaddq_u32(bits, vaddq_u32(lsb, vdupq_n_u32(0x7fff)));
    const uint32x4_t is_nan = vmvnq_u32(vceqq_f32(x, x));
    return vbslq_u32(is_nan, vorrq_u32(bits, vdupq_n_u32(0x00400000)), rounded);
}

static inline void neon_store_moment(void* base, int dt, int64_t i, float32x4_t x, int sr, uint32x4_t idx, uint32x4_t seed)
{
    if (dt == DT_F32) { vst1q_f32((float*)base + i, x); return; }
    uint32x4_t bits;
    if (sr) bits = vaddq_u32(vreinterpretq_u32_f32(x), vandq_u32(neon_hash(veorq_u32(idx, seed)), vdupq_n_u32(0xffff)));
    else bits = neon_round_rne(x);
    vst1_u16((uint16_t*)base + i, vshrn_n_u32(bits, 16));
}

static void fused_neon(float* p, const void* g, int gt, void* m, void* v, int mvt,
                       void* out, int ot, int64_t n, uint64_t off, const adamx_hparams* h)
{
    const float32x4_t b1 = vdupq_n_f32(h->beta1), b2 = vdupq_n_f32(h->beta2);
    const float32x4_t omb1 = vdupq_n_f32(h->one_minus_beta1), omb2 = vdupq_n_f32(h->one_minus_beta2);
    const float32x4_t decay = vdupq_n_f32(h->decay), step = vdupq_n_f32(h->step_size);
    const float32x4_t isbc2 = vdupq_n_f32(h->inv_sqrt_bc2), eps = vdupq_n_f32(h->eps);
    const uint32x4_t s_m = vdupq_n_u32(h->seed), s_v = vdupq_n_u32(seed_v_of(h->seed));
    static const uint32_t lane_init[4] = {0, 1, 2, 3};
    const uint32x4_t lane = vld1q_u32(lane_init);
    const int sr = h->stochastic;

    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t vg = neon_load(g, gt, i);
        float32x4_t vm = neon_load(m, mvt, i);
        float32x4_t vv = neon_load(v, mvt, i);
        float32x4_t vp = vmulq_f32(vld1q_f32(p + i), decay);

        vm = vfmaq_f32(vmulq_f32(omb1, vg), b1, vm);
        vv = vfmaq_f32(vmulq_f32(omb2, vmulq_f32(vg, vg)), b2, vv);
        const float32x4_t denom = vfmaq_f32(eps, vsqrtq_f32(vv), isbc2);
        vp = vfmsq_f32(vp, step, vdivq_f32(vm, denom));
        vst1q_f32(p + i, vp);

        const uint32x4_t idx = vaddq_u32(vdupq_n_u32((uint32_t)(off + (uint64_t)i)), lane);
        neon_store_moment(m, mvt, i, vm, sr, idx, s_m);
        neon_store_moment(v, mvt, i, vv, sr, idx, s_v);
        if (ot == DT_F32) vst1q_f32((float*)out + i, vp);
        else if (ot == DT_BF16) vst1_u16((uint16_t*)out + i, vshrn_n_u32(neon_round_rne(vp), 16));
    }
    if (i < n)
        fused_scalar(p + i, cadv(g, gt, i), gt, adv(m, mvt, i), adv(v, mvt, i), mvt,
                     adv(out, ot, i), ot, n - i, off + (uint64_t)i, h);
}

#endif /* ADAMX_NEON */

/* ------------------------------------------------------------------------- */
/* Runtime dispatch                                                          */
/* ------------------------------------------------------------------------- */

static int detect_simd(void)
{
#if defined(ADAMX_X86)
    unsigned r[4];
    cpuid_count(0, 0, r);
    if (r[0] < 7) return SIMD_SCALAR;
    cpuid_count(1, 0, r);
    const unsigned ecx = r[2];
    const int osxsave = (ecx >> 27) & 1, avx = (ecx >> 28) & 1, fma = (ecx >> 12) & 1;
    if (!(osxsave && avx && fma)) return SIMD_SCALAR;
    const unsigned long long xcr0 = read_xcr0();
    if ((xcr0 & 0x6) != 0x6) return SIMD_SCALAR;          /* XMM + YMM state */
    cpuid_count(7, 0, r);
    const unsigned ebx = r[1];
    if (((ebx >> 16) & 1) && (xcr0 & 0xe6) == 0xe6) return SIMD_AVX512; /* + opmask/ZMM */
    if ((ebx >> 5) & 1) return SIMD_AVX2;
    return SIMD_SCALAR;
#elif defined(ADAMX_NEON)
    return SIMD_NEON;
#else
    return SIMD_SCALAR;
#endif
}

static int g_detected = -1;
static int g_active = -1;

static void ensure_init(void)
{
    if (g_detected < 0) {
        g_detected = detect_simd();
        g_active = g_detected;
    }
}

ADAMX_API int adamx_abi_version(void) { return ADAMX_ABI_VERSION; }

ADAMX_API int adamx_detected_simd_level(void) { ensure_init(); return g_detected; }

ADAMX_API int adamx_simd_level(void) { ensure_init(); return g_active; }

/* Restrict the active ISA (e.g. to benchmark or test the scalar path).
 * A level the CPU does not support falls back to the detected one.
 * Returns the level now active. */
ADAMX_API int adamx_set_simd_level(int level)
{
    ensure_init();
    if (level < 0) g_active = g_detected;
    else if (level == SIMD_SCALAR) g_active = SIMD_SCALAR;
    else if (g_detected == SIMD_AVX512 && (level == SIMD_AVX2 || level == SIMD_AVX512)) g_active = level;
    else g_active = g_detected;
    return g_active;
}

ADAMX_API int adamx_hparams_size(void) { return (int)sizeof(adamx_hparams); }

/* Returns 0 on success, negative on invalid arguments. */
ADAMX_API int adamx_fused_adamw(float* p, const void* g, int g_dtype,
                                void* m, void* v, int mv_dtype,
                                void* out, int out_dtype,
                                int64_t n, uint64_t offset, const adamx_hparams* hp)
{
    if (n < 0 || !hp) return -1;
    if (n == 0) return 0;
    if (!p || !g || !m || !v) return -2;
    if (g_dtype != DT_F32 && g_dtype != DT_BF16) return -3;
    if (mv_dtype != DT_F32 && mv_dtype != DT_BF16) return -4;
    if (out_dtype != DT_NONE && out_dtype != DT_F32 && out_dtype != DT_BF16) return -5;
    if (out_dtype != DT_NONE && !out) return -6;
    if (out_dtype == DT_NONE) out = NULL;

    ensure_init();
    switch (g_active) {
#if defined(ADAMX_X86)
    case SIMD_AVX512: fused_avx512(p, g, g_dtype, m, v, mv_dtype, out, out_dtype, n, offset, hp); break;
    case SIMD_AVX2:   fused_avx2(p, g, g_dtype, m, v, mv_dtype, out, out_dtype, n, offset, hp); break;
#endif
#if defined(ADAMX_NEON)
    case SIMD_NEON:   fused_neon(p, g, g_dtype, m, v, mv_dtype, out, out_dtype, n, offset, hp); break;
#endif
    default:          fused_scalar(p, g, g_dtype, m, v, mv_dtype, out, out_dtype, n, offset, hp); break;
    }
    return 0;
}
