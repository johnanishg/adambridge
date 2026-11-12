/*
 * adambridge native kernels: single-pass fused AdamW update for host-offloaded
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
 *
 * Moments are stored as FP32, BF16, or FP8 (E4M3 m and E5M2 sqrt(v) with a
 * scale per 128-element block). adambridge_multi_step updates many tensors in one
 * call on a persistent native thread pool, so per-tensor cost from Python is
 * a pointer in an array rather than a thread hand-off.
 *
 * The master weights are FP32, or "FP24": the top 24 bits of the FP32 value
 * (sign, 8 exponent bits, 15 mantissa bits) in two planes, hi (uint16: exactly
 * the BF16 bits) and lo (uint8: the next 8 bits). 3 bytes per parameter instead
 * of 4, relative precision 2^-16; stores round stochastically (nearest-even when
 * stochastic rounding is off) from their own random stream. The low-precision
 * copy for the accelerator is always taken from the value actually stored.
 *
 * A segment can instead select Lion (one moment, sign update; see the Lion
 * section below), with the same storage formats, threading and chunking.
 */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#  define _GNU_SOURCE 1  /* pthread_setaffinity_np */
#endif
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_WIN32)
#  define ADAMBRIDGE_API __declspec(dllexport)
#else
#  define ADAMBRIDGE_API __attribute__((visibility("default")))
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define ADAMBRIDGE_X86 1
#  include <immintrin.h>
#  if defined(_MSC_VER) && !defined(__clang__)
#    include <intrin.h>
#    define ADAMBRIDGE_MSVC 1
#    define ADAMBRIDGE_TARGET(isa)
#  else
#    include <cpuid.h>
#    define ADAMBRIDGE_TARGET(isa) __attribute__((target(isa)))
#  endif
/* clang >= 18 gates 512-bit registers behind a separate "evex512" feature. */
#  if defined(__clang__) && __clang_major__ >= 18
#    define ADAMBRIDGE_AVX512 "avx512f,evex512"
#  else
#    define ADAMBRIDGE_AVX512 "avx512f"
#  endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#  define ADAMBRIDGE_NEON 1
#  include <arm_neon.h>
#endif

#define ADAMBRIDGE_ABI_VERSION 5

enum { DT_NONE = 0, DT_F32 = 1, DT_BF16 = 2, DT_FP8 = 3, DT_F24 = 4 };
enum { SIMD_SCALAR = 0, SIMD_AVX2 = 1, SIMD_AVX512 = 2, SIMD_NEON = 3 };
enum { ALGO_ADAMW = 0, ALGO_LION = 1 };

typedef struct adambridge_hparams {
    float beta1;
    float beta2;
    float one_minus_beta1;
    float one_minus_beta2;
    float decay;         /* 1 - lr * weight_decay */
    float step_size;     /* AdamW: lr / (1 - beta1^t); Lion: lr */
    float inv_sqrt_bc2;  /* 1 / sqrt(1 - beta2^t) */
    float eps;
    uint32_t seed;       /* stochastic rounding stream for this (param, step) */
    int32_t stochastic;  /* nonzero: stochastic rounding when storing BF16 moments */
} adambridge_hparams;

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

/* Master weights: p points at FP32 values (pt == DT_F32) or at the FP24 hi
 * plane (uint16) with plo the lo plane (pt == DT_F24); see the header comment. */
static inline float load_master(const void* p, const uint8_t* plo, int pt, int64_t i)
{
    if (pt == DT_F24) return bits_f32(((uint32_t)((const uint16_t*)p)[i] << 16) | ((uint32_t)plo[i] << 8));
    return ((const float*)p)[i];
}

/* Round FP32 bits to the top 24: stochastically with r, else to nearest even. */
static inline uint32_t f24_round_bits(uint32_t b, int sr, uint32_t r)
{
    if ((b & 0x7fffffffU) > 0x7f800000U) return b | 0x00400000U;  /* quiet NaN */
    b = sr ? b + (r & 0xffU) : b + 0x7fU + ((b >> 8) & 1U);
    return b & ~0xffU;
}

/* Store x as element i of the master; returns the value actually stored. */
static inline float store_master(void* p, uint8_t* plo, int pt, int64_t i, float x, int sr, uint32_t r)
{
    if (pt == DT_F24) {
        const uint32_t b = f24_round_bits(f32_bits(x), sr, r);
        ((uint16_t*)p)[i] = (uint16_t)(b >> 16);
        plo[i] = (uint8_t)(b >> 8);
        return bits_f32(b);
    }
    ((float*)p)[i] = x;
    return x;
}

static inline void* padv(void* p, int pt, int64_t i) { return (char*)p + (size_t)i * (pt == DT_F24 ? 2 : 4); }
static inline uint8_t* ladv(uint8_t* plo, int64_t i) { return plo ? plo + i : plo; }

static uint32_t seed_v_of(uint32_t seed) { return hash32(seed ^ 0xA511E9B3U); }
static uint32_t seed_p_of(uint32_t seed) { return hash32(seed ^ 0x6C8E9CF5U); }

static void fused_scalar(void* p, uint8_t* plo, int pt, const void* g, int gt, void* m, void* v, int mvt,
                         void* out, int ot, int64_t n, uint64_t off, const adambridge_hparams* h)
{
    const uint32_t s_m = h->seed, s_v = seed_v_of(h->seed), s_p = seed_p_of(h->seed);
    for (int64_t i = 0; i < n; ++i) {
        const float gi = load_scalar(g, gt, i);
        float mi = load_scalar(m, mvt, i);
        float vi = load_scalar(v, mvt, i);
        float pi = load_master(p, plo, pt, i) * h->decay;
        mi = h->beta1 * mi + h->one_minus_beta1 * gi;
        vi = h->beta2 * vi + h->one_minus_beta2 * (gi * gi);
        pi -= h->step_size * (mi / (sqrtf(vi) * h->inv_sqrt_bc2 + h->eps));

        const uint32_t idx = (uint32_t)(off + (uint64_t)i);
        pi = store_master(p, plo, pt, i, pi, h->stochastic, pt == DT_F24 ? hash32(idx ^ s_p) : 0U);
        store_moment_scalar(m, mvt, i, mi, h->stochastic, hash32(idx ^ s_m));
        store_moment_scalar(v, mvt, i, vi, h->stochastic, hash32(idx ^ s_v));
        if (ot == DT_F32) ((float*)out)[i] = pi;
        else if (ot == DT_BF16) ((uint16_t*)out)[i] = (uint16_t)(bf16_round_bits(f32_bits(pi)) >> 16);
    }
}

/* ------------------------------------------------------------------------- */
/* AVX2 + FMA: 8 lanes                                                       */
/* ------------------------------------------------------------------------- */
#if defined(ADAMBRIDGE_X86)

ADAMBRIDGE_TARGET("avx2,fma")
static inline __m256 avx2_load(const void* base, int dt, int64_t i)
{
    if (dt == DT_BF16) {
        __m128i h = _mm_loadu_si128((const __m128i*)((const uint16_t*)base + i));
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
    }
    return _mm256_loadu_ps((const float*)base + i);
}

ADAMBRIDGE_TARGET("avx2,fma")
static inline __m256i avx2_hash(__m256i x)
{
    x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 16));
    x = _mm256_mullo_epi32(x, _mm256_set1_epi32(0x7feb352d));
    x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 15));
    x = _mm256_mullo_epi32(x, _mm256_set1_epi32((int)0x846ca68bU));
    return _mm256_xor_si256(x, _mm256_srli_epi32(x, 16));
}

ADAMBRIDGE_TARGET("avx2,fma")
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
ADAMBRIDGE_TARGET("avx2,fma")
static inline __m128i avx2_pack_hi16(__m256i bits)
{
    __m256i hi = _mm256_srli_epi32(bits, 16);
    __m256i packed = _mm256_packus_epi32(hi, hi);     /* [a0-3 a0-3 | a4-7 a4-7] */
    packed = _mm256_permute4x64_epi64(packed, 0x08);  /* qwords 0, 2 -> low half */
    return _mm256_castsi256_si128(packed);
}

ADAMBRIDGE_TARGET("avx2,fma")
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

ADAMBRIDGE_TARGET("avx2,fma")
static inline void avx2_store_codes(uint8_t* dst, __m256i codes)  /* 8 lanes in 0..255 -> 8 bytes */
{
    const __m128i w = _mm_packus_epi32(_mm256_castsi256_si128(codes), _mm256_extracti128_si256(codes, 1));
    _mm_storel_epi64((__m128i*)dst, _mm_packus_epi16(w, w));
}

ADAMBRIDGE_TARGET("avx2,fma")
static inline __m256 avx2_load_master(const void* p, const uint8_t* plo, int pt, int64_t i)
{
    if (pt == DT_F24) {
        const __m256i hi = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)((const uint16_t*)p + i))), 16);
        const __m256i lo = _mm256_slli_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(plo + i))), 8);
        return _mm256_castsi256_ps(_mm256_or_si256(hi, lo));
    }
    return _mm256_loadu_ps((const float*)p + i);
}

/* f24_round_bits on 8 lanes. */
ADAMBRIDGE_TARGET("avx2,fma")
static inline __m256i avx2_round_f24(__m256 x, int sr, __m256i idx, __m256i seed)
{
    const __m256i bits = _mm256_castps_si256(x);
    __m256i rounded;
    if (sr) {
        rounded = _mm256_add_epi32(bits, _mm256_and_si256(avx2_hash(_mm256_xor_si256(idx, seed)), _mm256_set1_epi32(0xff)));
    } else {
        const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 8), _mm256_set1_epi32(1));
        rounded = _mm256_add_epi32(bits, _mm256_add_epi32(lsb, _mm256_set1_epi32(0x7f)));
    }
    rounded = _mm256_andnot_si256(_mm256_set1_epi32(0xff), rounded);
    const __m256i qnan = _mm256_or_si256(bits, _mm256_set1_epi32(0x00400000));
    const __m256 is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(rounded), _mm256_castsi256_ps(qnan), is_nan));
}

/* Store 8 master lanes; returns the values stored. */
ADAMBRIDGE_TARGET("avx2,fma")
static inline __m256 avx2_store_master(void* p, uint8_t* plo, int pt, int64_t i, __m256 x, int sr, __m256i idx, __m256i seed)
{
    if (pt == DT_F24) {
        const __m256i bits = avx2_round_f24(x, sr, idx, seed);
        _mm_storeu_si128((__m128i*)((uint16_t*)p + i), avx2_pack_hi16(bits));
        avx2_store_codes(plo + i, _mm256_and_si256(_mm256_srli_epi32(bits, 8), _mm256_set1_epi32(0xff)));
        return _mm256_castsi256_ps(bits);
    }
    _mm256_storeu_ps((float*)p + i, x);
    return x;
}

ADAMBRIDGE_TARGET("avx2,fma")
static void fused_avx2(void* p, uint8_t* plo, int pt, const void* g, int gt, void* m, void* v, int mvt,
                       void* out, int ot, int64_t n, uint64_t off, const adambridge_hparams* h)
{
    const __m256 b1 = _mm256_set1_ps(h->beta1), b2 = _mm256_set1_ps(h->beta2);
    const __m256 omb1 = _mm256_set1_ps(h->one_minus_beta1), omb2 = _mm256_set1_ps(h->one_minus_beta2);
    const __m256 decay = _mm256_set1_ps(h->decay), step = _mm256_set1_ps(h->step_size);
    const __m256 isbc2 = _mm256_set1_ps(h->inv_sqrt_bc2), eps = _mm256_set1_ps(h->eps);
    const __m256i s_m = _mm256_set1_epi32((int)h->seed), s_v = _mm256_set1_epi32((int)seed_v_of(h->seed));
    const __m256i s_p = _mm256_set1_epi32((int)seed_p_of(h->seed));
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const int sr = h->stochastic;

    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 vg = avx2_load(g, gt, i);
        __m256 vm = avx2_load(m, mvt, i);
        __m256 vv = avx2_load(v, mvt, i);
        __m256 vp = _mm256_mul_ps(avx2_load_master(p, plo, pt, i), decay);

        vm = _mm256_fmadd_ps(b1, vm, _mm256_mul_ps(omb1, vg));
        vv = _mm256_fmadd_ps(b2, vv, _mm256_mul_ps(omb2, _mm256_mul_ps(vg, vg)));
        const __m256 denom = _mm256_fmadd_ps(_mm256_sqrt_ps(vv), isbc2, eps);
        vp = _mm256_fnmadd_ps(step, _mm256_div_ps(vm, denom), vp);

        const __m256i idx = _mm256_add_epi32(_mm256_set1_epi32((int)(uint32_t)(off + (uint64_t)i)), lane);
        vp = avx2_store_master(p, plo, pt, i, vp, sr, idx, s_p);
        avx2_store_moment(m, mvt, i, vm, sr, idx, s_m);
        avx2_store_moment(v, mvt, i, vv, sr, idx, s_v);
        if (ot == DT_F32) _mm256_storeu_ps((float*)out + i, vp);
        else if (ot == DT_BF16) _mm_storeu_si128((__m128i*)((uint16_t*)out + i), avx2_pack_hi16(avx2_round_rne(vp)));
    }
    if (i < n)
        fused_scalar(padv(p, pt, i), ladv(plo, i), pt, cadv(g, gt, i), gt, adv(m, mvt, i), adv(v, mvt, i), mvt,
                     adv(out, ot, i), ot, n - i, off + (uint64_t)i, h);
}

/* ------------------------------------------------------------------------- */
/* AVX-512F: 16 lanes                                                        */
/* ------------------------------------------------------------------------- */

ADAMBRIDGE_TARGET(ADAMBRIDGE_AVX512)
static inline __m512 avx512_load(const void* base, int dt, int64_t i)
{
    if (dt == DT_BF16) {
        __m256i h = _mm256_loadu_si256((const __m256i*)((const uint16_t*)base + i));
        return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
    }
    return _mm512_loadu_ps((const float*)base + i);
}

ADAMBRIDGE_TARGET(ADAMBRIDGE_AVX512)
static inline __m512i avx512_hash(__m512i x)
{
    x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 16));
    x = _mm512_mullo_epi32(x, _mm512_set1_epi32(0x7feb352d));
    x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 15));
    x = _mm512_mullo_epi32(x, _mm512_set1_epi32((int)0x846ca68bU));
    return _mm512_xor_si512(x, _mm512_srli_epi32(x, 16));
}

ADAMBRIDGE_TARGET(ADAMBRIDGE_AVX512)
static inline __m512i avx512_round_rne(__m512 x)
{
    const __m512i bits = _mm512_castps_si512(x);
    const __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
    const __m512i rounded = _mm512_add_epi32(bits, _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7fff)));
    const __mmask16 is_nan = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
    return _mm512_mask_mov_epi32(rounded, is_nan, _mm512_or_si512(bits, _mm512_set1_epi32(0x00400000)));
}

ADAMBRIDGE_TARGET(ADAMBRIDGE_AVX512)
static inline void avx512_store_bf16(void* base, int64_t i, __m512i bits)
{
    _mm256_storeu_si256((__m256i*)((uint16_t*)base + i), _mm512_cvtepi32_epi16(_mm512_srli_epi32(bits, 16)));
}

ADAMBRIDGE_TARGET(ADAMBRIDGE_AVX512)
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

ADAMBRIDGE_TARGET(ADAMBRIDGE_AVX512)
static inline __m512 avx512_load_master(const void* p, const uint8_t* plo, int pt, int64_t i)
{
    if (pt == DT_F24) {
        const __m512i hi = _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)((const uint16_t*)p + i))), 16);
        const __m512i lo = _mm512_slli_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(plo + i))), 8);
        return _mm512_castsi512_ps(_mm512_or_si512(hi, lo));
    }
    return _mm512_loadu_ps((const float*)p + i);
}

ADAMBRIDGE_TARGET(ADAMBRIDGE_AVX512)
static inline __m512i avx512_round_f24(__m512 x, int sr, __m512i idx, __m512i seed)
{
    const __m512i bits = _mm512_castps_si512(x);
    __m512i rounded;
    if (sr) {
        rounded = _mm512_add_epi32(bits, _mm512_and_si512(avx512_hash(_mm512_xor_si512(idx, seed)), _mm512_set1_epi32(0xff)));
    } else {
        const __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(bits, 8), _mm512_set1_epi32(1));
        rounded = _mm512_add_epi32(bits, _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7f)));
    }
    rounded = _mm512_andnot_si512(_mm512_set1_epi32(0xff), rounded);
    const __mmask16 is_nan = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
    return _mm512_mask_mov_epi32(rounded, is_nan, _mm512_or_si512(bits, _mm512_set1_epi32(0x00400000)));
}

ADAMBRIDGE_TARGET(ADAMBRIDGE_AVX512)
static inline __m512 avx512_store_master(void* p, uint8_t* plo, int pt, int64_t i, __m512 x, int sr, __m512i idx, __m512i seed)
{
    if (pt == DT_F24) {
        const __m512i bits = avx512_round_f24(x, sr, idx, seed);
        /* VPMOVDW / VPMOVDB truncate each lane: the high and the middle byte land as they are */
        _mm256_storeu_si256((__m256i*)((uint16_t*)p + i), _mm512_cvtepi32_epi16(_mm512_srli_epi32(bits, 16)));
        _mm_storeu_si128((__m128i*)(plo + i), _mm512_cvtepi32_epi8(_mm512_srli_epi32(bits, 8)));
        return _mm512_castsi512_ps(bits);
    }
    _mm512_storeu_ps((float*)p + i, x);
    return x;
}

ADAMBRIDGE_TARGET(ADAMBRIDGE_AVX512)
static void fused_avx512(void* p, uint8_t* plo, int pt, const void* g, int gt, void* m, void* v, int mvt,
                         void* out, int ot, int64_t n, uint64_t off, const adambridge_hparams* h)
{
    const __m512 b1 = _mm512_set1_ps(h->beta1), b2 = _mm512_set1_ps(h->beta2);
    const __m512 omb1 = _mm512_set1_ps(h->one_minus_beta1), omb2 = _mm512_set1_ps(h->one_minus_beta2);
    const __m512 decay = _mm512_set1_ps(h->decay), step = _mm512_set1_ps(h->step_size);
    const __m512 isbc2 = _mm512_set1_ps(h->inv_sqrt_bc2), eps = _mm512_set1_ps(h->eps);
    const __m512i s_m = _mm512_set1_epi32((int)h->seed), s_v = _mm512_set1_epi32((int)seed_v_of(h->seed));
    const __m512i s_p = _mm512_set1_epi32((int)seed_p_of(h->seed));
    const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const int sr = h->stochastic;

    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 vg = avx512_load(g, gt, i);
        __m512 vm = avx512_load(m, mvt, i);
        __m512 vv = avx512_load(v, mvt, i);
        __m512 vp = _mm512_mul_ps(avx512_load_master(p, plo, pt, i), decay);

        vm = _mm512_fmadd_ps(b1, vm, _mm512_mul_ps(omb1, vg));
        vv = _mm512_fmadd_ps(b2, vv, _mm512_mul_ps(omb2, _mm512_mul_ps(vg, vg)));
        const __m512 denom = _mm512_fmadd_ps(_mm512_sqrt_ps(vv), isbc2, eps);
        vp = _mm512_fnmadd_ps(step, _mm512_div_ps(vm, denom), vp);

        const __m512i idx = _mm512_add_epi32(_mm512_set1_epi32((int)(uint32_t)(off + (uint64_t)i)), lane);
        vp = avx512_store_master(p, plo, pt, i, vp, sr, idx, s_p);
        avx512_store_moment(m, mvt, i, vm, sr, idx, s_m);
        avx512_store_moment(v, mvt, i, vv, sr, idx, s_v);
        if (ot == DT_F32) _mm512_storeu_ps((float*)out + i, vp);
        else if (ot == DT_BF16) avx512_store_bf16(out, i, avx512_round_rne(vp));
    }
    if (i < n)
        fused_scalar(padv(p, pt, i), ladv(plo, i), pt, cadv(g, gt, i), gt, adv(m, mvt, i), adv(v, mvt, i), mvt,
                     adv(out, ot, i), ot, n - i, off + (uint64_t)i, h);
}

static unsigned long long read_xcr0(void)
{
#  if defined(ADAMBRIDGE_MSVC)
    return _xgetbv(0);
#  else
    unsigned int lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
#  endif
}

static void cpuid_count(unsigned leaf, unsigned sub, unsigned r[4])
{
#  if defined(ADAMBRIDGE_MSVC)
    int x[4];
    __cpuidex(x, (int)leaf, (int)sub);
    r[0] = (unsigned)x[0]; r[1] = (unsigned)x[1]; r[2] = (unsigned)x[2]; r[3] = (unsigned)x[3];
#  else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#  endif
}

#endif /* ADAMBRIDGE_X86 */

/* ------------------------------------------------------------------------- */
/* NEON (AArch64): 4 lanes                                                   */
/* ------------------------------------------------------------------------- */
#if defined(ADAMBRIDGE_NEON)

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

static inline float32x4_t neon_load_master(const void* p, const uint8_t* plo, int pt, int64_t i)
{
    if (pt == DT_F24) {
        const uint32x4_t hi = vshll_n_u16(vld1_u16((const uint16_t*)p + i), 16);
        uint32_t four;
        memcpy(&four, plo + i, 4);   /* exactly 4 bytes: a vld1_u8 would read past the end of the plane */
        const uint8x8_t b8 = vreinterpret_u8_u32(vdup_n_u32(four));
        const uint32x4_t lo = vshlq_n_u32(vmovl_u16(vget_low_u16(vmovl_u8(b8))), 8);
        return vreinterpretq_f32_u32(vorrq_u32(hi, lo));
    }
    return vld1q_f32((const float*)p + i);
}

static inline uint32x4_t neon_round_f24(float32x4_t x, int sr, uint32x4_t idx, uint32x4_t seed)
{
    const uint32x4_t bits = vreinterpretq_u32_f32(x);
    uint32x4_t rounded;
    if (sr) rounded = vaddq_u32(bits, vandq_u32(neon_hash(veorq_u32(idx, seed)), vdupq_n_u32(0xff)));
    else rounded = vaddq_u32(bits, vaddq_u32(vandq_u32(vshrq_n_u32(bits, 8), vdupq_n_u32(1)), vdupq_n_u32(0x7f)));
    rounded = vbicq_u32(rounded, vdupq_n_u32(0xff));
    const uint32x4_t is_nan = vmvnq_u32(vceqq_f32(x, x));
    return vbslq_u32(is_nan, vorrq_u32(bits, vdupq_n_u32(0x00400000)), rounded);
}

static inline float32x4_t neon_store_master(void* p, uint8_t* plo, int pt, int64_t i, float32x4_t x, int sr, uint32x4_t idx, uint32x4_t seed)
{
    if (pt == DT_F24) {
        const uint32x4_t bits = neon_round_f24(x, sr, idx, seed);
        vst1_u16((uint16_t*)p + i, vshrn_n_u32(bits, 16));
        const uint16x4_t mid = vmovn_u32(vshrq_n_u32(bits, 8));   /* bits 8..23 of each lane */
        const uint8x8_t b8 = vmovn_u16(vcombine_u16(mid, mid));    /* their low bytes: b0 b1 b2 b3 b0 b1 b2 b3 */
        const uint32_t four = vget_lane_u32(vreinterpret_u32_u8(b8), 0);
        memcpy(plo + i, &four, 4);
        return vreinterpretq_f32_u32(bits);
    }
    vst1q_f32((float*)p + i, x);
    return x;
}

static void fused_neon(void* p, uint8_t* plo, int pt, const void* g, int gt, void* m, void* v, int mvt,
                       void* out, int ot, int64_t n, uint64_t off, const adambridge_hparams* h)
{
    const float32x4_t b1 = vdupq_n_f32(h->beta1), b2 = vdupq_n_f32(h->beta2);
    const float32x4_t omb1 = vdupq_n_f32(h->one_minus_beta1), omb2 = vdupq_n_f32(h->one_minus_beta2);
    const float32x4_t decay = vdupq_n_f32(h->decay), step = vdupq_n_f32(h->step_size);
    const float32x4_t isbc2 = vdupq_n_f32(h->inv_sqrt_bc2), eps = vdupq_n_f32(h->eps);
    const uint32x4_t s_m = vdupq_n_u32(h->seed), s_v = vdupq_n_u32(seed_v_of(h->seed));
    const uint32x4_t s_p = vdupq_n_u32(seed_p_of(h->seed));
    static const uint32_t lane_init[4] = {0, 1, 2, 3};
    const uint32x4_t lane = vld1q_u32(lane_init);
    const int sr = h->stochastic;

    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t vg = neon_load(g, gt, i);
        float32x4_t vm = neon_load(m, mvt, i);
        float32x4_t vv = neon_load(v, mvt, i);
        float32x4_t vp = vmulq_f32(neon_load_master(p, plo, pt, i), decay);

        vm = vfmaq_f32(vmulq_f32(omb1, vg), b1, vm);
        vv = vfmaq_f32(vmulq_f32(omb2, vmulq_f32(vg, vg)), b2, vv);
        const float32x4_t denom = vfmaq_f32(eps, vsqrtq_f32(vv), isbc2);
        vp = vfmsq_f32(vp, step, vdivq_f32(vm, denom));

        const uint32x4_t idx = vaddq_u32(vdupq_n_u32((uint32_t)(off + (uint64_t)i)), lane);
        vp = neon_store_master(p, plo, pt, i, vp, sr, idx, s_p);
        neon_store_moment(m, mvt, i, vm, sr, idx, s_m);
        neon_store_moment(v, mvt, i, vv, sr, idx, s_v);
        if (ot == DT_F32) vst1q_f32((float*)out + i, vp);
        else if (ot == DT_BF16) vst1_u16((uint16_t*)out + i, vshrn_n_u32(neon_round_rne(vp), 16));
    }
    if (i < n)
        fused_scalar(padv(p, pt, i), ladv(plo, i), pt, cadv(g, gt, i), gt, adv(m, mvt, i), adv(v, mvt, i), mvt,
                     adv(out, ot, i), ot, n - i, off + (uint64_t)i, h);
}

#endif /* ADAMBRIDGE_NEON */

/* ------------------------------------------------------------------------- */
/* Threads, locks, atomics                                                   */
/* ------------------------------------------------------------------------- */
#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
typedef HANDLE ax_thread;
typedef SRWLOCK ax_lock;
typedef CONDITION_VARIABLE ax_cond;
static void ax_lock_init(ax_lock* l) { InitializeSRWLock(l); }
static void ax_lock_acquire(ax_lock* l) { AcquireSRWLockExclusive(l); }
static void ax_lock_release(ax_lock* l) { ReleaseSRWLockExclusive(l); }
static void ax_cond_init(ax_cond* c) { InitializeConditionVariable(c); }
static void ax_cond_wait(ax_cond* c, ax_lock* l) { SleepConditionVariableSRW(c, l, INFINITE, 0); }
static void ax_cond_broadcast(ax_cond* c) { WakeAllConditionVariable(c); }
static void ax_yield(void) { SwitchToThread(); }
#else
#  include <pthread.h>
#  include <sched.h>
typedef pthread_t ax_thread;
typedef pthread_mutex_t ax_lock;
typedef pthread_cond_t ax_cond;
static void ax_lock_init(ax_lock* l) { pthread_mutex_init(l, NULL); }
static void ax_lock_acquire(ax_lock* l) { pthread_mutex_lock(l); }
static void ax_lock_release(ax_lock* l) { pthread_mutex_unlock(l); }
static void ax_cond_init(ax_cond* c) { pthread_cond_init(c, NULL); }
static void ax_cond_wait(ax_cond* c, ax_lock* l) { pthread_cond_wait(c, l); }
static void ax_cond_broadcast(ax_cond* c) { pthread_cond_broadcast(c); }
static void ax_yield(void) { sched_yield(); }
#endif

#if defined(ADAMBRIDGE_MSVC)
static int32_t ax_load(volatile int32_t* p) { return (int32_t)InterlockedCompareExchange((volatile LONG*)p, 0, 0); }
static void ax_store(volatile int32_t* p, int32_t v) { InterlockedExchange((volatile LONG*)p, (LONG)v); }
static int32_t ax_fetch_add(volatile int32_t* p, int32_t v) { return (int32_t)InterlockedExchangeAdd((volatile LONG*)p, (LONG)v); }
static int64_t ax_load64(volatile int64_t* p) { return InterlockedCompareExchange64(p, 0, 0); }
static void ax_store64(volatile int64_t* p, int64_t v) { InterlockedExchange64(p, v); }
#else
static inline int32_t ax_load(volatile int32_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static inline void ax_store(volatile int32_t* p, int32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
static inline int32_t ax_fetch_add(volatile int32_t* p, int32_t v) { return __atomic_fetch_add(p, v, __ATOMIC_ACQ_REL); }
static inline int64_t ax_load64(volatile int64_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static inline void ax_store64(volatile int64_t* p, int64_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
#endif

static inline void ax_relax(void)
{
#if defined(ADAMBRIDGE_X86)
    _mm_pause();
#elif defined(__aarch64__) && !defined(_MSC_VER)
    __asm__ __volatile__("yield");
#endif
}

#if defined(_MSC_VER) && !defined(__clang__)
#  define AX_ALIGN64 __declspec(align(64))
#else
#  define AX_ALIGN64 __attribute__((aligned(64)))
#endif

/* ------------------------------------------------------------------------- */
/* Runtime dispatch                                                          */
/* ------------------------------------------------------------------------- */

static int detect_simd(void)
{
#if defined(ADAMBRIDGE_X86)
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
#elif defined(ADAMBRIDGE_NEON)
    return SIMD_NEON;
#else
    return SIMD_SCALAR;
#endif
}

static int g_detected = -1;
static int g_active = -1;
static volatile int32_t g_ready = 0;

static void build_fp8_luts(void);

static void ensure_init(void)
{
    if (ax_load(&g_ready)) return;
    build_fp8_luts();   /* concurrent first calls compute identical tables */
    g_detected = detect_simd();
    g_active = g_detected;
    ax_store(&g_ready, 1);
}

static void fused_dispatch(void* p, uint8_t* plo, int pt, const void* g, int gt, void* m, void* v, int mvt,
                           void* out, int ot, int64_t n, uint64_t off, const adambridge_hparams* h)
{
    switch (g_active) {
#if defined(ADAMBRIDGE_X86)
    case SIMD_AVX512: fused_avx512(p, plo, pt, g, gt, m, v, mvt, out, ot, n, off, h); break;
    case SIMD_AVX2:   fused_avx2(p, plo, pt, g, gt, m, v, mvt, out, ot, n, off, h); break;
#endif
#if defined(ADAMBRIDGE_NEON)
    case SIMD_NEON:   fused_neon(p, plo, pt, g, gt, m, v, mvt, out, ot, n, off, h); break;
#endif
    default:          fused_scalar(p, plo, pt, g, gt, m, v, mvt, out, ot, n, off, h); break;
    }
}

/* ------------------------------------------------------------------------- */
/* Segments: one tensor's buffers plus its parameter group                   */
/* ------------------------------------------------------------------------- */

typedef struct adambridge_group {
    double lr;
    double beta1;
    double beta2;
    double eps;
    double weight_decay;
    uint64_t seed;       /* base seed for stochastic rounding */
    int32_t stochastic;
    int32_t reserved;
} adambridge_group;

typedef struct adambridge_segment {
    void* p;                 /* master weights: FP32 values, or the FP24 hi plane (uint16) when p_dtype is DT_F24 */
    uint8_t* p_lo;           /* FP24 only: the lo plane (one byte per element); NULL for FP32 */
    const void* g;           /* gradient (g_dtype) */
    void* m;                 /* first moment (mv_dtype; FP8: E4M3 codes) */
    void* v;                 /* second moment (mv_dtype; FP8: E5M2 codes of sqrt(v)) */
    float* m_scale;          /* FP8 only: one scale per 128 elements */
    float* v_scale;
    void* out;               /* optional copy of p in out_dtype (write-back buffer) */
    const adambridge_group* group;
    int64_t n;
    int64_t step;            /* step number of this update (caller increments) */
    uint64_t index;          /* parameter index, selects the rounding stream */
    int64_t offset;          /* element offset of p[0] within the parameter: a chunk of a tensor
                                draws the same rounding bits as the whole tensor would
                                (a multiple of 128 for FP8 moments, whose scales are per block) */
    int32_t g_dtype;
    int32_t mv_dtype;
    int32_t out_dtype;
    int32_t algorithm;       /* ALGO_ADAMW, or ALGO_LION (v and v_scale unused, may be NULL) */
    int32_t p_dtype;         /* DT_F32 (or 0, the default) or DT_F24 */
    int32_t reserved;
} adambridge_segment;

static inline int master_type(const adambridge_segment* s) { return s->p_dtype == DT_F24 ? DT_F24 : DT_F32; }

/* Same mixing as adambridge.kernels.mix_seed(seed, index, step). */
static uint64_t mix64(uint64_t h, uint64_t v)
{
    h = (h ^ v) * 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 31;
    h *= 0x94D049BB133111EBULL;
    h ^= h >> 29;
    return h;
}

static uint32_t step_seed(uint64_t base, uint64_t index, int64_t step)
{
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    h = mix64(h, base);
    h = mix64(h, index);
    h = mix64(h, (uint64_t)step);
    return (uint32_t)h;
}

/* The per-step scalars, computed in double like adambridge.kernels.StepCoefficients. */
static void hparams_for(const adambridge_segment* s, adambridge_hparams* h)
{
    const adambridge_group* G = s->group;
    const double t = (double)s->step;
    const double bc1 = 1.0 - pow(G->beta1, t);
    const double bc2 = 1.0 - pow(G->beta2, t);
    h->beta1 = (float)G->beta1;
    h->beta2 = (float)G->beta2;
    h->one_minus_beta1 = (float)(1.0 - G->beta1);
    h->one_minus_beta2 = (float)(1.0 - G->beta2);
    h->decay = (float)(1.0 - G->lr * G->weight_decay);
    h->step_size = (float)(s->algorithm == ALGO_LION ? G->lr : G->lr / bc1);
    h->inv_sqrt_bc2 = (float)(1.0 / sqrt(bc2));
    h->eps = (float)G->eps;
    h->seed = step_seed(G->seed, s->index, s->step);
    h->stochastic = G->stochastic;
}

/* ------------------------------------------------------------------------- */
/* FP8 moments                                                               */
/*                                                                           */
/* m is stored as E4M3 (the "fn" variant: no infinities, max 448) and        */
/* sqrt(v) as E5M2 (max 57344), each with one FP32 scale per 128-element     */
/* block: scale = block absmax / format max. Layout, arithmetic and the      */
/* stochastic-rounding stream match adambridge.precision.FP8Store, so this        */
/* kernel and the PyTorch one make the same rounding decisions.              */
/* ------------------------------------------------------------------------- */

#define FP8_BLOCK 128

static float g_lut_e4m3[256];
static float g_lut_e5m2[256];

static float fp8_decode(uint32_t c, int mbits, int bias, int ieee_specials)
{
    const uint32_t sign = (c & 0x80u) << 24;
    const int ebits = 7 - mbits;
    const uint32_t emax = (1u << ebits) - 1u;
    const uint32_t e = (c >> mbits) & emax;
    const uint32_t m = c & ((1u << mbits) - 1u);
    if (ieee_specials) {                       /* E5M2: top exponent is inf/NaN */
        if (e == emax) return bits_f32(sign | (m ? 0x7fc00000u : 0x7f800000u));
    } else if (e == emax && m == (1u << mbits) - 1u) {
        return bits_f32(sign | 0x7fc00000u);   /* E4M3fn: only S.1111.111 is NaN */
    }
    if (e == 0) {                              /* subnormal: m * 2^(1 - bias - mbits) */
        const float mag = (float)m * bits_f32((uint32_t)(127 + 1 - bias - mbits) << 23);
        return bits_f32(sign | f32_bits(mag));
    }
    return bits_f32(sign | ((uint32_t)((int)e - bias + 127) << 23) | (m << (23 - mbits)));
}

static void build_fp8_luts(void)
{
    for (uint32_t c = 0; c < 256; ++c) {
        g_lut_e4m3[c] = fp8_decode(c, 3, 7, 0);
        g_lut_e5m2[c] = fp8_decode(c, 2, 15, 1);
    }
}

/* fp32 -> FP8 code, exactly as adambridge.precision does it: optional stochastic
 * truncation of the fp32 bits to the format's mantissa width, clamp to the
 * format max, then round-to-nearest-even into the format. Branch-free so the
 * per-block loops vectorize. */
static inline uint32_t fp8_encode(float x, int sr, uint32_t r, int mbits, int bias, float fmax, uint32_t nan_code)
{
    const int drop = 23 - mbits;
    const uint32_t low = (1u << drop) - 1u;
    const uint32_t b = f32_bits(x);
    const uint32_t sign = (b >> 24) & 0x80u;
    const int is_nan = (b & 0x7fffffffu) > 0x7f800000u;
    const uint32_t t = sr ? ((b + (r & low)) & ~low) : b;
    float a = bits_f32(t & 0x7fffffffu);
    a = a > fmax ? fmax : a;
    const uint32_t ab = f32_bits(a);
    const uint32_t rn = sr ? ab : ab + (low >> 1) + ((ab >> drop) & 1u);
    const uint32_t normal = ((((rn >> 23) & 0xffu) + (uint32_t)(bias - 127)) << mbits) |
                            ((rn >> drop) & ((1u << mbits) - 1u));
    const float sub_scale = bits_f32((uint32_t)(127 + bias + mbits - 1) << 23);
    const uint32_t sub = f32_bits(a * sub_scale + 8388608.0f) - 0x4b000000u;  /* RNE to integer */
    const float min_normal = bits_f32((uint32_t)(127 + 1 - bias) << 23);
    const uint32_t code = a >= min_normal ? normal : sub;
    return sign | (is_nan ? nan_code : code);
}

/* The three per-block steps around the update: decode the codes to FP32,
 * take sqrt(v) and both block maxima, re-encode with the new scales. The
 * generic versions are plain loops; clang vectorizes them, but gcc keeps the
 * table lookups and the encoder scalar, which made FP8 moments slower than
 * BF16 moments in gcc builds (10.9 vs 8.1 ms at 32M parameters; clang build:
 * 7.1 vs 7.9 ms). The AVX2 versions do the same operations with intrinsics
 * and give the same codes bit for bit (checked by the self-test). */
static inline void fp8_decode_block(const uint8_t* mq, const uint8_t* vq, float sm, float sv,
                                    float* mt, float* vt, int nb)
{
    for (int k = 0; k < nb; ++k) mt[k] = g_lut_e4m3[mq[k]] * sm;
    for (int k = 0; k < nb; ++k) { const float u = g_lut_e5m2[vq[k]] * sv; vt[k] = u * u; }
}

/* vt <- sqrt(max(vt, 0)); *am <- max |mt|, *au <- max vt (both >= 0, NaN skipped). */
static inline void fp8_absmax_block(const float* mt, float* vt, int nb, float* am_out, float* au_out)
{
    float am = 0.0f, au = 0.0f;
    for (int k = 0; k < nb; ++k) {
        const float a = fabsf(mt[k]);
        const float u = sqrtf(vt[k] > 0.0f ? vt[k] : 0.0f);
        vt[k] = u;
        am = a > am ? a : am;
        au = u > au ? u : au;
    }
    *am_out = am;
    *au_out = au;
}

static inline void fp8_encode_block(const float* mt, const float* vt, float scm, float scu, int sr, uint32_t idx0,
                                    uint32_t s_m, uint32_t s_v, uint8_t* mq, uint8_t* vq, int nb)
{
    for (int k = 0; k < nb; ++k) {
        const uint32_t idx = idx0 + (uint32_t)k;
        mq[k] = (uint8_t)fp8_encode(mt[k] / scm, sr, hash32(idx ^ s_m), 3, 7, 448.0f, 0x7fu);
        vq[k] = (uint8_t)fp8_encode(vt[k] / scu, sr, hash32(idx ^ s_v), 2, 15, 57344.0f, 0x7eu);
    }
}

#if defined(ADAMBRIDGE_X86)
ADAMBRIDGE_TARGET("avx2,fma")
static inline void fp8_decode_block_avx2(const uint8_t* mq, const uint8_t* vq, float sm, float sv,
                                         float* mt, float* vt, int nb)
{
    const __m256 vsm = _mm256_set1_ps(sm), vsv = _mm256_set1_ps(sv);
    int k = 0;
    for (; k + 8 <= nb; k += 8) {
        const __m256i cm = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(mq + k)));
        const __m256i cv = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(vq + k)));
        _mm256_storeu_ps(mt + k, _mm256_mul_ps(_mm256_i32gather_ps(g_lut_e4m3, cm, 4), vsm));
        const __m256 u = _mm256_mul_ps(_mm256_i32gather_ps(g_lut_e5m2, cv, 4), vsv);
        _mm256_storeu_ps(vt + k, _mm256_mul_ps(u, u));
    }
    for (; k < nb; ++k) {
        mt[k] = g_lut_e4m3[mq[k]] * sm;
        const float u = g_lut_e5m2[vq[k]] * sv;
        vt[k] = u * u;
    }
}

ADAMBRIDGE_TARGET("avx2,fma")
static inline void fp8_absmax_block_avx2(const float* mt, float* vt, int nb, float* am_out, float* au_out)
{
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)), zero = _mm256_setzero_ps();
    __m256 vam = zero, vau = zero;
    int k = 0;
    for (; k + 8 <= nb; k += 8) {
        const __m256 a = _mm256_and_ps(_mm256_loadu_ps(mt + k), absmask);
        const __m256 u = _mm256_sqrt_ps(_mm256_max_ps(_mm256_loadu_ps(vt + k), zero));  /* v > 0 ? v : 0 */
        _mm256_storeu_ps(vt + k, u);
        vam = _mm256_max_ps(a, vam);  /* a > am ? a : am: a NaN lane keeps am */
        vau = _mm256_max_ps(u, vau);
    }
    AX_ALIGN64 float lm[8], lu[8];
    _mm256_store_ps(lm, vam);
    _mm256_store_ps(lu, vau);
    float am = 0.0f, au = 0.0f;  /* lanes are >= 0 and never NaN, so the order of the maxima is irrelevant */
    for (int i = 0; i < 8; ++i) {
        am = lm[i] > am ? lm[i] : am;
        au = lu[i] > au ? lu[i] : au;
    }
    for (; k < nb; ++k) {
        const float a = fabsf(mt[k]);
        const float u = sqrtf(vt[k] > 0.0f ? vt[k] : 0.0f);
        vt[k] = u;
        am = a > am ? a : am;
        au = u > au ? u : au;
    }
    *am_out = am;
    *au_out = au;
}

/* fp8_encode on 8 lanes: the same integer and float operations, in the same order. */
ADAMBRIDGE_TARGET("avx2,fma")
static inline __m256i fp8_encode_avx2(__m256 x, int sr, __m256i r, int mbits, int bias, float fmax, uint32_t nan_code)
{
    const int drop = 23 - mbits;
    const __m128i cdrop = _mm_cvtsi32_si128(drop), cmbits = _mm_cvtsi32_si128(mbits);
    const __m256i low = _mm256_set1_epi32((int)((1u << drop) - 1u));
    const __m256i one = _mm256_set1_epi32(1), absbits = _mm256_set1_epi32(0x7fffffff);
    const __m256i b = _mm256_castps_si256(x);
    const __m256i sign = _mm256_and_si256(_mm256_srli_epi32(b, 24), _mm256_set1_epi32(0x80));
    /* |b| < 2^31, so the signed compare is the unsigned one */
    const __m256i is_nan = _mm256_cmpgt_epi32(_mm256_and_si256(b, absbits), _mm256_set1_epi32(0x7f800000));
    const __m256i t = sr ? _mm256_andnot_si256(low, _mm256_add_epi32(b, _mm256_and_si256(r, low))) : b;
    /* a > fmax ? fmax : a (a NaN stays NaN) */
    const __m256 a = _mm256_min_ps(_mm256_set1_ps(fmax), _mm256_castsi256_ps(_mm256_and_si256(t, absbits)));
    const __m256i ab = _mm256_castps_si256(a);
    const __m256i rn = sr ? ab : _mm256_add_epi32(_mm256_add_epi32(ab, _mm256_srli_epi32(low, 1)),
                                                  _mm256_and_si256(_mm256_srl_epi32(ab, cdrop), one));
    const __m256i expo = _mm256_and_si256(_mm256_srli_epi32(rn, 23), _mm256_set1_epi32(0xff));
    const __m256i normal = _mm256_or_si256(
        _mm256_sll_epi32(_mm256_add_epi32(expo, _mm256_set1_epi32(bias - 127)), cmbits),
        _mm256_and_si256(_mm256_srl_epi32(rn, cdrop), _mm256_set1_epi32((1 << mbits) - 1)));
    const __m256 sub_scale = _mm256_set1_ps(bits_f32((uint32_t)(127 + bias + mbits - 1) << 23));
    const __m256i sub = _mm256_sub_epi32(  /* RNE to integer; the product is exact, so no FMA question */
        _mm256_castps_si256(_mm256_add_ps(_mm256_mul_ps(a, sub_scale), _mm256_set1_ps(8388608.0f))),
        _mm256_set1_epi32(0x4b000000));
    const __m256 min_normal = _mm256_set1_ps(bits_f32((uint32_t)(127 + 1 - bias) << 23));
    const __m256i is_normal = _mm256_castps_si256(_mm256_cmp_ps(a, min_normal, _CMP_GE_OQ));
    const __m256i code = _mm256_blendv_epi8(sub, normal, is_normal);
    return _mm256_or_si256(sign, _mm256_blendv_epi8(code, _mm256_set1_epi32((int)nan_code), is_nan));
}

ADAMBRIDGE_TARGET("avx2,fma")
static inline void fp8_encode_block_avx2(const float* mt, const float* vt, float scm, float scu, int sr,
                                         uint32_t idx0, uint32_t s_m, uint32_t s_v, uint8_t* mq, uint8_t* vq, int nb)
{
    const __m256 vscm = _mm256_set1_ps(scm), vscu = _mm256_set1_ps(scu);
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i vsm = _mm256_set1_epi32((int)s_m), vsv = _mm256_set1_epi32((int)s_v);
    int k = 0;
    for (; k + 8 <= nb; k += 8) {
        __m256i rm = _mm256_setzero_si256(), rv = rm;
        if (sr) {
            const __m256i idx = _mm256_add_epi32(_mm256_set1_epi32((int)(idx0 + (uint32_t)k)), lane);
            rm = avx2_hash(_mm256_xor_si256(idx, vsm));
            rv = avx2_hash(_mm256_xor_si256(idx, vsv));
        }
        avx2_store_codes(mq + k, fp8_encode_avx2(_mm256_div_ps(_mm256_loadu_ps(mt + k), vscm), sr, rm,
                                                 3, 7, 448.0f, 0x7fu));
        avx2_store_codes(vq + k, fp8_encode_avx2(_mm256_div_ps(_mm256_loadu_ps(vt + k), vscu), sr, rv,
                                                 2, 15, 57344.0f, 0x7eu));
    }
    for (; k < nb; ++k) {
        const uint32_t idx = idx0 + (uint32_t)k;
        mq[k] = (uint8_t)fp8_encode(mt[k] / scm, sr, hash32(idx ^ s_m), 3, 7, 448.0f, 0x7fu);
        vq[k] = (uint8_t)fp8_encode(vt[k] / scu, sr, hash32(idx ^ s_v), 2, 15, 57344.0f, 0x7eu);
    }
}
#endif

/* One segment range [lo, hi): lo is a multiple of FP8_BLOCK, hi is a multiple
 * or the end of the segment. Each block is decoded to FP32, updated by the
 * regular SIMD kernel, re-scaled and re-encoded. BLK picks the block steps
 * (empty: the generic loops; _avx2: intrinsics). */
#define FP8_RANGE_BODY(FUSED, BLK)                                                               \
{                                                                                                \
    AX_ALIGN64 float mt[FP8_BLOCK];                                                              \
    AX_ALIGN64 float vt[FP8_BLOCK];                                                              \
    uint8_t* mq = (uint8_t*)s->m;                                                                \
    uint8_t* vq = (uint8_t*)s->v;                                                                \
    const uint32_t s_m = h->seed, s_v = seed_v_of(h->seed);                                      \
    const int sr = h->stochastic, pt = master_type(s);                                           \
    for (int64_t b0 = lo; b0 < hi; b0 += FP8_BLOCK) {                                           \
        const int nb = (int)(hi - b0 < FP8_BLOCK ? hi - b0 : FP8_BLOCK);                         \
        const int64_t blk = b0 / FP8_BLOCK;                                                      \
        fp8_decode_block##BLK(mq + b0, vq + b0, s->m_scale[blk], s->v_scale[blk], mt, vt, nb);  \
        FUSED(padv(s->p, pt, b0), ladv(s->p_lo, b0), pt, cadv(s->g, s->g_dtype, b0), s->g_dtype, \
              mt, vt, DT_F32, adv(s->out, s->out_dtype, b0), s->out_dtype, nb,                  \
              (uint64_t)(s->offset + b0), h);                                                    \
        float am, au;                                                                            \
        fp8_absmax_block##BLK(mt, vt, nb, &am, &au);                                             \
        const float scm = am > 0.0f ? am / 448.0f : 1.0f;                                        \
        const float scu = au > 0.0f ? au / 57344.0f : 1.0f;                                      \
        fp8_encode_block##BLK(mt, vt, scm, scu, sr, (uint32_t)(s->offset + b0), s_m, s_v,       \
                              mq + b0, vq + b0, nb);                                             \
        for (int k = nb; k < FP8_BLOCK; ++k) { mq[b0 + k] = 0; vq[b0 + k] = 0; }               \
        s->m_scale[blk] = scm;                                                                   \
        s->v_scale[blk] = scu;                                                                   \
    }                                                                                            \
}

static void fp8_range_scalar(const adambridge_segment* s, const adambridge_hparams* h, int64_t lo, int64_t hi)
FP8_RANGE_BODY(fused_scalar, )

#if defined(ADAMBRIDGE_X86)
ADAMBRIDGE_TARGET("avx2,fma")
static void fp8_range_avx2(const adambridge_segment* s, const adambridge_hparams* h, int64_t lo, int64_t hi)
FP8_RANGE_BODY(fused_avx2, _avx2)

ADAMBRIDGE_TARGET(ADAMBRIDGE_AVX512)
static void fp8_range_avx512(const adambridge_segment* s, const adambridge_hparams* h, int64_t lo, int64_t hi)
FP8_RANGE_BODY(fused_avx512, )
#endif

#if defined(ADAMBRIDGE_NEON)
static void fp8_range_neon(const adambridge_segment* s, const adambridge_hparams* h, int64_t lo, int64_t hi)
FP8_RANGE_BODY(fused_neon, )
#endif

static void fp8_dispatch(const adambridge_segment* s, const adambridge_hparams* h, int64_t lo, int64_t hi)
{
    switch (g_active) {
#if defined(ADAMBRIDGE_X86)
    case SIMD_AVX512: fp8_range_avx512(s, h, lo, hi); break;
    case SIMD_AVX2:   fp8_range_avx2(s, h, lo, hi); break;
#endif
#if defined(ADAMBRIDGE_NEON)
    case SIMD_NEON:   fp8_range_neon(s, h, lo, hi); break;
#endif
    default:          fp8_range_scalar(s, h, lo, hi); break;
    }
}

/* ------------------------------------------------------------------------- */
/* Lion (Chen et al., 2023, "Symbolic Discovery of Optimization Algorithms") */
/*                                                                           */
/*   c = b1 * m + (1 - b1) * g                                               */
/*   p = p * (1 - lr * wd) - lr * sign(c)                                    */
/*   m = b2 * m + (1 - b2) * g                                               */
/*                                                                           */
/* One moment instead of two: 8 bytes of host state per parameter with FP32  */
/* m (AdamW: 12), and less memory traffic per update. step_size holds lr     */
/* (no bias correction). Every multiply-add is a single fused operation on   */
/* every path, scalar included, so the sign of c -- and with it the update   */
/* -- is bit-identical whatever the ISA. sign(0) = 0, as torch.sign.         */
/* ------------------------------------------------------------------------- */

static void lion_scalar(void* p, uint8_t* plo, int pt, const void* g, int gt, void* m, int mt,
                        void* out, int ot, int64_t n, uint64_t off, const adambridge_hparams* h)
{
    const uint32_t s_p = seed_p_of(h->seed);
    for (int64_t i = 0; i < n; ++i) {
        const float gi = load_scalar(g, gt, i);
        const float mi = load_scalar(m, mt, i);
        const float ci = fmaf(h->beta1, mi, h->one_minus_beta1 * gi);
        const float u = ci > 0.0f ? h->step_size : (ci < 0.0f ? -h->step_size : 0.0f);
        const uint32_t idx = (uint32_t)(off + (uint64_t)i);
        const float pi = store_master(p, plo, pt, i, fmaf(load_master(p, plo, pt, i), h->decay, -u), h->stochastic,
                                      pt == DT_F24 ? hash32(idx ^ s_p) : 0U);

        store_moment_scalar(m, mt, i, fmaf(h->beta2, mi, h->one_minus_beta2 * gi), h->stochastic, hash32(idx ^ h->seed));
        if (ot == DT_F32) ((float*)out)[i] = pi;
        else if (ot == DT_BF16) ((uint16_t*)out)[i] = (uint16_t)(bf16_round_bits(f32_bits(pi)) >> 16);
    }
}

#if defined(ADAMBRIDGE_X86)
ADAMBRIDGE_TARGET("avx2,fma")
static void lion_avx2(void* p, uint8_t* plo, int pt, const void* g, int gt, void* m, int mt,
                      void* out, int ot, int64_t n, uint64_t off, const adambridge_hparams* h)
{
    const __m256 b1 = _mm256_set1_ps(h->beta1), b2 = _mm256_set1_ps(h->beta2);
    const __m256 omb1 = _mm256_set1_ps(h->one_minus_beta1), omb2 = _mm256_set1_ps(h->one_minus_beta2);
    const __m256 decay = _mm256_set1_ps(h->decay), lr = _mm256_set1_ps(h->step_size);
    const __m256 zero = _mm256_setzero_ps();
    const __m256i s_m = _mm256_set1_epi32((int)h->seed), s_p = _mm256_set1_epi32((int)seed_p_of(h->seed));
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const int sr = h->stochastic;

    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 vg = avx2_load(g, gt, i);
        const __m256 vm = avx2_load(m, mt, i);
        const __m256 vc = _mm256_fmadd_ps(b1, vm, _mm256_mul_ps(omb1, vg));
        const __m256 upd = _mm256_sub_ps(_mm256_and_ps(_mm256_cmp_ps(vc, zero, _CMP_GT_OQ), lr),
                                         _mm256_and_ps(_mm256_cmp_ps(vc, zero, _CMP_LT_OQ), lr));
        const __m256i idx = _mm256_add_epi32(_mm256_set1_epi32((int)(uint32_t)(off + (uint64_t)i)), lane);
        const __m256 vp = avx2_store_master(p, plo, pt, i, _mm256_fmsub_ps(avx2_load_master(p, plo, pt, i), decay, upd),
                                            sr, idx, s_p);

        avx2_store_moment(m, mt, i, _mm256_fmadd_ps(b2, vm, _mm256_mul_ps(omb2, vg)), sr, idx, s_m);
        if (ot == DT_F32) _mm256_storeu_ps((float*)out + i, vp);
        else if (ot == DT_BF16) _mm_storeu_si128((__m128i*)((uint16_t*)out + i), avx2_pack_hi16(avx2_round_rne(vp)));
    }
    if (i < n)
        lion_scalar(padv(p, pt, i), ladv(plo, i), pt, cadv(g, gt, i), gt, adv(m, mt, i), mt, adv(out, ot, i), ot, n - i,
                    off + (uint64_t)i, h);
}

ADAMBRIDGE_TARGET(ADAMBRIDGE_AVX512)
static void lion_avx512(void* p, uint8_t* plo, int pt, const void* g, int gt, void* m, int mt,
                        void* out, int ot, int64_t n, uint64_t off, const adambridge_hparams* h)
{
    const __m512 b1 = _mm512_set1_ps(h->beta1), b2 = _mm512_set1_ps(h->beta2);
    const __m512 omb1 = _mm512_set1_ps(h->one_minus_beta1), omb2 = _mm512_set1_ps(h->one_minus_beta2);
    const __m512 decay = _mm512_set1_ps(h->decay), lr = _mm512_set1_ps(h->step_size);
    const __m512 zero = _mm512_setzero_ps();
    const __m512i s_m = _mm512_set1_epi32((int)h->seed), s_p = _mm512_set1_epi32((int)seed_p_of(h->seed));
    const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const int sr = h->stochastic;

    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 vg = avx512_load(g, gt, i);
        const __m512 vm = avx512_load(m, mt, i);
        const __m512 vc = _mm512_fmadd_ps(b1, vm, _mm512_mul_ps(omb1, vg));
        const __m512 upd = _mm512_sub_ps(_mm512_maskz_mov_ps(_mm512_cmp_ps_mask(vc, zero, _CMP_GT_OQ), lr),
                                         _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(vc, zero, _CMP_LT_OQ), lr));
        const __m512i idx = _mm512_add_epi32(_mm512_set1_epi32((int)(uint32_t)(off + (uint64_t)i)), lane);
        const __m512 vp = avx512_store_master(p, plo, pt, i, _mm512_fmsub_ps(avx512_load_master(p, plo, pt, i), decay, upd),
                                              sr, idx, s_p);

        avx512_store_moment(m, mt, i, _mm512_fmadd_ps(b2, vm, _mm512_mul_ps(omb2, vg)), sr, idx, s_m);
        if (ot == DT_F32) _mm512_storeu_ps((float*)out + i, vp);
        else if (ot == DT_BF16) avx512_store_bf16(out, i, avx512_round_rne(vp));
    }
    if (i < n)
        lion_scalar(padv(p, pt, i), ladv(plo, i), pt, cadv(g, gt, i), gt, adv(m, mt, i), mt, adv(out, ot, i), ot, n - i,
                    off + (uint64_t)i, h);
}
#endif /* ADAMBRIDGE_X86 */

#if defined(ADAMBRIDGE_NEON)
static void lion_neon(void* p, uint8_t* plo, int pt, const void* g, int gt, void* m, int mt,
                      void* out, int ot, int64_t n, uint64_t off, const adambridge_hparams* h)
{
    const float32x4_t b1 = vdupq_n_f32(h->beta1), b2 = vdupq_n_f32(h->beta2);
    const float32x4_t omb1 = vdupq_n_f32(h->one_minus_beta1), omb2 = vdupq_n_f32(h->one_minus_beta2);
    const float32x4_t decay = vdupq_n_f32(h->decay);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const uint32x4_t lr = vreinterpretq_u32_f32(vdupq_n_f32(h->step_size));
    const uint32x4_t s_m = vdupq_n_u32(h->seed), s_p = vdupq_n_u32(seed_p_of(h->seed));
    static const uint32_t lane_init[4] = {0, 1, 2, 3};
    const uint32x4_t lane = vld1q_u32(lane_init);
    const int sr = h->stochastic;

    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t vg = neon_load(g, gt, i);
        const float32x4_t vm = neon_load(m, mt, i);
        const float32x4_t vc = vfmaq_f32(vmulq_f32(omb1, vg), b1, vm);
        const float32x4_t upd = vsubq_f32(vreinterpretq_f32_u32(vandq_u32(vcgtq_f32(vc, zero), lr)),
                                          vreinterpretq_f32_u32(vandq_u32(vcltq_f32(vc, zero), lr)));
        const uint32x4_t idx = vaddq_u32(vdupq_n_u32((uint32_t)(off + (uint64_t)i)), lane);
        const float32x4_t vp = neon_store_master(p, plo, pt, i, vfmaq_f32(vnegq_f32(upd), neon_load_master(p, plo, pt, i), decay),
                                                 sr, idx, s_p);

        neon_store_moment(m, mt, i, vfmaq_f32(vmulq_f32(omb2, vg), b2, vm), sr, idx, s_m);
        if (ot == DT_F32) vst1q_f32((float*)out + i, vp);
        else if (ot == DT_BF16) vst1_u16((uint16_t*)out + i, vshrn_n_u32(neon_round_rne(vp), 16));
    }
    if (i < n)
        lion_scalar(padv(p, pt, i), ladv(plo, i), pt, cadv(g, gt, i), gt, adv(m, mt, i), mt, adv(out, ot, i), ot, n - i,
                    off + (uint64_t)i, h);
}
#endif /* ADAMBRIDGE_NEON */

static void lion_dispatch(void* p, uint8_t* plo, int pt, const void* g, int gt, void* m, int mt,
                          void* out, int ot, int64_t n, uint64_t off, const adambridge_hparams* h)
{
    switch (g_active) {
#if defined(ADAMBRIDGE_X86)
    case SIMD_AVX512: lion_avx512(p, plo, pt, g, gt, m, mt, out, ot, n, off, h); break;
    case SIMD_AVX2:   lion_avx2(p, plo, pt, g, gt, m, mt, out, ot, n, off, h); break;
#endif
#if defined(ADAMBRIDGE_NEON)
    case SIMD_NEON:   lion_neon(p, plo, pt, g, gt, m, mt, out, ot, n, off, h); break;
#endif
    default:          lion_scalar(p, plo, pt, g, gt, m, mt, out, ot, n, off, h); break;
    }
}

/* Lion with FP8 m: the same block layout, codec and rounding stream as the
 * AdamW first moment. */
static void lion_fp8_range(const adambridge_segment* s, const adambridge_hparams* h, int64_t lo, int64_t hi)
{
    AX_ALIGN64 float mt[FP8_BLOCK];
    uint8_t* mq = (uint8_t*)s->m;
    const int sr = h->stochastic, pt = master_type(s);
    for (int64_t b0 = lo; b0 < hi; b0 += FP8_BLOCK) {
        const int nb = (int)(hi - b0 < FP8_BLOCK ? hi - b0 : FP8_BLOCK);
        const int64_t blk = b0 / FP8_BLOCK;
        const float sm = s->m_scale[blk];
        for (int k = 0; k < nb; ++k) mt[k] = g_lut_e4m3[mq[b0 + k]] * sm;
        lion_dispatch(padv(s->p, pt, b0), ladv(s->p_lo, b0), pt, cadv(s->g, s->g_dtype, b0), s->g_dtype, mt, DT_F32,
                      adv(s->out, s->out_dtype, b0), s->out_dtype, nb, (uint64_t)(s->offset + b0), h);
        float am = 0.0f;
        for (int k = 0; k < nb; ++k) { const float a = fabsf(mt[k]); am = a > am ? a : am; }
        const float scm = am > 0.0f ? am / 448.0f : 1.0f;
        for (int k = 0; k < nb; ++k) {
            const uint32_t idx = (uint32_t)(s->offset + b0 + k);
            mq[b0 + k] = (uint8_t)fp8_encode(mt[k] / scm, sr, hash32(idx ^ h->seed), 3, 7, 448.0f, 0x7fu);
        }
        for (int k = nb; k < FP8_BLOCK; ++k) mq[b0 + k] = 0;
        s->m_scale[blk] = scm;
    }
}

/* ------------------------------------------------------------------------- */
/* Work items and the thread pool                                            */
/* ------------------------------------------------------------------------- */

typedef struct ax_item {
    const adambridge_segment* seg;
    const adambridge_hparams* h;
    int64_t lo;
    int64_t hi;
} ax_item;

typedef struct ax_job {
    const ax_item* items;
    int32_t nitems;
} ax_job;

static void run_item(const ax_item* it)
{
    const adambridge_segment* s = it->seg;
    const int64_t lo = it->lo;
    const int pt = master_type(s);
    if (s->algorithm == ALGO_LION) {
        if (s->mv_dtype == DT_FP8) lion_fp8_range(s, it->h, lo, it->hi);
        else lion_dispatch(padv(s->p, pt, lo), ladv(s->p_lo, lo), pt, cadv(s->g, s->g_dtype, lo), s->g_dtype,
                           adv(s->m, s->mv_dtype, lo), s->mv_dtype, adv(s->out, s->out_dtype, lo), s->out_dtype,
                           it->hi - lo, (uint64_t)(s->offset + lo), it->h);
        return;
    }
    if (s->mv_dtype == DT_FP8) {
        fp8_dispatch(s, it->h, lo, it->hi);
        return;
    }
    fused_dispatch(padv(s->p, pt, lo), ladv(s->p_lo, lo), pt, cadv(s->g, s->g_dtype, lo), s->g_dtype,
                   adv(s->m, s->mv_dtype, lo), adv(s->v, s->mv_dtype, lo), s->mv_dtype,
                   adv(s->out, s->out_dtype, lo), s->out_dtype, it->hi - lo, (uint64_t)(s->offset + lo), it->h);
}

typedef struct adambridge_pool adambridge_pool;

typedef struct ax_helper {
    adambridge_pool* pool;
    int32_t id;              /* 1 .. nthreads-1; the calling thread is 0 */
    ax_lock lock;            /* each helper sleeps on its own cv, so a job */
    ax_cond cv;              /* wakes only the helpers it needs */
} ax_helper;

struct adambridge_pool {
    int32_t nthreads;        /* helpers + the calling thread */
    ax_thread* threads;
    ax_helper* helpers;
    int32_t* cpus;           /* affinity set for helpers (may be empty) */
    int32_t ncpus;
    ax_lock run_lock;        /* one job at a time */
    volatile int64_t ticket; /* (generation << 32) | helpers taking part */
    volatile int32_t remaining;
    volatile int32_t next;
    volatile int32_t stop;
    int32_t spin;
    int64_t generation;
    const ax_job* job;
};

static void pool_work(adambridge_pool* P, const ax_job* job)
{
    for (;;) {
        const int32_t i = ax_fetch_add(&P->next, 1);
        if (i >= job->nitems) return;
        run_item(&job->items[i]);
    }
}

static void pin_current_thread(const adambridge_pool* P)
{
    if (P->ncpus <= 0) return;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int32_t i = 0; i < P->ncpus; ++i)
        if (P->cpus[i] >= 0 && P->cpus[i] < CPU_SETSIZE) CPU_SET(P->cpus[i], &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#elif defined(_WIN32)
    DWORD_PTR mask = 0;
    for (int32_t i = 0; i < P->ncpus; ++i)
        if (P->cpus[i] >= 0 && P->cpus[i] < (int32_t)(8 * sizeof(DWORD_PTR))) mask |= (DWORD_PTR)1 << P->cpus[i];
    if (mask) SetThreadAffinityMask(GetCurrentThread(), mask);
#endif
}

/* True when ticket t is a job this helper takes part in. Participation and
 * generation come from one atomic snapshot, so a late helper can never act
 * on another job's helper count. */
static inline int wanted(const ax_helper* me, int64_t t) { return me->id <= (int32_t)(t & 0xffffffff); }

static void helper_main(ax_helper* me)
{
    adambridge_pool* P = me->pool;
    pin_current_thread(P);
    /* The ticket the pool was created with (no job yet), not the current one: a
     * job posted before this thread got to run must still be seen as new. */
    int64_t seen = 0;
    int32_t spin = 0;                    /* spin only right after real work */
    for (;;) {
        int64_t t;
        for (;;) {
            if (ax_load(&P->stop)) return;
            t = ax_load64(&P->ticket);
            if (t != seen) {
                seen = t;
                if (wanted(me, t)) break;
                continue;                /* a job for other helpers: keep waiting */
            }
            if (spin > 0) { --spin; ax_relax(); continue; }
            ax_lock_acquire(&me->lock);
            while (ax_load64(&P->ticket) == seen && !ax_load(&P->stop)) ax_cond_wait(&me->cv, &me->lock);
            ax_lock_release(&me->lock);
        }
        pool_work(P, P->job);
        ax_fetch_add(&P->remaining, -1);
        spin = P->spin;
    }
}

#if defined(_WIN32)
static DWORD WINAPI helper_entry(LPVOID arg) { helper_main((ax_helper*)arg); return 0; }
#else
static void* helper_entry(void* arg) { helper_main((ax_helper*)arg); return NULL; }
#endif

static void pool_run(adambridge_pool* P, const ax_job* job, int32_t threads)
{
    if (!P || threads <= 1 || job->nitems <= 1 || P->nthreads <= 1) {
        for (int32_t i = 0; i < job->nitems; ++i) run_item(&job->items[i]);
        return;
    }
    ax_lock_acquire(&P->run_lock);
    int32_t helpers = threads - 1;
    if (helpers > P->nthreads - 1) helpers = P->nthreads - 1;
    if (helpers > job->nitems - 1) helpers = job->nitems - 1;
    P->job = job;
    ax_store(&P->next, 0);
    ax_store(&P->remaining, helpers);
    P->generation += 1;
    ax_store64(&P->ticket, (P->generation << 32) | (int64_t)helpers);
    for (int32_t i = 0; i < helpers; ++i) {  /* wake exactly the helpers taking part */
        ax_helper* h = &P->helpers[i];
        ax_lock_acquire(&h->lock);
        ax_cond_broadcast(&h->cv);
        ax_lock_release(&h->lock);
    }
    pool_work(P, job);
    for (int32_t spins = 0; ax_load(&P->remaining) > 0; ++spins) {
        if (spins < (1 << 16)) ax_relax();
        else ax_yield();
    }
    ax_lock_release(&P->run_lock);
}

/* Threads that update segments: `nthreads` includes the calling thread, so
 * nthreads - 1 helpers are started. cpus (may be NULL) pins the helpers. */
ADAMBRIDGE_API void* adambridge_pool_create(int32_t nthreads, const int32_t* cpus, int32_t ncpus)
{
    ensure_init();
    if (nthreads < 1) nthreads = 1;
    adambridge_pool* P = (adambridge_pool*)calloc(1, sizeof(adambridge_pool));
    if (!P) return NULL;
    P->nthreads = nthreads;
    P->spin = 4000;
    ax_lock_init(&P->run_lock);
    if (cpus && ncpus > 0) {
        P->cpus = (int32_t*)malloc(sizeof(int32_t) * (size_t)ncpus);
        if (P->cpus) { memcpy(P->cpus, cpus, sizeof(int32_t) * (size_t)ncpus); P->ncpus = ncpus; }
    }
    if (nthreads > 1) {
        P->threads = (ax_thread*)calloc((size_t)nthreads - 1, sizeof(ax_thread));
        P->helpers = (ax_helper*)calloc((size_t)nthreads - 1, sizeof(ax_helper));
        if (!P->threads || !P->helpers) { P->nthreads = 1; return P; }
        for (int32_t i = 0; i < nthreads - 1; ++i) {
            P->helpers[i].pool = P;
            P->helpers[i].id = i + 1;
            ax_lock_init(&P->helpers[i].lock);
            ax_cond_init(&P->helpers[i].cv);
#if defined(_WIN32)
            P->threads[i] = CreateThread(NULL, 0, helper_entry, &P->helpers[i], 0, NULL);
            if (!P->threads[i]) { P->nthreads = i + 1; break; }
#else
            if (pthread_create(&P->threads[i], NULL, helper_entry, &P->helpers[i]) != 0) { P->nthreads = i + 1; break; }
#endif
        }
    }
    return P;
}

ADAMBRIDGE_API void adambridge_pool_destroy(void* pool)
{
    adambridge_pool* P = (adambridge_pool*)pool;
    if (!P) return;
    ax_store(&P->stop, 1);
    for (int32_t i = 0; i < P->nthreads - 1; ++i) {
        ax_lock_acquire(&P->helpers[i].lock);
        ax_cond_broadcast(&P->helpers[i].cv);
        ax_lock_release(&P->helpers[i].lock);
    }
    for (int32_t i = 0; i < P->nthreads - 1; ++i) {
#if defined(_WIN32)
        WaitForSingleObject(P->threads[i], INFINITE);
        CloseHandle(P->threads[i]);
#else
        pthread_join(P->threads[i], NULL);
#endif
    }
    free(P->threads);
    free(P->helpers);
    free(P->cpus);
    free(P);
}

ADAMBRIDGE_API int32_t adambridge_pool_threads(void* pool) { return pool ? ((adambridge_pool*)pool)->nthreads : 1; }

/* How long helpers spin (in pause instructions) before sleeping between jobs. */
ADAMBRIDGE_API void adambridge_pool_set_spin(void* pool, int32_t spin) { if (pool) ((adambridge_pool*)pool)->spin = spin < 0 ? 0 : spin; }

/* ------------------------------------------------------------------------- */
/* Public entry points                                                       */
/* ------------------------------------------------------------------------- */

ADAMBRIDGE_API int adambridge_abi_version(void) { return ADAMBRIDGE_ABI_VERSION; }

ADAMBRIDGE_API int adambridge_detected_simd_level(void) { ensure_init(); return g_detected; }

ADAMBRIDGE_API int adambridge_simd_level(void) { ensure_init(); return g_active; }

/* Restrict the active ISA (e.g. to benchmark or test the scalar path).
 * A level the CPU does not support falls back to the detected one.
 * Returns the level now active. */
ADAMBRIDGE_API int adambridge_set_simd_level(int level)
{
    ensure_init();
    if (level < 0) g_active = g_detected;
    else if (level == SIMD_SCALAR) g_active = SIMD_SCALAR;
    else if (g_detected == SIMD_AVX512 && (level == SIMD_AVX2 || level == SIMD_AVX512)) g_active = level;
    else g_active = g_detected;
    return g_active;
}

ADAMBRIDGE_API int adambridge_hparams_size(void) { return (int)sizeof(adambridge_hparams); }
ADAMBRIDGE_API int adambridge_group_size(void) { return (int)sizeof(adambridge_group); }
ADAMBRIDGE_API int adambridge_segment_size(void) { return (int)sizeof(adambridge_segment); }

/* Returns 0 on success, negative on invalid arguments. Single-threaded,
 * FP32/BF16 moments only (kept for tests and simple callers). */
ADAMBRIDGE_API int adambridge_fused_adamw(float* p, const void* g, int g_dtype,
                                          void* m, void* v, int mv_dtype,
                                          void* out, int out_dtype,
                                          int64_t n, uint64_t offset, const adambridge_hparams* hp)
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
    fused_dispatch(p, NULL, DT_F32, g, g_dtype, m, v, mv_dtype, out, out_dtype, n, offset, hp);
    return 0;
}

static int check_segment(const adambridge_segment* s, int need_group)
{
    if (!s || s->n < 0) return -1;
    if (s->algorithm != ALGO_ADAMW && s->algorithm != ALGO_LION) return -12;
    if (s->n == 0) return 0;
    const int lion = s->algorithm == ALGO_LION;
    if (s->p_dtype != DT_NONE && s->p_dtype != DT_F32 && s->p_dtype != DT_F24) return -13;
    if (!s->p || !s->g || !s->m || (!lion && !s->v) || (s->p_dtype == DT_F24 && !s->p_lo)) return -2;
    if (s->g_dtype != DT_F32 && s->g_dtype != DT_BF16) return -3;
    if (s->mv_dtype != DT_F32 && s->mv_dtype != DT_BF16 && s->mv_dtype != DT_FP8) return -4;
    if (s->out_dtype != DT_NONE && s->out_dtype != DT_F32 && s->out_dtype != DT_BF16) return -5;
    if (s->out_dtype != DT_NONE && !s->out) return -6;
    if (s->mv_dtype == DT_FP8 && (!s->m_scale || (!lion && !s->v_scale))) return -7;
    if (need_group && !s->group) return -8;
    if (s->offset < 0 || (s->mv_dtype == DT_FP8 && s->offset % FP8_BLOCK)) return -11;
    return 0;
}

/* Split segments into work items: about 4 per thread for load balance, never
 * smaller than 32K elements, always starting on a 128-element boundary (the
 * FP8 block; also keeps the SIMD/tail split independent of the thread count). */
#define AX_MIN_PIECE 32768
/* At most one thread per this many elements of a job: waking a helper costs more than it saves below it
 * (WSL, 2026-11-06: 0.26M elements took 0.065 ms on 2 threads and 0.21 ms on 24; 1M: 0.20 ms on 8, 0.28 on 24;
 * from 4M on, every thread helps). Results do not depend on the thread count. */
#define AX_PER_THREAD 131072

static int run_segments(adambridge_pool* P, adambridge_segment* const* segs, const adambridge_hparams* hps,
                        int32_t nsegs, int32_t threads)
{
    if (threads <= 0) threads = P ? P->nthreads : 1;
    int64_t elements = 0;
    for (int32_t i = 0; i < nsegs; ++i)
        if (segs[i]->n > 0) elements += segs[i]->n;
    if (elements / AX_PER_THREAD < threads) threads = elements / AX_PER_THREAD > 1 ? (int32_t)(elements / AX_PER_THREAD) : 1;
    int64_t total = 0;
    for (int32_t i = 0; i < nsegs; ++i) {
        const int64_t n = segs[i]->n;
        if (n <= 0) continue;
        int64_t piece = (n + (int64_t)threads * 4 - 1) / ((int64_t)threads * 4);
        piece = (piece + FP8_BLOCK - 1) / FP8_BLOCK * FP8_BLOCK;
        if (piece < AX_MIN_PIECE) piece = AX_MIN_PIECE;
        total += (n + piece - 1) / piece;
    }
    if (total == 0) return 0;
    if (total > 0x7fffffff) return -9;
    ax_item* items = (ax_item*)malloc(sizeof(ax_item) * (size_t)total);
    if (!items) return -10;
    int32_t k = 0;
    for (int32_t i = 0; i < nsegs; ++i) {
        const int64_t n = segs[i]->n;
        if (n <= 0) continue;
        int64_t piece = (n + (int64_t)threads * 4 - 1) / ((int64_t)threads * 4);
        piece = (piece + FP8_BLOCK - 1) / FP8_BLOCK * FP8_BLOCK;
        if (piece < AX_MIN_PIECE) piece = AX_MIN_PIECE;
        for (int64_t lo = 0; lo < n; lo += piece) {
            items[k].seg = segs[i];
            items[k].h = &hps[i];
            items[k].lo = lo;
            items[k].hi = lo + piece < n ? lo + piece : n;
            ++k;
        }
    }
    ax_job job = { items, k };
    pool_run(P, &job, threads < k ? threads : k);
    free(items);
    return 0;
}

/* Update `nsegs` segments, parallel over `threads` threads of `pool` (NULL:
 * the calling thread only). Each segment's coefficients come from its group
 * and its step field, which the caller sets to this update's step number.
 * Validates every segment first; on error nothing is updated and the
 * negative code of the first bad segment is returned. */
ADAMBRIDGE_API int adambridge_multi_step(void* pool, adambridge_segment* const* segs, int32_t nsegs, int32_t threads)
{
    if (nsegs <= 0) return 0;
    if (!segs) return -1;
    ensure_init();
    for (int32_t i = 0; i < nsegs; ++i) {
        const int rc = check_segment(segs[i], 1);
        if (rc) return rc;
    }
    adambridge_hparams* hps = (adambridge_hparams*)malloc(sizeof(adambridge_hparams) * (size_t)nsegs);
    if (!hps) return -10;
    for (int32_t i = 0; i < nsegs; ++i) hparams_for(segs[i], &hps[i]);
    const int rc = run_segments((adambridge_pool*)pool, segs, hps, nsegs, threads);
    free(hps);
    return rc;
}

/* Same as adambridge_multi_step with caller-supplied coefficients (hps[i] for
 * segs[i]); groups and steps are ignored. Used by tests and benchmarks. */
ADAMBRIDGE_API int adambridge_multi_apply(void* pool, adambridge_segment* const* segs, const adambridge_hparams* hps,
                                          int32_t nsegs, int32_t threads)
{
    if (nsegs <= 0) return 0;
    if (!segs || !hps) return -1;
    ensure_init();
    for (int32_t i = 0; i < nsegs; ++i) {
        const int rc = check_segment(segs[i], 0);
        if (rc) return rc;
    }
    return run_segments((adambridge_pool*)pool, segs, hps, nsegs, threads);
}
