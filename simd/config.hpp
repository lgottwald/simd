#ifndef _SIMD_CONFIG_HPP_
#define _SIMD_CONFIG_HPP_

#include <cstddef>

#if defined(__LP64__) || defined(_WIN64)
#define MALLOC_ALIGNMENT 16
#else
#define MALLOC_ALIGNMENT 8
#endif

#include <immintrin.h>
#include "traits.hpp"
/* set different simd sizes and intrinsics names for avx/sse and define correct types */
#if defined(__AVX__) || defined(__AVX2__)

/**
 * Gives the proper name of a simd intrinsic. I.e.
 * for SSE SIMD_INTRIN(add_pd) is substituted by
 * _mm_add_pd and for AVX it is subtituted by
 * _mm256_add_pd
 */
#define SIMD_INTRIN(func) _mm256_##func

/**
 * Evaluates to empty string if AVX instructions
 * are not enabled.
 */
#define IF_AVX(command) command

/**
 * Evaluates to empty string if AVX instructions
 * instread of SSE instructions are enabled.
 */
#define IF_SSE(command)

namespace simd {

/**
 * Returns the alignment requirement for operations in this header.
 * I.e. if this function returns 16 all functions requesting an
 * aligned pointer expect a 16 byte aligned pointer.
 * aligned_alloc is guaranteed to allocate a pointer that fullfills
 * this alignment.
 */
constexpr std::size_t alignment() { return 32; }

template<typename T>
constexpr std::size_t pack_size() { return 32/sizeof(T); }

}

typedef  __m256d xmmd_t;
typedef  __m256 xmm_t;

#elif defined (__SSE2__)

/**
 * Gives the proper name of a simd intrinsic. I.e.
 * for SSE SIMD_INTRIN(add_pd) is substituted by
 * _mm_add_pd and for AVX it is subtituted by
 * _mm256_add_pd
 */
#define SIMD_INTRIN(func) _mm_##func

/**
 * Evaluates to empty string if AVX instructions
 * are not enabled.
 */
#define IF_AVX(command)

/**
 * Evaluates to empty string if AVX instructions
 * instread of SSE instructions are enabled.
 */
#define IF_SSE(command) command

namespace simd {

/**
 * Returns the alignment requirement for operations in this header.
 * I.e. if this function returns 16 all functions requesting an
 * aligned pointer expect a 16 byte aligned pointer.
 * aligned_alloc is guaranteed to allocate a pointer that fullfills
 * this alignment.
 */
constexpr std::size_t alignment() { return 16; }

template<typename T>
constexpr std::size_t pack_size() { return 16/sizeof(T); }

}

typedef  __m128d xmmd_t;
typedef  __m128 xmm_t;

#else

/**
 * Scalar replacements not supported since SSE2
 * is widely supported.
 */
#error "At least SSE2 instructions are required for SIMD wrapper"
#endif

/**
 * Asserts that the given ptr is suitably aligned for simd operations that have
 * special alignment requirements.
 */
#define ASSERT_ALIGNED(ptr) assert( reinterpret_cast<uintptr_t>(ptr) % simd::alignment() == 0 )

#if defined(_MSC_VER)

#include <intrin.h>
#define __builtin_popcount __popcnt

#endif

constexpr std::size_t cache_alignment() { return 64; }

#endif