#include "core.h"
#include <immintrin.h>

template <typename T>
void cast_to_pad_AVX2(EngineParams& ep, unsigned int thread_id, int plane, const uint8_t* ptr, int depth, int stride, int width, int height);

template <typename T>
void cast_from_pad_AVX2(EngineParams& ep, unsigned int thread_id, int plane, uint8_t* ptr, int depth, int stride, int width, int height);

template <>
void cast_to_pad_AVX2<uint8_t>(EngineParams& ep, unsigned int thread_id, int plane, const uint8_t* ptr, int depth, int stride, int width, int height)
{
  auto pad = ep.padBuffer[plane][thread_id];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 16) {
      auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(ptr + x));
      auto src32 = _mm256_cvtepu8_epi32(src);
      _mm256_store_ps(pad + x, _mm256_cvtepi32_ps(src32));
      src = _mm_srli_si128(src, 8);
      src32 = _mm256_cvtepu8_epi32(src);
      _mm256_store_ps(pad + x + 8, _mm256_cvtepi32_ps(src32));
    }
    ptr += stride;
    pad += ep.padStrideElements[plane];
  }
}

template <>
void cast_to_pad_AVX2<uint16_t>(EngineParams& ep, unsigned int thread_id, int plane, const uint8_t* ptr, int depth, int stride, int width, int height)
{
  auto pad = ep.padBuffer[plane][thread_id];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 8) {
      auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(ptr + x * 2));
      auto src32 = _mm256_cvtepu16_epi32(src);
      _mm256_store_ps(pad + x, _mm256_cvtepi32_ps(src32));
    }
    ptr += stride;
    pad += ep.padStrideElements[plane];
  }
}

template <>
void cast_from_pad_AVX2<uint8_t>(EngineParams& ep, unsigned int thread_id, int plane, uint8_t* ptr, int depth, int stride, int width, int height)
{
  auto pad = ep.padBuffer[plane][thread_id];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 32) {
      auto src1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_load_ps(pad + x), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
      auto src2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_load_ps(pad + x + 8), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
      auto src3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_load_ps(pad + x + 16), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
      auto src4 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_load_ps(pad + x + 24), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
      auto dst1 = _mm256_packus_epi32(src1, src2);
      dst1 = _mm256_permute4x64_epi64(dst1, 216);
      auto dst2 = _mm256_packus_epi32(src3, src4);
      dst2 = _mm256_permute4x64_epi64(dst2, 216);
      auto dst = _mm256_packus_epi16(dst1, dst2);
      dst = _mm256_permute4x64_epi64(dst, 216);
      if (x + 8 >= width)
        _mm_storel_epi64(reinterpret_cast<__m128i*>(ptr + x), _mm256_castsi256_si128(dst));
      else if (x + 16 >= width)
        _mm_store_si128(reinterpret_cast<__m128i*>(ptr + x), _mm256_castsi256_si128(dst));
      else
        _mm256_store_si256(reinterpret_cast<__m256i*>(ptr + x), dst);
    }
    ptr += stride;
    pad += ep.padStrideElements[plane];
  }
}

template <>
void cast_from_pad_AVX2<uint16_t>(EngineParams& ep, unsigned int thread_id, int plane, uint8_t* ptr, int depth, int stride, int width, int height)
{
  auto peak = _mm256_set1_epi16(ep.peak);
  auto pad = ep.padBuffer[plane][thread_id];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 16) {
      auto src1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_load_ps(pad + x), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
      auto src2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_load_ps(pad + x + 8), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
      auto dst = _mm256_packus_epi32(src1, src2);
      dst = _mm256_permute4x64_epi64(dst, 216);
      _mm256_store_si256(reinterpret_cast<__m256i*>(ptr + x * 2), _mm256_min_epu16(dst, peak));
    }
    ptr += stride;
    pad += ep.padStrideElements[plane];
  }
}
