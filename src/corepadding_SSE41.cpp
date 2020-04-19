#include "core.h"
#include <smmintrin.h>

template <typename T>
void cast_to_pad_SSE41(EngineParams& ep, unsigned int thread_id, int plane, const uint8_t* ptr, int depth, int stride, int width, int height);

template <typename T>
void cast_from_pad_SSE41(EngineParams& ep, unsigned int thread_id, int plane, uint8_t* ptr, int depth, int stride, int width, int height);

template <>
void cast_to_pad_SSE41<uint8_t>(EngineParams& ep, unsigned int thread_id, int plane, const uint8_t* ptr, int depth, int stride, int width, int height)
{
  auto pad = ep.padBuffer[plane][thread_id];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 8) {
      auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(ptr + x));
      auto src32 = _mm_cvtepu8_epi32(src);
      _mm_store_ps(pad + x, _mm_cvtepi32_ps(src32));
      src = _mm_srli_si128(src, 4);
      src32 = _mm_cvtepu8_epi32(src);
      _mm_store_ps(pad + x + 4, _mm_cvtepi32_ps(src32));
      x += 8;
      if (x >= width) break;
      src = _mm_srli_si128(src, 4);
      src32 = _mm_cvtepu8_epi32(src);
      _mm_store_ps(pad + x, _mm_cvtepi32_ps(src32));
      src = _mm_srli_si128(src, 4);
      src32 = _mm_cvtepu8_epi32(src);
      _mm_store_ps(pad + x + 4, _mm_cvtepi32_ps(src32));
    }
    ptr += stride;
    pad += ep.padStrideElements[plane];
  }
}

template <>
void cast_to_pad_SSE41<uint16_t>(EngineParams& ep, unsigned int thread_id, int plane, const uint8_t* ptr, int depth, int stride, int width, int height)
{
  auto pad = ep.padBuffer[plane][thread_id];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 8) {
      auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(ptr + x * 2));
      auto src32 = _mm_cvtepu16_epi32(src);
      _mm_store_ps(pad + x, _mm_cvtepi32_ps(src32));
      src = _mm_srli_si128(src, 8);
      src32 = _mm_cvtepu16_epi32(src);
      _mm_store_ps(pad + x + 4, _mm_cvtepi32_ps(src32));
    }
    ptr += stride;
    pad += ep.padStrideElements[plane];
  }
}

template <>
void cast_from_pad_SSE41<uint8_t>(EngineParams& ep, unsigned int thread_id, int plane, uint8_t* ptr, int depth, int stride, int width, int height)
{
  auto half = _mm_set1_ps(0.5f);
  auto pad = ep.padBuffer[plane][thread_id];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 16) {
      // bit-identical floor(v+0.5f), which does not equal to SSE round()
      auto src1 = _mm_cvtps_epi32(_mm_floor_ps(_mm_add_ps(_mm_load_ps(pad + x), half)));
      auto src2 = _mm_cvtps_epi32(_mm_floor_ps(_mm_add_ps(_mm_load_ps(pad + x + 4), half)));
      auto src3 = _mm_cvtps_epi32(_mm_floor_ps(_mm_add_ps(_mm_load_ps(pad + x + 8), half)));
      auto src4 = _mm_cvtps_epi32(_mm_floor_ps(_mm_add_ps(_mm_load_ps(pad + x + 12), half)));
      auto dst1 = _mm_packus_epi32(src1, src2);
      auto dst2 = _mm_packus_epi32(src3, src4);
      auto dst = _mm_packus_epi16(dst1, dst2);
      if (x + 8 >= width)
        _mm_storel_epi64(reinterpret_cast<__m128i*>(ptr + x), dst);
      else
        _mm_store_si128(reinterpret_cast<__m128i*>(ptr + x), dst);
    }
    ptr += stride;
    pad += ep.padStrideElements[plane];
  }
}

template <>
void cast_from_pad_SSE41<uint16_t>(EngineParams& ep, unsigned int thread_id, int plane, uint8_t* ptr, int depth, int stride, int width, int height)
{
  auto half = _mm_set1_ps(0.5f);
  auto peak = _mm_set1_epi16(ep.peak);
  auto pad = ep.padBuffer[plane][thread_id];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 8) {
      auto src1 = _mm_cvtps_epi32(_mm_floor_ps(_mm_add_ps(_mm_load_ps(pad + x), half)));
      auto src2 = _mm_cvtps_epi32(_mm_floor_ps(_mm_add_ps(_mm_load_ps(pad + x + 4), half)));
      auto dst = _mm_packus_epi32(src1, src2);
      _mm_store_si128(reinterpret_cast<__m128i*>(ptr + x * 2), _mm_min_epu16(dst, peak));
    }
    ptr += stride;
    pad += ep.padStrideElements[plane];
  }
}
