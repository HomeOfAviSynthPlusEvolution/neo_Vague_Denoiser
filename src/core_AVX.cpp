#include "core.h"
#include <immintrin.h>

#ifdef _MSC_VER
  #ifndef __clang__
    inline __m256 operator+(const __m256 &a, const __m256 &b) { return _mm256_add_ps(a, b); }
    inline __m256 operator-(const __m256 &a, const __m256 &b) { return _mm256_sub_ps(a, b); }
    inline __m256 operator*(const __m256 &a, const __m256 &b) { return _mm256_mul_ps(a, b); }
    inline __m256 operator/(const __m256 &a, const __m256 &b) { return _mm256_div_ps(a, b); }
    inline __m256 &operator+=(__m256 &a, const __m256 &b) { return a = _mm256_add_ps(a, b); }
    inline __m256 &operator-=(__m256 &a, const __m256 &b) { return a = _mm256_sub_ps(a, b); }
    inline __m256 &operator*=(__m256 &a, const __m256 &b) { return a = _mm256_mul_ps(a, b); }
    inline __m256 &operator/=(__m256 &a, const __m256 &b) { return a = _mm256_div_ps(a, b); }
  #endif
#endif

__m256 rcpnr_ps(const __m256 &a) {
  auto r = _mm256_rcp_ps(a);
  return r + r - r * a * r;
}

static inline void copy(const float * p1, float * p2, const int length) {
    memcpy(p2, p1, length * sizeof(float));
}

static inline void copy(const float * p1, const int stride1, float * p2, const int length) {
    for (int i = 0; i < length; i++) {
        p2[i] = *p1;
        p1 += stride1;
    }
}

static inline void copy(const float * p1, float * p2, const int stride2, const int length) {
    for (int i = 0; i < length; i++) {
        *p2 = p1[i];
        p2 += stride2;
    }
}

// Do symmetric extension of data using prescribed symmetries
// Original values are in output[npad] through output[npad+size-1]
// New values will be placed in output[0] through output[npad] and in output[npad+size] through output[2*npad+size-1] (note: end values may not be filled in)
// extension at left bdry is ... 3 2 1 0 | 0 1 2 3 ...
// same for right boundary
// if rightExt=1 then ... 3 2 1 0 | 1 2 3
static void symmetricExtension(float * output, const int size, const int leftExt, const int rightExt) {
    int first = NPADDING;
    int last = NPADDING - 1 + size;

    const int originalLast = last;

    if (leftExt == 2)
        output[--first] = output[NPADDING];
    if (rightExt == 2)
        output[++last] = output[originalLast];

    // extend left end
    int nextend = first;
    for (int i = 0; i < nextend; i++)
        output[--first] = output[NPADDING + 1 + i];

    const int idx = NPADDING + NPADDING - 1 + size;

    // extend right end
    nextend = idx - last;
    for (int i = 0; i < nextend; i++)
        output[++last] = output[originalLast - 1 - i];
}

static void transformStep(float * input, float * output, const int size, const int lowSize, EngineParams& ep) {
  symmetricExtension(input, size, 1, 1);

  int boundary = ((NPADDING + lowSize - 1) | 7) + 1;

  auto anaL0 = _mm256_set1_ps(ep.analysisLow[0]);
  auto anaL1 = _mm256_set1_ps(ep.analysisLow[1]);
  auto anaL2 = _mm256_set1_ps(ep.analysisLow[2]);
  auto anaL3 = _mm256_set1_ps(ep.analysisLow[3]);
  auto anaL4 = _mm256_set1_ps(ep.analysisLow[4]);

  for (int i = 8; i < boundary + 8; i += 8) {
    auto inpL0 = _mm256_loadu_ps(input + 2 * i - 14) + _mm256_loadu_ps(input + 2 * i - 6);
    auto inpL1 = _mm256_loadu_ps(input + 2 * i - 13) + _mm256_loadu_ps(input + 2 * i - 7);
    auto inpL2 = _mm256_loadu_ps(input + 2 * i - 12) + _mm256_load_ps(input + 2 * i - 8);
    auto inpL3 = _mm256_loadu_ps(input + 2 * i - 11) + _mm256_loadu_ps(input + 2 * i - 9);
    auto inpL4 = _mm256_loadu_ps(input + 2 * i - 10);
    auto resultL = (inpL0 * anaL0 + inpL1 * anaL1) + (inpL2 * anaL2 + inpL3 * anaL3) + inpL4 * anaL4;
    auto inpH0 = _mm256_loadu_ps(input + 2 * i - 6) + _mm256_loadu_ps(input + 2 * i + 2);
    auto inpH1 = _mm256_loadu_ps(input + 2 * i - 5) + _mm256_loadu_ps(input + 2 * i + 1);
    auto inpH2 = _mm256_loadu_ps(input + 2 * i - 4) + _mm256_load_ps(input + 2 * i - 0);
    auto inpH3 = _mm256_loadu_ps(input + 2 * i - 3) + _mm256_loadu_ps(input + 2 * i - 1);
    auto inpH4 = _mm256_loadu_ps(input + 2 * i - 2);
    auto resultH = (inpH0 * anaL0 + inpH1 * anaL1) + (inpH2 * anaL2 + inpH3 * anaL3) + inpH4 * anaL4;
    auto result_lo = _mm256_permute2f128_ps(resultL, resultH, 32);
    auto result_hi = _mm256_permute2f128_ps(resultL, resultH, 49);
    auto result = _mm256_blend_ps(_mm256_permute_ps(result_lo, 0x8), _mm256_permute_ps(result_hi, 0x80), 204);
    _mm256_store_ps(output + i, result);
  }

  auto anaH0 = _mm256_set1_ps(ep.analysisHigh[0]);
  auto anaH1 = _mm256_set1_ps(ep.analysisHigh[1]);
  auto anaH2 = _mm256_set1_ps(ep.analysisHigh[2]);
  auto anaH3 = _mm256_set1_ps(ep.analysisHigh[3]);

  for (int i = NPADDING; i < NPADDING + lowSize; i += 8) {
    auto inpL0 = _mm256_loadu_ps(input + 2 * i - 12) + _mm256_loadu_ps(input + 2 * i - 6);
    auto inpL1 = _mm256_loadu_ps(input + 2 * i - 11) + _mm256_loadu_ps(input + 2 * i - 7);
    auto inpL2 = _mm256_loadu_ps(input + 2 * i - 10) + _mm256_loadu_ps(input + 2 * i - 8);
    auto inpL3 = _mm256_loadu_ps(input + 2 * i - 9);
    auto resultL = inpL0 * anaH0 + inpL1 * anaH1 + inpL2 * anaH2 + inpL3 * anaH3;
    auto inpH0 = _mm256_loadu_ps(input + 2 * i - 4) + _mm256_loadu_ps(input + 2 * i + 2);
    auto inpH1 = _mm256_loadu_ps(input + 2 * i - 3) + _mm256_loadu_ps(input + 2 * i + 1);
    auto inpH2 = _mm256_loadu_ps(input + 2 * i - 2) + _mm256_loadu_ps(input + 2 * i - 0);
    auto inpH3 = _mm256_loadu_ps(input + 2 * i - 1);
    auto resultH = inpH0 * anaH0 + inpH1 * anaH1 + inpH2 * anaH2 + inpH3 * anaH3;
    auto result_lo = _mm256_permute2f128_ps(resultL, resultH, 32);
    auto result_hi = _mm256_permute2f128_ps(resultL, resultH, 49);
    auto result = _mm256_blend_ps(_mm256_permute_ps(result_lo, 0x8), _mm256_permute_ps(result_hi, 0x80), 204);
    _mm256_storeu_ps(output + i + lowSize, result);
  }
}

static void invertStep(const float * input, float * output, float * temp, const int size, EngineParams& ep) {
  const int lowSize = (size + 1) >> 1;
  const int highSize = size >> 1;

  memcpy(temp + NPADDING, input + NPADDING, lowSize * sizeof(float));

  int leftExt = 1;
  int rightExt = (size % 2 == 0) ? 2 : 1;
  symmetricExtension(temp, lowSize, leftExt, rightExt);

  memset(output, 0, (NPADDING + NPADDING + size) * sizeof(float));
  const int findex = (size + 2) >> 1;

  auto synL0 = _mm256_set1_ps(ep.synthesisLow[0]);
  auto synL1 = _mm256_set1_ps(ep.synthesisLow[1]);
  auto synL2 = _mm256_set1_ps(ep.synthesisLow[2]);
  auto synL3 = _mm256_set1_ps(ep.synthesisLow[3]);

  for (int i = 8; i < findex + 8; i += 8) {
    auto tmp0 = _mm256_load_ps(temp + i + 0);
    auto tmp1 = _mm256_loadu_ps(temp + i + 1);
    auto tmp2 = _mm256_loadu_ps(temp + i + 2);
    auto tmp3 = _mm256_loadu_ps(temp + i + 3);
    // o[8] = t8 * sl1 + t9 * sl3 + t10 * sl1
    auto r0 = (tmp0 + tmp2) * synL1 + tmp1 * synL3;
    // o[9] = t8 * sl0 + t9 * sl2 + t10 * sl2 + t11 * sl0
    auto r1 = (tmp0 + tmp3) * synL0 + (tmp1 + tmp2) * synL2;
    auto interleave_lo = _mm256_unpacklo_ps(r0, r1);
    auto interleave_hi = _mm256_unpackhi_ps(r0, r1);
    auto result_lo = _mm256_permute2f128_ps(interleave_lo, interleave_hi, 32);
    auto result_hi = _mm256_permute2f128_ps(interleave_lo, interleave_hi, 49);
    auto out_lo = _mm256_load_ps(output + 2 * i - 8);
    auto out_hi = _mm256_load_ps(output + 2 * i);
    out_lo += result_lo;
    out_hi += result_hi;
    _mm256_store_ps(output + 2 * i - 8, out_lo);
    _mm256_store_ps(output + 2 * i, out_hi);
  }

  memcpy(temp + NPADDING, input + NPADDING + lowSize, highSize * sizeof(float));

  leftExt = 2;
  rightExt = (size % 2 == 0) ? 1 : 2;
  symmetricExtension(temp, highSize, leftExt, rightExt);

  auto synH0 = _mm256_set1_ps(ep.synthesisHigh[0]);
  auto synH1 = _mm256_set1_ps(ep.synthesisHigh[1]);
  auto synH2 = _mm256_set1_ps(ep.synthesisHigh[2]);
  auto synH3 = _mm256_set1_ps(ep.synthesisHigh[3]);
  auto synH4 = _mm256_set1_ps(ep.synthesisHigh[4]);

  for (int i = 8; i < findex + 8; i += 8) {
    auto tmp0 = _mm256_loadu_ps(temp + i - 1);
    auto tmp1 = _mm256_load_ps(temp + i + 0);
    auto tmp2 = _mm256_loadu_ps(temp + i + 1);
    auto tmp3 = _mm256_loadu_ps(temp + i + 2);
    auto tmp4 = _mm256_loadu_ps(temp + i + 3);
    // o[8] = t7 * sh1 + t8 * sh3 + t9 * sh3 + t10 * sh1
    auto r0 = (tmp0 + tmp3) * synH1 + (tmp1 + tmp2) * synH3;
    // o[9] = t8 * sh2 + t9 * sh4 + t10 * sh2 + t11 * sh0
    auto r1 = (tmp0 + tmp4) * synH0 + (tmp1 + tmp3) * synH2 + tmp2 * synH4;
    auto interleave_lo = _mm256_unpacklo_ps(r0, r1);
    auto interleave_hi = _mm256_unpackhi_ps(r0, r1);
    auto result_lo = _mm256_permute2f128_ps(interleave_lo, interleave_hi, 32);
    auto result_hi = _mm256_permute2f128_ps(interleave_lo, interleave_hi, 49);
    auto out_lo = _mm256_load_ps(output + 2 * i - 8);
    auto out_hi = _mm256_load_ps(output + 2 * i);
    out_lo += result_lo;
    out_hi += result_hi;
    _mm256_store_ps(output + 2 * i - 8, out_lo);
    _mm256_store_ps(output + 2 * i, out_hi);
  }
}

static void hardThresholding(float * block, const int width, const int height, const int stride, const float threshold, const float percent) {
  const float frac = 1.f - percent * 0.01f;
  auto m_frac = _mm256_set1_ps(frac);
  auto m_threshold = _mm256_set1_ps(threshold);
  int32_t positive = 0x7FFFFFFF;
  auto m_positive = _mm256_set1_ps(reinterpret_cast<float&>(positive));

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 8) {
      auto v = _mm256_load_ps(block + x);
      auto vpos = _mm256_and_ps(v, m_positive);
      auto mask = _mm256_cmp_ps(vpos, m_threshold, _CMP_LE_OS);
      v = _mm256_mul_ps(v, m_frac);
      _mm256_maskstore_ps(block + x, _mm256_castps_si256(mask), v);
    }
    block += stride;
  }
}

static void softThresholding(float * block, const int width, const int height, const int stride, const float threshold, const float percent, const int nsteps) {
    const float frac = 1.f - percent * 0.01f;
    const float shift = threshold * 0.01f * percent;

    int w = width;
    int h = height;
    for (int l = 0; l < nsteps; l++) {
        w = (w + 1) >> 1;
        h = (h + 1) >> 1;
    }

    for (int y = 0; y < height; y++) {
        const int x0 = (y < h) ? w : 0;
        for (int x = x0; x < width; x++) {
            const float temp = std::abs(block[x]);
            if (temp <= threshold)
                block[x] *= frac;
            else
                block[x] = (block[x] < 0.f ? -1.f : (block[x] > 0.f ? 1.f : 0.f)) * (temp - shift);
        }
        block += stride;
    }
}

static void qianThresholding(float * block, const int width, const int height, const int stride, const float threshold, const float percent) {
  const float percent01 = percent * 0.01f;
  const float tr2 = threshold * threshold * percent01;
  const float frac = 1.f - percent01;
  auto m_tr2 = _mm256_set1_ps(tr2);
  auto m_frac = _mm256_set1_ps(frac);
  auto m_threshold = _mm256_set1_ps(threshold);
  int32_t positive = 0x7FFFFFFF;
  auto m_positive = _mm256_set1_ps(reinterpret_cast<float&>(positive));

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 8) {
      auto v = _mm256_load_ps(block + x);
      auto vpos = _mm256_and_ps(v, m_positive);
      auto vsq = v * v;
      auto vsqrcp = rcpnr_ps(vsq);
      auto mask = _mm256_cmp_ps(vpos, m_threshold, _CMP_LE_OS);
      auto m_frac1 = (vsq - m_tr2) * vsqrcp;
      auto coef = _mm256_blendv_ps(m_frac1, m_frac, mask);
      _mm256_store_ps(block + x, v * coef);
    }
    block += stride;
  }
}

void filter_AVX(EngineParams& ep, unsigned int thread_id, int plane, float * src, int src_stride_elements, int width, int height)
{
  float* tempIn = ep.tmpBuffer[plane][0][thread_id];
  float* tempOut = ep.tmpBuffer[plane][1][thread_id];
  float* temp2 = ep.tmpBuffer[plane][2][thread_id];

  int* hLowSize = ep.hvBuffer[plane][0][thread_id];
  int* hHighSize = ep.hvBuffer[plane][1][thread_id];
  int* vLowSize = ep.hvBuffer[plane][2][thread_id];
  int* vHighSize = ep.hvBuffer[plane][3][thread_id];

  int hLowSize0 = width;
  int vLowSize0 = height;
  int nstepsTransform = ep.nsteps;
  while (nstepsTransform--) {
    int lowSize = (hLowSize0 + 1) >> 1;
    float * input = src;
    for (int j = 0; j < vLowSize0; j++) {
      copy(input, tempIn + NPADDING, hLowSize0);
      transformStep(tempIn, tempOut, hLowSize0, lowSize, ep);
      copy(tempOut + NPADDING, input, hLowSize0);
      input += src_stride_elements;
    }

    lowSize = (vLowSize0 + 1) >> 1;
    input = src;
    for (int j = 0; j < hLowSize0; j++) {
      copy(input, src_stride_elements, tempIn + NPADDING, vLowSize0);
      transformStep(tempIn, tempOut, vLowSize0, lowSize, ep);
      copy(tempOut + NPADDING, input, src_stride_elements, vLowSize0);
      input++;
    }

    hLowSize0 = (hLowSize0 + 1) >> 1;
    vLowSize0 = (vLowSize0 + 1) >> 1;
  }

  if (ep.method == 0)
    hardThresholding(src, width, height, src_stride_elements, ep.threshold, ep.percent);
  else if (ep.method == 1)
    softThresholding(src, width, height, src_stride_elements, ep.threshold, ep.percent, ep.nsteps);
  else
    qianThresholding(src, width, height, src_stride_elements, ep.threshold, ep.percent);

  hLowSize[0] = (width + 1) >> 1;
  hHighSize[0] = width >> 1;
  vLowSize[0] = (height + 1) >> 1;
  vHighSize[0] = height >> 1;
  for (int i = 1; i < ep.nsteps; i++) {
    hLowSize[i] = (hLowSize[i - 1] + 1) >> 1;
    hHighSize[i] = hLowSize[i - 1] >> 1;
    vLowSize[i] = (vLowSize[i - 1] + 1) >> 1;
    vHighSize[i] = vLowSize[i - 1] >> 1;
  }
  int nstepsInvert = ep.nsteps;
  while (nstepsInvert--) {
    const int idx = vLowSize[nstepsInvert] + vHighSize[nstepsInvert];
    const int idx2 = hLowSize[nstepsInvert] + hHighSize[nstepsInvert];
    float * idx3 = src;
    for (int i = 0; i < idx2; i++) {
      copy(idx3, src_stride_elements, tempIn + NPADDING, idx);
      invertStep(tempIn, tempOut, temp2, idx, ep);
      copy(tempOut + NPADDING, idx3, src_stride_elements, idx);
      idx3++;
    }

    idx3 = src;
    for (int i = 0; i < idx; i++) {
      copy(idx3, tempIn + NPADDING, idx2);
      invertStep(tempIn, tempOut, temp2, idx2, ep);
      copy(tempOut + NPADDING, idx3, idx2);
      idx3 += src_stride_elements;
    }
  }
}
