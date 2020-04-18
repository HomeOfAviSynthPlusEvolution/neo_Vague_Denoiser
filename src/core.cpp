#include "core.h"

static constexpr int NPAD = 10;

template <typename T>
void cast_to_pad_c(EngineParams& ep, unsigned int thread_id, int plane, const uint8_t* ptr, int depth, int stride, int width, int height);

template <typename T>
void cast_from_pad_c(EngineParams& ep, unsigned int thread_id, int plane, uint8_t* ptr, int depth, int stride, int width, int height);

template <>
void cast_to_pad_c<uint8_t>(EngineParams& ep, unsigned int thread_id, int plane, const uint8_t* ptr, int depth, int stride, int width, int height)
{
  auto pad = ep.padBuffer[plane][thread_id];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++)
      pad[x] = ptr[x];
    ptr += stride;
    pad += ep.padStrideElements[plane];
  }
}

template <>
void cast_to_pad_c<uint16_t>(EngineParams& ep, unsigned int thread_id, int plane, const uint8_t* ptr0, int depth, int stride, int width, int height)
{
  const uint16_t* ptr = reinterpret_cast<const uint16_t*>(ptr0);
  stride >>= 1;
  auto pad = ep.padBuffer[plane][thread_id];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++)
      pad[x] = ptr[x];
    ptr += stride;
    pad += ep.padStrideElements[plane];
  }
}

template <>
void cast_from_pad_c<uint8_t>(EngineParams& ep, unsigned int thread_id, int plane, uint8_t* ptr, int depth, int stride, int width, int height)
{
  auto pad = ep.padBuffer[plane][thread_id];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++)
      ptr[x] = std::min(std::max(static_cast<int>(pad[x] + 0.5f), 0), ep.peak);
    ptr += stride;
    pad += ep.padStrideElements[plane];
  }
}

template <>
void cast_from_pad_c<uint16_t>(EngineParams& ep, unsigned int thread_id, int plane, uint8_t* ptr0, int depth, int stride, int width, int height)
{
  uint16_t* ptr = reinterpret_cast<uint16_t*>(ptr0);
  stride >>= 1;
  auto pad = ep.padBuffer[plane][thread_id];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++)
      ptr[x] = std::min(std::max(static_cast<int>(pad[x] + 0.5f), 0), ep.peak);
    ptr += stride;
    pad += ep.padStrideElements[plane];
  }
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
    int first = NPAD;
    int last = NPAD - 1 + size;

    const int originalLast = last;

    if (leftExt == 2)
        output[--first] = output[NPAD];
    if (rightExt == 2)
        output[++last] = output[originalLast];

    // extend left end
    int nextend = first;
    for (int i = 0; i < nextend; i++)
        output[--first] = output[NPAD + 1 + i];

    const int idx = NPAD + NPAD - 1 + size;

    // extend right end
    nextend = idx - last;
    for (int i = 0; i < nextend; i++)
        output[++last] = output[originalLast - 1 - i];
}

static void transformStep(float * input, float * output, const int size, const int lowSize, EngineParams& ep) {
    symmetricExtension(input, size, 1, 1);

    for (int i = NPAD; i < NPAD + lowSize; i++) {
        const float a = input[2 * i - 14] * ep.analysisLow[0];
        const float b = input[2 * i - 13] * ep.analysisLow[1];
        const float c = input[2 * i - 12] * ep.analysisLow[2];
        const float d = input[2 * i - 11] * ep.analysisLow[3];
        const float e = input[2 * i - 10] * ep.analysisLow[4];
        const float f = input[2 * i - 9] * ep.analysisLow[3];
        const float g = input[2 * i - 8] * ep.analysisLow[2];
        const float h = input[2 * i - 7] * ep.analysisLow[1];
        const float k = input[2 * i - 6] * ep.analysisLow[0];
        output[i] = a + b + c + d + e + f + g + h + k;
    }
    for (int i = NPAD; i < NPAD + lowSize; i++) {
        const float a = input[2 * i - 12] * ep.analysisHigh[0];
        const float b = input[2 * i - 11] * ep.analysisHigh[1];
        const float c = input[2 * i - 10] * ep.analysisHigh[2];
        const float d = input[2 * i - 9] * ep.analysisHigh[3];
        const float e = input[2 * i - 8] * ep.analysisHigh[2];
        const float f = input[2 * i - 7] * ep.analysisHigh[1];
        const float g = input[2 * i - 6] * ep.analysisHigh[0];
        output[i + lowSize] = a + b + c + d + e + f + g;
    }
}

static void invertStep(const float * input, float * output, float * temp, const int size, EngineParams& ep) {
    const int lowSize = (size + 1) >> 1;
    const int highSize = size >> 1;

    memcpy(temp + NPAD, input + NPAD, lowSize * sizeof(float));

    int leftExt = 1;
    int rightExt = (size % 2 == 0) ? 2 : 1;
    symmetricExtension(temp, lowSize, leftExt, rightExt);

    memset(output, 0, (NPAD + NPAD + size) * sizeof(float));
    const int findex = (size + 2) >> 1;

    for (int i = 9; i < findex + 11; i++) {
        const float a = temp[i] * ep.synthesisLow[0];
        const float b = temp[i] * ep.synthesisLow[1];
        const float c = temp[i] * ep.synthesisLow[2];
        const float d = temp[i] * ep.synthesisLow[3];
        output[2 * i - 13] += a;
        output[2 * i - 12] += b;
        output[2 * i - 11] += c;
        output[2 * i - 10] += d;
        output[2 * i - 9] += c;
        output[2 * i - 8] += b;
        output[2 * i - 7] += a;
    }

    memcpy(temp + NPAD, input + NPAD + lowSize, highSize * sizeof(float));

    leftExt = 2;
    rightExt = (size % 2 == 0) ? 1 : 2;
    symmetricExtension(temp, highSize, leftExt, rightExt);

    for (int i = 8; i < findex + 11; i++) {
        const float a = temp[i] * ep.synthesisHigh[0];
        const float b = temp[i] * ep.synthesisHigh[1];
        const float c = temp[i] * ep.synthesisHigh[2];
        const float d = temp[i] * ep.synthesisHigh[3];
        const float e = temp[i] * ep.synthesisHigh[4];
        output[2 * i - 13] += a;
        output[2 * i - 12] += b;
        output[2 * i - 11] += c;
        output[2 * i - 10] += d;
        output[2 * i - 9] += e;
        output[2 * i - 8] += d;
        output[2 * i - 7] += c;
        output[2 * i - 6] += b;
        output[2 * i - 5] += a;
    }
}

static void hardThresholding(float * block, const int width, const int height, const int stride, const float threshold, const float percent) {
    const float frac = 1.f - percent * 0.01f;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (std::abs(block[x]) <= threshold)
                block[x] *= frac;
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

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const float temp = std::abs(block[x]);
            if (temp <= threshold) {
                block[x] *= frac;
            } else {
                const float tp2 = temp * temp;
                block[x] *= (tp2 - tr2) / tp2;
            }
        }
        block += stride;
    }
}

void filter(EngineParams& ep, unsigned int thread_id, int plane, float * src, int src_stride_elements, int width, int height)
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
      copy(input, tempIn + NPAD, hLowSize0);
      transformStep(tempIn, tempOut, hLowSize0, lowSize, ep);
      copy(tempOut + NPAD, input, hLowSize0);
      input += src_stride_elements;
    }

    lowSize = (vLowSize0 + 1) >> 1;
    input = src;
    for (int j = 0; j < hLowSize0; j++) {
      copy(input, src_stride_elements, tempIn + NPAD, vLowSize0);
      transformStep(tempIn, tempOut, vLowSize0, lowSize, ep);
      copy(tempOut + NPAD, input, src_stride_elements, vLowSize0);
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
      copy(idx3, src_stride_elements, tempIn + NPAD, idx);
      invertStep(tempIn, tempOut, temp2, idx, ep);
      copy(tempOut + NPAD, idx3, src_stride_elements, idx);
      idx3++;
    }

    idx3 = src;
    for (int i = 0; i < idx; i++) {
      copy(idx3, tempIn + NPAD, idx2);
      invertStep(tempIn, tempOut, temp2, idx2, ep);
      copy(tempOut + NPAD, idx3, idx2);
      idx3 += src_stride_elements;
    }
  }
}
