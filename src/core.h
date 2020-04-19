#pragma once

#include "common.h"
static constexpr int NPADDING = 10;

template <typename T>
void cast_to_pad_C(EngineParams& ep, unsigned int thread_id, int plane, const uint8_t* ptr, int depth, int stride, int width, int height);
template <typename T>
void cast_from_pad_C(EngineParams& ep, unsigned int thread_id, int plane, uint8_t* ptr, int depth, int stride, int width, int height);
template <typename T>
void cast_to_pad_SSE41(EngineParams& ep, unsigned int thread_id, int plane, const uint8_t* ptr, int depth, int stride, int width, int height);
template <typename T>
void cast_from_pad_SSE41(EngineParams& ep, unsigned int thread_id, int plane, uint8_t* ptr, int depth, int stride, int width, int height);
template <typename T>
void cast_to_pad_AVX2(EngineParams& ep, unsigned int thread_id, int plane, const uint8_t* ptr, int depth, int stride, int width, int height);
template <typename T>
void cast_from_pad_AVX2(EngineParams& ep, unsigned int thread_id, int plane, uint8_t* ptr, int depth, int stride, int width, int height);

void filter_C(EngineParams& ep, unsigned int thread_id, int plane, float * src, int src_stride_elements, int width, int height);
void filter_SSE(EngineParams& ep, unsigned int thread_id, int plane, float * src, int src_stride_elements, int width, int height);
void filter_AVX(EngineParams& ep, unsigned int thread_id, int plane, float * src, int src_stride_elements, int width, int height);
