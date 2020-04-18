#pragma once

#include "common.h"

template <typename T>
void cast_to_pad_c(EngineParams& ep, unsigned int thread_id, int plane, const uint8_t* ptr, int depth, int stride, int width, int height);

template <typename T>
void cast_from_pad_c(EngineParams& ep, unsigned int thread_id, int plane, uint8_t* ptr, int depth, int stride, int width, int height);

void filter(EngineParams& ep, unsigned int thread_id, int plane, float * src, int src_stride_elements, int width, int height);
