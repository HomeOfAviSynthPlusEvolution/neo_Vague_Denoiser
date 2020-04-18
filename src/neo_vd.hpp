/*
 * Copyright 2020 Xinyue Lu
 *
 * DFTTest bridge.
 *
 */

#pragma once

#include <numeric>
#include <execution>
#include <mutex>
#include "common.h"
#include "core.h"
#include "version.hpp"

struct VagueDenoiser final : Filter {
  const float analysisLow[9] {
    0.037828455506995f, -0.023849465019380f, -0.110624404418423f, 0.377402855612654f,
    0.852698679009403f, 0.377402855612654f, -0.110624404418423f, -0.023849465019380f, 0.037828455506995f
  };
  const float analysisHigh[7] {
    -0.064538882628938f, 0.040689417609558f, 0.418092273222212f, -0.788485616405664f,
    0.418092273222212f, 0.040689417609558f, -0.064538882628938f
  };
  const float synthesisLow[7] {
    -0.064538882628938f, -0.040689417609558f, 0.418092273222212f, 0.788485616405664f,
    0.418092273222212f, -0.040689417609558f, -0.064538882628938f
  };
  const float synthesisHigh[9] {
    -0.037828455506995f, -0.023849465019380f, 0.110624404418423f, 0.377402855612654f,
    -0.852698679009403f, 0.377402855612654f, 0.110624404418423f, -0.023849465019380f, -0.037828455506995f
  };

  InDelegator* _in;
  DSVideoInfo out_vi;
  EngineParams ep;

  std::mutex thread_check_mutex;
  std::vector<bool> thread_id_store;

  const char* VSName() const override { return "VagueDenoiser"; }
  const char* AVSName() const override { return "neo_vd"; }
  const MtMode AVSMode() const override { return MT_NICE_FILTER; }
  const VSFilterMode VSMode() const override { return fmParallel; }
  const std::vector<Param> Params() const override {
    return std::vector<Param> {
      Param {"clip", Clip, false, true, true, false},
      Param {"threshold", Float},
      Param {"method", Integer},
      Param {"nsteps", Integer},
      Param {"percent", Float},
      Param {"planes", Integer, true, false, true},
      Param {"y", Integer, false, true, false},
      Param {"u", Integer, false, true, false},
      Param {"v", Integer, false, true, false},

      Param {"opt", Integer},
      Param {"threads", Integer}
    };
  }
  void Initialize(InDelegator* in, DSVideoInfo in_vi, FetchFrameFunctor* fetch_frame) override
  {
    // in_vi and fetch_frame are useless for source filter
    Filter::Initialize(in, in_vi, fetch_frame);

    in->Read("threshold", ep.threshold);
    in->Read("method", ep.method);
    in->Read("nsteps", ep.nsteps);
    in->Read("percent", ep.percent);

    // in->Read("opt", opt);
    // in->Read("threads", ep.threads);

    try {
      ep.process[0] =
      ep.process[1] =
      ep.process[2] =
      ep.process[3] = 2;
      std::vector<int> user_planes {0, 1, 2};
      in->Read("planes", user_planes);
      for (auto &&p : user_planes)
      {
        if (p < in_vi.Format.Planes)
          ep.process[p] = 3;
        else
          throw "plane index out of range";
      }
    }
    catch (const char *) {
      ep.process[0] =
      ep.process[1] =
      ep.process[2] = 3;
      in->Read("y", ep.process[0]);
      in->Read("u", ep.process[1]);
      in->Read("v", ep.process[2]);
    }

    int planeW = in_vi.Width;
    int planeH = in_vi.Height;

    if (in_vi.Format.IsFamilyYUV) {
      planeW >>= in_vi.Format.SSW;
      planeH >>= in_vi.Format.SSH;
    }

    if (in_vi.Width <= 0 || in_vi.Height <= 0)
      throw "only constant format input supported";

    if (planeW <= 4 || planeH <= 4)
      throw "the plane's width and height must be greater than 4";

    if ((in_vi.Format.IsInteger && in_vi.Format.BitsPerSample > 16) ||
        (in_vi.Format.IsFloat   && in_vi.Format.BitsPerSample != 32))
      throw "only 8-16 bit integer and 32 bit float input supported";

    if (ep.threshold <= 0.f) {
        throw "threshold must be greater than 0.0";
        return;
    }
    if (ep.method < 0 || ep.method > 2) {
        throw "method must be set to 0, 1 or 2";
        return;
    }
    if (ep.nsteps < 1) {
        throw "nsteps must be greater than or equal to 1";
        return;
    }
    if (ep.percent < 0.f || ep.percent > 100.f) {
        throw "percent must be between 0.0 and 100.0 (inclusive)";
        return;
    }

    ep.width[0] = in_vi.Width;
    ep.width[1] =
    ep.width[2] = planeW;

    ep.height[0] = in_vi.Height;
    ep.height[1] =
    ep.height[2] = planeH;

    if (in_vi.Format.IsFamilyYUV) {
      ep.lower[1] = ep.lower[2] = -0.5f;
      ep.upper[1] = ep.upper[2] = 0.5f;
    }

    uint32_t width_height = ((ep.process[2] == 3 || ep.process[3] == 3) ? std::min(planeW, planeH) : std::min(in_vi.Width, in_vi.Height)) - 1;
    // int max_steps = 32 - _lzcnt_u32(width_height) - 2;
    int max_steps = -2;
    while (width_height > 0) {
      max_steps++;
      width_height >>= 1;
    }

    if (ep.nsteps > max_steps)
      ep.nsteps = max_steps;

    // selectFunctions(ftype, opt, &ep);

    int width_height_y = std::max(in_vi.Width, in_vi.Height);
    int width_height_uv = std::max(planeW, planeH);

    if (in_vi.Format.IsInteger) {
        ep.threshold *= 1 << (in_vi.Format.BitsPerSample - 8);
        ep.peak = (1 << in_vi.Format.BitsPerSample) - 1;
        // Align to 64 bytes
        ep.padStrideElements[0] = ((in_vi.Width - 1) | 15) + 1;
        ep.padStrideElements[1] =
        ep.padStrideElements[2] = ((planeW - 1) | 15) + 1;
        ep.padSizeBytes[0] = ep.padStrideElements[0] * in_vi.Height * sizeof(float);
        ep.padSizeBytes[1] =
        ep.padSizeBytes[2] = ep.padStrideElements[1] * planeH * sizeof(float);

        ep.tmpSizeBytes[0] = (32 + width_height_y) * sizeof(float);
        ep.tmpSizeBytes[1] =
        ep.tmpSizeBytes[2] = (32 + width_height_uv) * sizeof(float);
    }
    else
      ep.threshold /= 255.f;
    
    ep.analysisLow = analysisLow;
    ep.analysisHigh = analysisHigh;
    ep.synthesisLow = synthesisLow;
    ep.synthesisHigh = synthesisHigh;
  }

  DSFrame GetFrame(int n, std::unordered_map<int, DSFrame> in_frames) override
  {
    unsigned int thread_id;

    {
      std::lock_guard<std::mutex> lock(thread_check_mutex);
      // Find empty slot
      auto it = std::find(thread_id_store.begin(), thread_id_store.end(), false);
      thread_id = static_cast<int>(std::distance(thread_id_store.begin(), it));
      if (it == thread_id_store.end())
        thread_id_store.push_back(false);
      thread_id_store[thread_id] = true;

      for (int p = 0; p < in_vi.Format.Planes; p++) {
        if (ep.process[p] == 3) {
          while (ep.padBuffer[p].size() <= thread_id)
            ep.padBuffer[p].push_back((float *)_aligned_malloc(ep.padSizeBytes[p], FRAME_ALIGN));
          while (ep.tmpBuffer[p][0].size() <= thread_id) {
            ep.tmpBuffer[p][0].push_back((float *)_aligned_malloc(ep.tmpSizeBytes[p], FRAME_ALIGN));
            ep.tmpBuffer[p][1].push_back((float *)_aligned_malloc(ep.tmpSizeBytes[p], FRAME_ALIGN));
            ep.tmpBuffer[p][2].push_back((float *)_aligned_malloc(ep.tmpSizeBytes[p], FRAME_ALIGN));
          }
          while (ep.hvBuffer[p][0].size() <= thread_id) {
            ep.hvBuffer[p][0].push_back((int *)_aligned_malloc(ep.nsteps, FRAME_ALIGN));
            ep.hvBuffer[p][1].push_back((int *)_aligned_malloc(ep.nsteps, FRAME_ALIGN));
            ep.hvBuffer[p][2].push_back((int *)_aligned_malloc(ep.nsteps, FRAME_ALIGN));
            ep.hvBuffer[p][3].push_back((int *)_aligned_malloc(ep.nsteps, FRAME_ALIGN));
          }
        }
      }
    }

    auto src0 = in_frames[n];
    auto dst = src0.Create(false);

    for (int p = 0; p < in_vi.Format.Planes; p++) {
      bool chroma = in_vi.Format.IsFamilyYUV && p > 0 && p < 3;
      auto height = in_vi.Height;
      auto width = in_vi.Width;
      auto src0_stride = src0.StrideBytes[p];
      auto src0_ptr = src0.SrcPointers[p];
      auto dst0_stride = dst.StrideBytes[p];
      auto dst0_ptr = dst.DstPointers[p];

      if (chroma) {
        height >>= in_vi.Format.SSH;
        width >>= in_vi.Format.SSW;
      }

      if (ep.process[p] == 3) {
        float * block;
        int block_stride_elements;

        if (in_vi.Format.IsInteger) {
          if (in_vi.Format.BitsPerSample == 8)
            cast_to_pad_c<uint8_t>(ep, thread_id, p, src0_ptr, in_vi.Format.BitsPerSample, src0_stride, width, height);
          else
            cast_to_pad_c<uint16_t>(ep, thread_id, p, src0_ptr, in_vi.Format.BitsPerSample, src0_stride, width, height);

          block = ep.padBuffer[p][thread_id];
          block_stride_elements = ep.padStrideElements[p];
        } else {
          framecpy(dst0_ptr, dst0_stride, src0_ptr, src0_stride, width * in_vi.Format.BytesPerSample, height);

          block = reinterpret_cast<float*>(dst0_ptr);
          block_stride_elements = dst0_stride / sizeof(float);
        }

        filter(ep, thread_id, p, block, block_stride_elements, width, height);

        if (in_vi.Format.IsInteger) {
          if (in_vi.Format.BitsPerSample == 8)
            cast_from_pad_c<uint8_t>(ep, thread_id, p, dst0_ptr, in_vi.Format.BitsPerSample, dst0_stride, width, height);
          else
            cast_from_pad_c<uint16_t>(ep, thread_id, p, dst0_ptr, in_vi.Format.BitsPerSample, dst0_stride, width, height);
        }
      }
      else if (ep.process[p] == 2) {
        framecpy(dst0_ptr, dst0_stride, src0_ptr, src0_stride, width * in_vi.Format.BytesPerSample, height);
      }
    }

    thread_id_store[thread_id] = false;
    return dst;
  }

  void framecpy(unsigned char * dst_ptr, int dst_stride, const unsigned char * src_ptr, int src_stride, int width_byte, int height) {
    if (src_stride == dst_stride) {
      memcpy(dst_ptr, src_ptr, dst_stride * height);
      return;
    }
    for (int h = 0; h < height; h++)
    {
      memcpy(dst_ptr, src_ptr, width_byte);
      dst_ptr += dst_stride;
      src_ptr += src_stride;
    }
  }

  ~VagueDenoiser() {
  }
};


namespace Plugin {
  const char* Identifier = "in.7086.neo_vd";
  const char* Namespace = "neo_vd";
  const char* Description = "Neo Vague Denoiser Filter " PLUGIN_VERSION;
}

std::vector<register_vsfilter_proc> RegisterVSFilters()
{
  return std::vector<register_vsfilter_proc> { VSInterface::RegisterFilter<VagueDenoiser> };
}

std::vector<register_avsfilter_proc> RegisterAVSFilters()
{
  return std::vector<register_avsfilter_proc> { AVSInterface::RegisterFilter<VagueDenoiser> };
}
