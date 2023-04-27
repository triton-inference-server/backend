// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <cupti.h>

static_assert(
    sizeof(uint64_t) >= sizeof(uintptr_t),
    "The implementation is storing address pointer as uint64_t, "
    "must ensure the space for pointer is <= sizeof(uint64_t).");

namespace triton { namespace backend {

/// DeviceMemoryTracker is a backend utility provided to track the memory
/// allocated for a particular model and associated model instance. This utility
/// is often used for backend that set
/// TRITONBACKEND_BackendAttributeSetEnableMemoryTracker to true, which provides
/// additional information to Triton for making decision on model scaling and
/// deployment.
///
/// Caveat: The memory tracker is implemented with CUPTI library which currently
/// only supports single client/subscriber. This is an known limitation and as a
/// result, the memory tracker can cause unexpected application failure if other
/// component of the Triton process also uses CUPTI with a different
/// configuration, for example, the framework used by the backend may have
/// implemented similar profiler with CUPTI. Therefore, before enabling this
/// memory tracker utilities, you should make sure that there is no other CUPTI
/// client in the process. This tracker is implemented with the assumption that
/// all other possible CUPTI clients are using the same implementation so that
/// as long as all backends are compiled with this memory tracker and they may
/// interact with CUPTI initialized externally to the backend without issues.
///
/// Typical usage:
///
/// The backend must implement TRITONBACKEND_GetBackendAttribute which will
/// call TRITONBACKEND_BackendAttributeSetEnableMemoryTracker with proper
/// arguments, and must implement TRITONBACKEND_ModelMemoryUsage.
///
/// On TRITONBACKEND_Initialize
///  - Call DeviceMemoryTracker::InitTrace
///
/// On TRITONBACKEND_GetBackendAttribute
///  - Call TRITONBACKEND_BackendAttributeSetEnableMemoryTracker with value
///    returned from DeviceMemoryTracker::InitTrace
///
/// If true, call DeviceMemoryTracker::TrackThreadMemoryUsage and
/// DeviceMemoryTracker::UntrackThreadMemoryUsage accordingly to track memory
/// allocation in the scope between the function, the memory usage should be
/// stored in ScopedMemoryUsage object that has the same lifetime as
/// TRITONBACKEND_Model object.
///
/// On TRITONBACKEND_ModelMemoryUsage
///  - Call ScopedMemoryUsage::SerializeToBufferAttributes to set the value
///    to be returned.
class DeviceMemoryTracker {
 public:
  // [WIP] convert to C struct to avoid ABI compatibility
  struct ScopedMemoryUsage {
    ~ScopedMemoryUsage()
    {
      if (tracked_) {
        UntrackThreadMemoryUsage(this);
      }
    }

    // [WIP] convert to TritonBuff
    // std::vector<BufferAttributes> SerializeToBufferAttributes()
    // {
    //   std::vector<BufferAttributes> res;
    //   for (const auto& usage : system_byte_size_) {
    //     res.emplace_back(
    //         usage.second, TRITONSERVER_MEMORY_CPU, usage.first, nullptr);
    //   }
    //   for (const auto& usage : pinned_byte_size_) {
    //     res.emplace_back(
    //         usage.second, TRITONSERVER_MEMORY_CPU_PINNED, usage.first, nullptr);
    //   }
    //   for (const auto& usage : cuda_byte_size_) {
    //     res.emplace_back(
    //         usage.second, TRITONSERVER_MEMORY_GPU, usage.first, nullptr);
    //   }
    //   return res;
    // }

    std::map<int64_t, size_t> system_byte_size_;
    std::map<int64_t, size_t> pinned_byte_size_;
    std::map<int64_t, size_t> cuda_byte_size_;
    // Byte size of allocated memory tracked,
    // 'system_byte_size_' is likely to be empty as system memory allocation
    // is not controlled by CUDA driver. But keeping it for completeness.
    bool tracked_{false};
  };
  static bool InitTrace();

  // The memory usage will be tracked and modified until it's untracked, 'usage'
  // must be valid and not to be modified externally until untrack is called.
  // Currently can distinguish activity by correlation id which is thread
  // specific, which implies that there will be mssing records if tracking
  // region switching threads to handle other activities.
  static void TrackThreadMemoryUsage(ScopedMemoryUsage* usage);
  static void UntrackThreadMemoryUsage(ScopedMemoryUsage* usage);

  static void TrackActivity(CUpti_Activity* record)
  {
    tracker_->TrackActivityInternal(record);
  }

  ~DeviceMemoryTracker() {
    if (subscriber_) {
      cuptiUnsubscribe(subscriber_);
    }
  }

 private:
  void TrackActivityInternal(CUpti_Activity* record);

  static std::unique_ptr<DeviceMemoryTracker> tracker_;
  std::mutex mtx_;
  std::unordered_map<uint32_t, uintptr_t> activity_to_memory_usage_;
  CUpti_SubscriberHandle subscriber_;
};

}}  // namespace triton::backend
