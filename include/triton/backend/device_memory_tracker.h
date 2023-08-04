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

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "triton/backend/backend_common.h"
#include "triton/common/logging.h"
#include "triton/core/tritonserver.h"

#if defined(TRITON_ENABLE_GPU) && defined(TRITON_ENABLE_MEMORY_TRACKER)
#include <cupti.h>
#endif

static_assert(
    sizeof(uint64_t) >= sizeof(uintptr_t),
    "The implementation is storing address pointer as uint64_t, "
    "must ensure the space for pointer is <= sizeof(uint64_t).");

namespace triton { namespace backend {

/// DeviceMemoryTracker is a backend utility provided to track the memory
/// allocated for a particular model and associated model instances.
/// This utility is often used for backend to report memory usage through
/// TRITONBACKEND_ModelReportMemoryUsage and
/// TRITONBACKEND_ModelInstanceReportMemoryUsage, which provides
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
/// all other CUPTI clients are using the same implementation so that
/// as long as all backends are compiled with this memory tracker, they may
/// interact with an externally-initialized CUPTI to the backend without issues.
///
/// Typical usage:
///
/// On TRITONBACKEND_Initialize
///  - Call DeviceMemoryTracker::Init
///
/// If DeviceMemoryTracker::Init returns true,
/// DeviceMemoryTracker::TrackThreadMemoryUsage and
/// DeviceMemoryTracker::UntrackThreadMemoryUsage can be called accordingly to
/// track memory allocation in the scope between the two calls. The memory usage
/// will be recorded in MemoryUsage object and may be reported through
/// TRITONBACKEND_ModelReportMemoryUsage or
/// TRITONBACKEND_ModelInstanceReportMemoryUsage based on the entity of the
/// memory usage.
///
/// On reporting memory usage
///  - Call MemoryUsage::SerializeToBufferAttributes to prepare the usage
///    in the desired format. The BufferAttributes will be owned by MemoryUsage.

extern "C" {

typedef struct TRITONBACKEND_CuptiTracker_t {
  // C struct require extra implementation for dynamic array, for simplicity,
  // the following assumptions are made to pre-allocate the array with max
  // possible length:
  //  - system / pinned memory allocation should only be on deviceId 0
  //  - CUDA allocation will only be on visible CUDA devices
  int64_t* system_memory_usage_byte_;
  int64_t* pinned_memory_usage_byte_;
  int64_t* cuda_memory_usage_byte_;
  uint32_t system_array_len_;
  uint32_t pinned_array_len_;
  uint32_t cuda_array_len_;

  // only set to false if somehow the CUPTI activity occurs on index out of
  // range. In that case, user should invalidate the whole tracker.
  bool valid_;
} TRITONBACKEND_CuptiTracker;
}

class DeviceMemoryTracker {
 public:
  struct MemoryUsage {
    MemoryUsage()
    {
      cuda_memory_usage_byte_.resize(CudaDeviceCount(), 0);

      cupti_tracker_.system_memory_usage_byte_ =
          system_memory_usage_byte_.data();
      cupti_tracker_.pinned_memory_usage_byte_ =
          pinned_memory_usage_byte_.data();
      cupti_tracker_.cuda_memory_usage_byte_ = cuda_memory_usage_byte_.data();
      cupti_tracker_.system_array_len_ = system_memory_usage_byte_.size();
      cupti_tracker_.pinned_array_len_ = pinned_memory_usage_byte_.size();
      cupti_tracker_.cuda_array_len_ = cuda_memory_usage_byte_.size();
      cupti_tracker_.valid_ = true;
    }

    ~MemoryUsage()
    {
      // Make sure all C struct reference are dropped before clearing.
      if (tracked_) {
        UntrackThreadMemoryUsage(this);
      }
      for (auto& ba : buffer_attributes_) {
        if (ba) {
          LOG_IF_ERROR(
              TRITONSERVER_BufferAttributesDelete(ba),
              "Releasing buffer attributes in MemoryUsage object");
        }
      }
    }

    // Disable copy and assign to better manage C struct lifecycle
    MemoryUsage(const MemoryUsage&) = delete;
    void operator=(const MemoryUsage&) = delete;

    // merge record from another MemoryUsage object
    MemoryUsage& operator+=(const MemoryUsage& rhs)
    {
      std::transform(
          rhs.system_memory_usage_byte_.begin(),
          rhs.system_memory_usage_byte_.end(),
          system_memory_usage_byte_.begin(), system_memory_usage_byte_.begin(),
          std::plus<int64_t>());
      std::transform(
          rhs.pinned_memory_usage_byte_.begin(),
          rhs.pinned_memory_usage_byte_.end(),
          pinned_memory_usage_byte_.begin(), pinned_memory_usage_byte_.begin(),
          std::plus<int64_t>());
      std::transform(
          rhs.cuda_memory_usage_byte_.begin(),
          rhs.cuda_memory_usage_byte_.end(), cuda_memory_usage_byte_.begin(),
          cuda_memory_usage_byte_.begin(), std::plus<int64_t>());
      return *this;
    }

    // Serialize the MemoryUsage into an array of TRITONSERVER_BufferAttributes,
    // the buffer attributes object are owned by the MemoryUsage object.
    // Empty usage will be returned if the MemoryUsage object is invalid.
    TRITONSERVER_Error* SerializeToBufferAttributes(
        TRITONSERVER_BufferAttributes*** usage, uint32_t* usage_size)
    {
      if (!cupti_tracker_.valid_) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "MemoryUsage record is invalid.");
      }
      uint32_t usage_idx = 0;

      // Define lambda to convert an vector of memory usage of the same type of
      // device into buffer attributes and set in 'usage'
      auto set_attributes_for_device_fn =
          [&](const std::vector<int64_t>& devices,
              const TRITONSERVER_MemoryType mem_type) -> TRITONSERVER_Error* {
        for (size_t idx = 0; idx < devices.size(); ++idx) {
          // skip if no allocation
          if (devices[idx] == 0) {
            continue;
          }
          // there is space in usage array
          if (usage_idx >= buffer_attributes_.size()) {
            buffer_attributes_.emplace_back(nullptr);
            RETURN_IF_ERROR(
                TRITONSERVER_BufferAttributesNew(&buffer_attributes_.back()));
          }
          auto entry = buffer_attributes_[usage_idx];

          RETURN_IF_ERROR(
              TRITONSERVER_BufferAttributesSetMemoryType(entry, mem_type));
          RETURN_IF_ERROR(
              TRITONSERVER_BufferAttributesSetMemoryTypeId(entry, idx));
          RETURN_IF_ERROR(
              TRITONSERVER_BufferAttributesSetByteSize(entry, devices[idx]));

          ++usage_idx;
        }
        return nullptr;  // success
      };

      RETURN_IF_ERROR(set_attributes_for_device_fn(
          system_memory_usage_byte_, TRITONSERVER_MEMORY_CPU));
      RETURN_IF_ERROR(set_attributes_for_device_fn(
          pinned_memory_usage_byte_, TRITONSERVER_MEMORY_CPU_PINNED));
      RETURN_IF_ERROR(set_attributes_for_device_fn(
          cuda_memory_usage_byte_, TRITONSERVER_MEMORY_GPU));

      *usage_size = usage_idx;
      *usage = buffer_attributes_.data();
      return nullptr;
    }

    // Byte size of allocated memory tracked,
    // 'system_memory_usage_byte_' is likely to be empty as system memory
    // allocation is not controlled by CUDA driver. But keeping it for
    // completeness.
    std::vector<int64_t> system_memory_usage_byte_{0};
    std::vector<int64_t> pinned_memory_usage_byte_{0};
    std::vector<int64_t> cuda_memory_usage_byte_{0};
    bool tracked_{false};

    std::vector<TRITONSERVER_BufferAttributes*> buffer_attributes_;

    TRITONBACKEND_CuptiTracker cupti_tracker_;
  };

  // Simple scope guard to make sure memory usage is untracked without coupling
  // with MemoryUsage lifecycle
  struct ScopeGuard {
    ScopeGuard(MemoryUsage* usage) : usage_(usage) {}
    ~ScopeGuard()
    {
      if (usage_ && usage_->tracked_) {
        UntrackThreadMemoryUsage(usage_);
      }
    }
    MemoryUsage* usage_{nullptr};
  };


#if defined(TRITON_ENABLE_GPU) && defined(TRITON_ENABLE_MEMORY_TRACKER)
  static bool Init();
  static void Fini();

  static int CudaDeviceCount();

  // The memory usage will be tracked and modified until it's untracked, 'usage'
  // must be valid and not to be modified externally until untrack is called.
  // Currently can distinguish activity by correlation id which is thread
  // specific, which implies that there will be missing records if tracking
  // region switching threads to handle other activities.
  // This function takes no affect if 'usage' is nullptr.
  static void TrackThreadMemoryUsage(MemoryUsage* usage);

  // Note that CUPTI always pop from the top of the thread-wise stack, must be
  // careful on the untrack order if there is need to use multiple MemoryUsage
  // objects.
  // This function takes no affect if 'usage' is nullptr.
  static void UntrackThreadMemoryUsage(MemoryUsage* usage);

  static bool EnableFromBackendConfig(
      triton::common::TritonJson::Value& backend_config)
  {
    triton::common::TritonJson::Value cmdline;
    if (backend_config.Find("cmdline", &cmdline)) {
      triton::common::TritonJson::Value value;
      std::string value_str;
      if (cmdline.Find("triton-backend-memory-tracker", &value)) {
        bool lvalue = false;
        auto err = value.AsString(&value_str);
        if (err != nullptr) {
          LOG_IF_ERROR(err, "Error parsing backend config: ");
          return false;
        }
        err = ParseBoolValue(value_str, &lvalue);
        if (err != nullptr) {
          LOG_IF_ERROR(err, "Error parsing backend config: ");
          return false;
        }
        return lvalue;
      }
    }
    return false;
  }

  ~DeviceMemoryTracker();

  static void TrackActivity(CUpti_Activity* record)
  {
    if (tracker_) {
      tracker_->TrackActivityInternal(record);
    }
  }

 private:
  DeviceMemoryTracker();

  void TrackActivityInternal(CUpti_Activity* record);
  bool UpdateMemoryTypeUsage(
      CUpti_ActivityMemory3* memory_record, const bool is_allocation,
      int64_t* memory_usage, uint32_t usage_len);

  std::mutex mtx_;
  std::unordered_map<uint32_t, uintptr_t> activity_to_memory_usage_;
  CUpti_SubscriberHandle subscriber_{nullptr};
  int device_cnt_{0};

  static std::unique_ptr<DeviceMemoryTracker> tracker_;
#else   // no-ops
  static bool Init() { return false; }
  static void Fini() {}
  static int CudaDeviceCount() { return 0; }
  static void TrackThreadMemoryUsage(MemoryUsage* usage) {}
  static void UntrackThreadMemoryUsage(MemoryUsage* usage) {}
  static bool EnableFromBackendConfig(
      const triton::common::TritonJson::Value& backend_config)
  {
    return false;
  }
#endif  // TRITON_ENABLE_GPU && TRITON_ENABLE_MEMORY_TRACKER
};

}}  // namespace triton::backend
