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

#ifdef TRITON_ENABLE_GPU
#include <cupti.h>
#endif

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
///  - Call DeviceMemoryTracker::Init
///
/// On TRITONBACKEND_GetBackendAttribute
///  - Call TRITONBACKEND_BackendAttributeSetEnableMemoryTracker with value
///    returned from DeviceMemoryTracker::Init
///
/// If true, call DeviceMemoryTracker::TrackThreadMemoryUsage and
/// DeviceMemoryTracker::UntrackThreadMemoryUsage accordingly to track memory
/// allocation in the scope between the function, the memory usage should be
/// stored in MemoryUsage object that has the same lifetime as
/// TRITONBACKEND_Model object.
///
/// On TRITONBACKEND_ModelMemoryUsage
///  - Call MemoryUsage::SerializeToBufferAttributes to set the value
///    to be returned.

extern "C" {

typedef struct TRITONBACKEND_CuptiTracker_t {
  // C struct require extra implementation for dynamic array, for simplicity,
  // the following assumptions are made to pre-allocate the array with max
  // possible length:
  //  - system / pinned memory allocation should only be on deviceId 0
  //  - CUDA allocation will only be on visible CUDA devices
  int64_t* system_byte_size_;
  int64_t* pinned_byte_size_;
  int64_t* cuda_byte_size_;
  uint32_t system_array_len_;
  uint32_t pinned_array_len_;
  uint32_t cuda_array_len_;

  // only set to false if somehow the CUPTI activity occurs on index out of
  // range. In that case, user should invalidate the whole tracker.
  bool good_record_;
} TRITONBACKEND_CuptiTracker;

}

class DeviceMemoryTracker {
 public:
  struct MemoryUsage {
    MemoryUsage()
    {
      cuda_byte_size_.resize(CudaDeviceCount(), 0);

      cupti_tracker_.system_byte_size_ = system_byte_size_.data();
      cupti_tracker_.pinned_byte_size_ = pinned_byte_size_.data();
      cupti_tracker_.cuda_byte_size_ = cuda_byte_size_.data();
      cupti_tracker_.system_array_len_ = system_byte_size_.size();
      cupti_tracker_.pinned_array_len_ = pinned_byte_size_.size();
      cupti_tracker_.cuda_array_len_ = cuda_byte_size_.size();
      cupti_tracker_.good_record_ = true;
    }

    ~MemoryUsage()
    {
      // Make sure all C struct reference are dropped before clearing.
      if (tracked_) {
        UntrackThreadMemoryUsage(usage_);
      }
    }

    // Disable copy and assign to better manage C struct lifecycle
    MemoryUsage(const MemoryUsage&) = delete;
    void operator=(const MemoryUsage&) = delete;

    // merge record from another MemoryUsage object
    MemoryUsage& operator+=(const MemoryUsage& rhs)
    {
      std::transform(rhs.system_byte_size_.begin(), rhs.system_byte_size_.end(), system_byte_size_.begin(),
      std::plus<int64_T>());
      std::transform(rhs.pinned_byte_size_.begin(), rhs.pinned_byte_size_.end(), pinned_byte_size_.begin(),
      std::plus<int64_T>());
      std::transform(rhs.cuda_byte_size_.begin(), rhs.cuda_byte_size_.end(), cuda_byte_size_.begin(),
      std::plus<int64_T>());
    }

    TRITONSERVER_Error* SerializeToBufferAttributes(
      TRITONSERVER_BufferAttributes** usage, int32_t* usage_size)
    {
      int32_t usage_idx = 0;

      // Define lambda to convert an vector of memory usage of the same type of
      // device into buffer attributes and set in 'usage'
      auto set_attributes_for_device_fn = [&](const std::vector<int64_t>& devices, const TRITONSERVER_MemoryType mem_type) {
        for (size_t idx=0; idx < devices.size(); ++idx) {
          // skip if no allocation
          if (devices[idx] == 0) {
            continue;
          }
          // there is space in usage array 
          if (usage_idx < *usage_size) {
            auto entry = usage[usage_idx];

            RETURN_IF_ERROR(TRITONSERVER_BufferAttributesSetMemoryType(entry, mem_type));
            RETURN_IF_ERROR(TRITONSERVER_BufferAttributesSetMemoryTypeId(entry, idx));
            RETURN_IF_ERROR(TRITONSERVER_BufferAttributesSetByteSize(entry, devices[idx]));

            ++usage_idx;
          } else {
            *usage_size = -1;
            return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "More entries needed to store memory usage");
          }
        }
        return nullptr;  // success
      };

      RETURN_IF_ERROR(set_attributes_for_device_fn(system_byte_size_, TRITONSERVER_MEMORY_CPU));
      RETURN_IF_ERROR(set_attributes_for_device_fn(pinned_byte_size_, TRITONSERVER_MEMORY_CPU_PINNED));
      RETURN_IF_ERROR(set_attributes_for_device_fn(cuda_byte_size_, TRITONSERVER_MEMORY_GPU));

      *usage_size = usage_idx;
      return nullptr;
    }

    // Byte size of allocated memory tracked,
    // 'system_byte_size_' is likely to be empty as system memory allocation
    // is not controlled by CUDA driver. But keeping it for completeness.
    std::vector<int64_t> system_byte_size_{0};
    std::vector<int64_t> pinned_byte_size_{0};
    std::vector<int64_t> cuda_byte_size_{0};
    bool tracked_{false};

    TRITONBACKEND_CuptiTracker cupti_tracker_;
  };

  // Simple scope guard to make sure memory usage is untracked without coupling
  // with MemoryUsage lifecycle
  struct ScopeGuard {
    ScopeGuard(MemoryUsage* usage) : usage_(usage) {}
    ~ScopeGuard() {
      if (usage_ && usage_->tracked_) {
        UntrackThreadMemoryUsage(usage_);
      }
    }
    MemoryUsage* usage_{nullptr};
  };
  

#ifdef TRITON_ENABLE_GPU
  static bool Init();

  static int CudaDeviceCount();

  // The memory usage will be tracked and modified until it's untracked, 'usage'
  // must be valid and not to be modified externally until untrack is called.
  // Currently can distinguish activity by correlation id which is thread
  // specific, which implies that there will be mssing records if tracking
  // region switching threads to handle other activities.
  static void TrackThreadMemoryUsage(MemoryUsage* usage);

  // Note that CUPTI always pop from the top of the thread-wise stack, must be
  // careful on the untrack order if there is need to use multiple MemoryUsage
  // objects.
  static void UntrackThreadMemoryUsage(MemoryUsage* usage);

  static void TrackActivity(CUpti_Activity* record)
  {
    tracker_->TrackActivityInternal(record);
  }

  static bool EnableFromBackendConfig(const triton::common::TritonJson::Value& backend_config)
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

  ~DeviceMemoryTracker() {
    if (subscriber_) {
      cuptiUnsubscribe(subscriber_);
    }
  }

 private:
  DeviceMemoryTracker();

  void TrackActivityInternal(CUpti_Activity* record);

  std::mutex mtx_;
  std::unordered_map<uint32_t, uintptr_t> activity_to_memory_usage_;
  CUpti_SubscriberHandle subscriber_{nullptr};
  int device_cnt_{0};

  static std::unique_ptr<DeviceMemoryTracker> tracker_;
#else // no-ops
  static bool Init() { return false; }
  static int CudaDeviceCount() {return 0; }
  static void TrackThreadMemoryUsage(MemoryUsage* usage) {}
  static void UntrackThreadMemoryUsage(MemoryUsage* usage) {}
  static bool EnableFromBackendConfig(const triton::common::TritonJson::Value& backend_config) { return false; }
#endif  // TRITON_ENABLE_GPU
};

}}  // namespace triton::backend
