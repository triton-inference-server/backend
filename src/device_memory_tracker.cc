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

#include "triton/backend/device_memory_tracker.h"

#include <iostream>
#include <stdexcept>

#include "triton/core/tritonserver.h"

namespace triton { namespace backend {

std::unique_ptr<DeviceMemoryTracker> DeviceMemoryTracker::tracker_{nullptr};
// Boilerplate from CUPTI examples
namespace {

#define LOG_IF_CUPTI_ERR(call)                                \
  do {                                                        \
    CUptiResult _status = call;                               \
    if (_status != CUPTI_SUCCESS) {                           \
      const char* errstr;                                     \
      cuptiGetResultString(_status, &errstr);                 \
      LOG_ERROR << #call << " failed with error: " << errstr; \
    }                                                         \
  } while (0)

#define THROW_IF_CUPTI_ERR(call)                                 \
  do {                                                           \
    CUptiResult _status = call;                                  \
    if (_status != CUPTI_SUCCESS) {                              \
      const char* errstr;                                        \
      cuptiGetResultString(_status, &errstr);                    \
      throw std::runtime_error(                                  \
          std::string(#call) + " failed with error: " + errstr); \
    }                                                            \
  } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                 \
  (((uintptr_t)(buffer) & ((align)-1))                              \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) \
       : (buffer))

void
bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords)
{
  uint8_t* bfr = (uint8_t*)malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr != nullptr) {
    *size = BUF_SIZE;
    *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
    *maxNumRecords = 0;
  } else {
    LOG_ERROR << "Failed to allocate buffer for CUPTI: out of memory";
  }
}

void
bufferCompleted(
    CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size,
    size_t validSize)
{
  CUptiResult status;
  CUpti_Activity* record = nullptr;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        DeviceMemoryTracker::TrackActivity(record);
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        LOG_IF_CUPTI_ERR(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped = 0;
    LOG_IF_CUPTI_ERR(
        cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      LOG_WARNING << "Dropped " << dropped << " activity records";
    }
  }

  free(buffer);
}

}  // namespace

DeviceMemoryTracker::DeviceMemoryTracker()
{
  cudaError_t cuerr = cudaGetDeviceCount(&device_cnt_);
  if ((cuerr == cudaErrorNoDevice) || (cuerr == cudaErrorInsufficientDriver)) {
    device_cnt_ = 0;
  } else if (cuerr != cudaSuccess) {
    throw std::runtime_error(
        "Unexpected failure on getting CUDA device count.");
  }

  // Use 'cuptiSubscribe' to check if the cupti has been initialized
  // elsewhere. Due to cupti limitation, there can only be one cupti client
  // within the process, so in the case of per-backend memory tracking, we
  // have to make the assumption that the other cupti client is using the same
  // memory tracker implementation so that the backend may use the cupti
  // configuration that is external to the backend without issue.
  auto cupti_res = cuptiSubscribe(&subscriber_, nullptr, nullptr);
  switch (cupti_res) {
    case CUPTI_SUCCESS: {
      THROW_IF_CUPTI_ERR(
          cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
      THROW_IF_CUPTI_ERR(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
      THROW_IF_CUPTI_ERR(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2));
      THROW_IF_CUPTI_ERR(
          cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
      break;
    }
    case CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED: {
      LOG_WARNING << "CUPTI has been initialized elsewhere, assuming the "
                     "implementation is the same";
      break;
    }
    default: {
      // other error, should propagate and disable memory tracking for the
      // backend
      const char* errstr;
      cuptiGetResultString(cupti_res, &errstr);
      throw std::runtime_error(
          std::string("Unexpected failure on configuring CUPTI: ") + errstr);
    }
  }
}

DeviceMemoryTracker::~DeviceMemoryTracker()
{
  if (subscriber_) {
    cuptiUnsubscribe(subscriber_);
  }
}

int
DeviceMemoryTracker::CudaDeviceCount()
{
  if (tracker_) {
    return tracker_->device_cnt_;
  }
  throw std::runtime_error(
      "DeviceMemoryTracker::Init() must be called before using any "
      "DeviceMemoryTracker features.");
}

bool
DeviceMemoryTracker::Init()
{
  if (tracker_ == nullptr) {
    try {
      tracker_.reset(new DeviceMemoryTracker());
    }
    catch (const std::runtime_error& ex) {
      // Fail initialization
      LOG_ERROR << ex.what();
      return false;
    }
  }
  return true;
}

void
DeviceMemoryTracker::Fini()
{
  tracker_.reset();
}

void
DeviceMemoryTracker::TrackThreadMemoryUsage(MemoryUsage* usage)
{
  if (!usage) {
    return;
  }
  if (tracker_) {
    THROW_IF_CUPTI_ERR(cuptiActivityPushExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN,
        reinterpret_cast<uint64_t>(&usage->cupti_tracker_)));
    usage->tracked_ = true;
  } else {
    throw std::runtime_error(
        "DeviceMemoryTracker::Init() must be called before using any "
        "DeviceMemoryTracker features.");
  }
}

void
DeviceMemoryTracker::UntrackThreadMemoryUsage(MemoryUsage* usage)
{
  if (!usage) {
    return;
  }
  if (tracker_) {
    THROW_IF_CUPTI_ERR(cuptiActivityFlushAll(0));
    uint64_t id = 0;
    THROW_IF_CUPTI_ERR(cuptiActivityPopExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id));
    usage->tracked_ = false;
  } else {
    throw std::runtime_error(
        "DeviceMemoryTracker::Init() must be called before using any "
        "DeviceMemoryTracker features.");
  }
}

void
DeviceMemoryTracker::TrackActivityInternal(CUpti_Activity* record)
{
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_MEMORY2: {
      CUpti_ActivityMemory3* memory_record = (CUpti_ActivityMemory3*)record;
      TRITONBACKEND_CuptiTracker* usage = nullptr;
      {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = activity_to_memory_usage_.find(memory_record->correlationId);
        if (it != activity_to_memory_usage_.end()) {
          usage = reinterpret_cast<TRITONBACKEND_CuptiTracker*>(it->second);
          activity_to_memory_usage_.erase(it);
        }
      }
      const bool is_allocation =
          (memory_record->memoryOperationType ==
           CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION);
      const bool is_release =
          (memory_record->memoryOperationType ==
           CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE);
      // Ignore memory record that is not associated with a
      // TRITONBACKEND_CuptiTracker object or not related to allocations
      if ((usage == nullptr) || (!usage->valid_) ||
          (!is_allocation && !is_release)) {
        break;
      }

      switch (memory_record->memoryKind) {
        case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE: {
          usage->valid_ = UpdateMemoryTypeUsage(
              memory_record, is_allocation, usage->cuda_memory_usage_byte_,
              usage->cuda_array_len_);
          break;
        }
        case CUPTI_ACTIVITY_MEMORY_KIND_PINNED: {
          usage->valid_ = UpdateMemoryTypeUsage(
              memory_record, is_allocation, usage->pinned_memory_usage_byte_,
              usage->pinned_array_len_);
          break;
        }
        case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE: {
          usage->valid_ = UpdateMemoryTypeUsage(
              memory_record, is_allocation, usage->system_memory_usage_byte_,
              usage->system_array_len_);
          break;
        }
        default:
          LOG_WARNING << "Unrecognized type of memory is allocated, kind "
                      << memory_record->memoryKind;
          usage->valid_ = false;
          break;
      }
      break;
    }
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
      CUpti_ActivityExternalCorrelation* corr =
          (CUpti_ActivityExternalCorrelation*)record;
      if (CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN == corr->externalKind) {
        std::lock_guard<std::mutex> lk(mtx_);
        activity_to_memory_usage_[corr->correlationId] =
            static_cast<uintptr_t>(corr->externalId);
      }
      break;
    }
    case CUPTI_ACTIVITY_KIND_RUNTIME: {
      // DO NOTHING, runtime API will be captured and reported to properly
      // initialize records for CUPTI_ACTIVITY_KIND_MEMORY2.
      break;
    }
    default:
      LOG_ERROR << "Unexpected capture of cupti record, kind: " << record->kind;
      break;
  }
}

inline bool
DeviceMemoryTracker::UpdateMemoryTypeUsage(
    CUpti_ActivityMemory3* memory_record, const bool is_allocation,
    int64_t* memory_usage, uint32_t usage_len)
{
  if (memory_record->deviceId >= usage_len) {
    return false;
  }
  if (is_allocation) {
    memory_usage[memory_record->deviceId] += memory_record->bytes;
  } else {
    memory_usage[memory_record->deviceId] -= memory_record->bytes;
  }
  return true;
}

}}  // namespace triton::backend
