// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "triton/backend/backend_memory.h"

#include <map>
#include "triton/backend/backend_common.h"

namespace triton { namespace backend {

TRITONSERVER_Error*
BackendMemory::Create(
    TRITONBACKEND_MemoryManager* manager, const AllocationType alloc_type,
    const int64_t memory_type_id, const size_t byte_size, BackendMemory** mem)
{
  *mem = nullptr;

  void* ptr = nullptr;
  switch (alloc_type) {
    case AllocationType::CPU_PINNED: {
#ifdef TRITON_ENABLE_GPU
      RETURN_IF_CUDA_ERROR(
          cudaHostAlloc(&ptr, byte_size, cudaHostAllocPortable),
          TRITONSERVER_ERROR_UNAVAILABLE,
          std::string("failed to allocate pinned system memory"));
#else
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "pinned-memory allocation not supported");
#endif  // TRITON_ENABLE_GPU
      break;
    }

    case AllocationType::GPU: {
#ifdef TRITON_ENABLE_GPU
      int current_device;
      RETURN_IF_CUDA_ERROR(
          cudaGetDevice(&current_device), TRITONSERVER_ERROR_INTERNAL,
          std::string("failed to get device"));
      bool overridden = (current_device != memory_type_id);
      if (overridden) {
        RETURN_IF_CUDA_ERROR(
            cudaSetDevice(memory_type_id), TRITONSERVER_ERROR_INTERNAL,
            std::string("failed to set device"));
      }

      auto err = cudaMalloc(&ptr, byte_size);

      if (overridden) {
        LOG_IF_CUDA_ERROR(
            cudaSetDevice(current_device), "failed to set CUDA device");
      }

      RETURN_ERROR_IF_FALSE(
          err == cudaSuccess, TRITONSERVER_ERROR_UNAVAILABLE,
          std::string("failed to allocate GPU memory: ") +
              cudaGetErrorString(err));
#else
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED, "GPU allocation not supported");
#endif  // TRITON_ENABLE_GPU
      break;
    }

    case AllocationType::CPU:
    case AllocationType::CPU_PINNED_POOL:
    case AllocationType::GPU_POOL:
      RETURN_IF_ERROR(TRITONBACKEND_MemoryManagerAllocate(
          manager, &ptr, AllocTypeToMemoryType(alloc_type), memory_type_id,
          byte_size));
      break;
  }

  *mem = new BackendMemory(
      manager, alloc_type, memory_type_id, reinterpret_cast<char*>(ptr),
      byte_size);

  return nullptr;  // success
}

TRITONSERVER_Error*
BackendMemory::Create(
    TRITONBACKEND_MemoryManager* manager,
    const std::vector<AllocationType>& alloc_types,
    const int64_t memory_type_id, const size_t byte_size, BackendMemory** mem)
{
  *mem = nullptr;
  RETURN_ERROR_IF_TRUE(
      alloc_types.size() == 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("BackendMemory::Create, at least one allocation type must be "
                  "specified"));

  bool success = false;
  std::unordered_map<AllocationType, TRITONSERVER_Error*> errors;
  for (const AllocationType alloc_type : alloc_types) {
    TRITONSERVER_Error* err =
        Create(manager, alloc_type, memory_type_id, byte_size, mem);
    if (err == nullptr) {
      success = true;
      break;
    }

    errors.insert({alloc_type, err});
  }

  // If allocation failed for all allocation types then display all
  // the error messages and show the entire allocation request as
  // failing.
  if (!success) {
    std::string msg = "BackendMemory::Create, all allocation types failed:";
    for (const auto& pr : errors) {
      const AllocationType alloc_type = pr.first;
      TRITONSERVER_Error* err = pr.second;
      msg += std::string("\n\t") + AllocTypeString(alloc_type) + ": " +
             TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
    }

    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNAVAILABLE, msg.c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
BackendMemory::Create(
    TRITONBACKEND_MemoryManager* manager, const AllocationType alloc_type,
    const int64_t memory_type_id, void* buffer, const size_t byte_size,
    BackendMemory** mem)
{
  *mem = new BackendMemory(
      manager, alloc_type, memory_type_id, reinterpret_cast<char*>(buffer),
      byte_size, false /* owns_buffer */);

  return nullptr;  // success
}

BackendMemory::~BackendMemory()
{
  if (owns_buffer_) {
    switch (alloctype_) {
      case AllocationType::CPU_PINNED:
#ifdef TRITON_ENABLE_GPU
        if (buffer_ != nullptr) {
          LOG_IF_CUDA_ERROR(
              cudaFreeHost(buffer_), "failed to free pinned memory");
        }
#endif  // TRITON_ENABLE_GPU
        break;

      case AllocationType::GPU:
#ifdef TRITON_ENABLE_GPU
        if (buffer_ != nullptr) {
          LOG_IF_CUDA_ERROR(cudaFree(buffer_), "failed to free CUDA memory");
        }
#endif  // TRITON_ENABLE_GPU
        break;

      case AllocationType::CPU:
      case AllocationType::CPU_PINNED_POOL:
      case AllocationType::GPU_POOL:
        LOG_IF_ERROR(
            TRITONBACKEND_MemoryManagerFree(
                manager_, buffer_, AllocTypeToMemoryType(alloctype_),
                memtype_id_),
            "failed to free memory buffer");
        break;
    }
  }
}

TRITONSERVER_MemoryType
BackendMemory::AllocTypeToMemoryType(const AllocationType a)
{
  switch (a) {
    case AllocationType::CPU:
      return TRITONSERVER_MEMORY_CPU;
    case AllocationType::CPU_PINNED:
    case AllocationType::CPU_PINNED_POOL:
      return TRITONSERVER_MEMORY_CPU_PINNED;
    case AllocationType::GPU:
    case AllocationType::GPU_POOL:
      return TRITONSERVER_MEMORY_GPU;
  }

  return TRITONSERVER_MEMORY_CPU;  // unreachable
}

const char*
BackendMemory::AllocTypeString(const AllocationType a)
{
  switch (a) {
    case AllocationType::CPU:
      return "CPU";
    case AllocationType::CPU_PINNED:
      return "CPU_PINNED";
    case AllocationType::GPU:
      return "GPU";
    case AllocationType::CPU_PINNED_POOL:
      return "CPU_PINNED_POOL";
    case AllocationType::GPU_POOL:
      return "GPU_POOL";
  }

  return "<unknown>";
}

}}  // namespace triton::backend
