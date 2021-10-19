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
#pragma once

#include <string>
#include <vector>
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend {

// Colletion of common properties that describes a buffer in Triton
struct MemoryDesc {
  MemoryDesc()
      : buffer_(nullptr), byte_size_(0), memory_type_(TRITONSERVER_MEMORY_CPU),
        memory_type_id_(0)
  {
  }
  MemoryDesc(
      const char* buffer, size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id)
      : buffer_(buffer), byte_size_(byte_size), memory_type_(memory_type),
        memory_type_id_(memory_type_id)
  {
  }
  const char* buffer_;
  size_t byte_size_;
  TRITONSERVER_MemoryType memory_type_;
  int64_t memory_type_id_;
};

//
// BackendMemory
//
// Utility class for allocating and deallocating memory using both
// TRITONBACKEND_MemoryManager and direct GPU and CPU malloc/free.
//
class BackendMemory {
 public:
  enum class AllocationType { CPU, CPU_PINNED, GPU, CPU_PINNED_POOL, GPU_POOL };

  // Allocate a contiguous block of 'alloc_type' memory.  'mem'
  // returns the pointer to the allocated memory.
  //
  // CPU, CPU_PINNED_POOL and GPU_POOL are allocated using
  // TRITONBACKEND_MemoryManagerAllocate. Note that CPU_PINNED and GPU
  // allocations can be much slower than the POOL variants.
  //
  // Two error codes have specific interpretations for this function:
  //
  //   TRITONSERVER_ERROR_UNSUPPORTED: Indicates that function is
  //     incapable of allocating the requested memory type and memory
  //     type ID. Requests for the memory type and ID will always fail
  //     no matter 'byte_size' of the request.
  //
  //   TRITONSERVER_ERROR_UNAVAILABLE: Indicates that function can
  //      allocate the memory type and ID but that currently it cannot
  //      allocate a contiguous block of memory of the requested
  //      'byte_size'.
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_MemoryManager* manager, const AllocationType alloc_type,
      const int64_t memory_type_id, const size_t byte_size,
      BackendMemory** mem);

  // Allocate a contiguous block of memory by attempting the
  // allocation using 'alloc_types' in order until one is successful.
  // See BackendMemory::Create() above for details.
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_MemoryManager* manager,
      const std::vector<AllocationType>& alloc_types,
      const int64_t memory_type_id, const size_t byte_size,
      BackendMemory** mem);

  // Creates a BackendMemory object from a pre-allocated buffer. The buffer
  // is not owned by the object created with this function. Hence, for
  // proper operation, the lifetime of the buffer should atleast extend till
  // the corresponding BackendMemory.
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_MemoryManager* manager, const AllocationType alloc_type,
      const int64_t memory_type_id, void* buffer, const size_t byte_size,
      BackendMemory** mem);

  ~BackendMemory();

  AllocationType AllocType() const { return alloctype_; }
  int64_t MemoryTypeId() const { return memtype_id_; }
  char* MemoryPtr() { return buffer_; }
  size_t ByteSize() const { return byte_size_; }
  TRITONSERVER_MemoryType MemoryType() const
  {
    return AllocTypeToMemoryType(alloctype_);
  }

  static TRITONSERVER_MemoryType AllocTypeToMemoryType(const AllocationType a);
  static const char* AllocTypeString(const AllocationType a);

 private:
  BackendMemory(
      TRITONBACKEND_MemoryManager* manager, const AllocationType alloctype,
      const int64_t memtype_id, char* buffer, const size_t byte_size,
      const bool owns_buffer = true)
      : manager_(manager), alloctype_(alloctype), memtype_id_(memtype_id),
        buffer_(buffer), byte_size_(byte_size), owns_buffer_(owns_buffer)
  {
  }

  TRITONBACKEND_MemoryManager* manager_;
  AllocationType alloctype_;
  int64_t memtype_id_;
  char* buffer_;
  size_t byte_size_;
  bool owns_buffer_;
};

}}  // namespace triton::backend
