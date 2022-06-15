// Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <list>
#include <memory>
#include <string>
#include <vector>
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_memory.h"
#include "triton/common/async_work_queue.h"
#include "triton/common/sync_queue.h"
#include "triton/core/tritonbackend.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace backend {

#ifndef TRITON_ENABLE_GPU
using cudaStream_t = void*;
using cudaEvent_t = void*;
#endif  // !TRITON_ENABLE_GPU

//
// BackendInputCollector
//
class BackendInputCollector {
 public:
  // The caller can optionally provide 'event' for internal synchronization
  // instead of using 'stream'. If 'host_policy_name' is provided, it must be
  // valid for the lifetime of the collector
  explicit BackendInputCollector(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      TRITONBACKEND_MemoryManager* memory_manager, const bool pinned_enabled,
      cudaStream_t stream, cudaEvent_t event = nullptr,
      cudaEvent_t buffer_ready_event = nullptr,
      const size_t kernel_buffer_threshold = 0,
      const char* host_policy_name = nullptr, const bool copy_on_stream = false,
      const bool coalesce_request_input = false)
      : need_sync_(false), requests_(requests), request_count_(request_count),
        responses_(responses), memory_manager_(memory_manager),
        pinned_enabled_(pinned_enabled),
        use_async_cpu_copy_(triton::common::AsyncWorkQueue::WorkerCount() > 1),
        stream_(stream), event_(event), buffer_ready_event_(buffer_ready_event),
        kernel_buffer_threshold_(kernel_buffer_threshold),
        pending_pinned_byte_size_(0), pending_pinned_offset_(0),
        pending_copy_kernel_buffer_byte_size_(0),
        pending_copy_kernel_buffer_offset_(0),
        pending_copy_kernel_input_buffer_counts_(0), async_task_count_(0),
        host_policy_cstr_(host_policy_name), copy_on_stream_(copy_on_stream),
        coalesce_request_input_(coalesce_request_input)
  {
  }

  ~BackendInputCollector() = default;

  // Process all requests for a named input tensor and return the
  // concatenated values of those requests in a single contiguous
  // buffer. This overload of the function can avoid data copy if the
  // tensor values are already contiguous and the caller doesn't
  // provide a destination 'buffer'.
  //
  // 'buffer' is used to determine whether the input should be placed at the
  //   'buffer' provided by the caller. If 'buffer' == nullptr, the returned
  //   buffer will be managed by the BackendInputCollector object and
  //   has the same lifecycle as the BackendInputCollector object.
  // 'buffer_byte_size' is the byte size of 'buffer' if it is not nullptr.
  // 'allowed_input_types' is the ordered list of the memory type and id pairs
  //   that the returned buffer can be. It must only contain the memory type
  //   and id of 'buffer' if 'buffer' is not nullptr.
  // 'dst_buffer' returns the contiguous buffer of the input tensor.
  // 'dst_buffer_byte_size' the byte size of 'dst_buffer'.
  // 'dst_memory_type' returns the memory type of 'dst_buffer'.
  // 'dst_memory_type_id' returns the memory type id of 'dst_buffer'.
  TRITONSERVER_Error* ProcessTensor(
      const char* input_name, char* buffer, const size_t buffer_byte_size,
      const std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>>&
          allowed_input_types,
      const char** dst_buffer, size_t* dst_buffer_byte_size,
      TRITONSERVER_MemoryType* dst_memory_type, int64_t* dst_memory_type_id);

  // Process all requests for a named input tensor and return the
  // concatenated values of those requests in a single contiguous
  // 'buffer'.
  //
  // 'buffer' The buffer to hold the concatenates tensor value. Must
  // be large enough to hold all tensor value.
  // 'buffer_byte_size' is the byte size of 'buffer'.
  // 'dst_memory_type' The memory type of 'buffer'.
  // 'dst_memory_type_id' The memory type id of 'buffer'.
  void ProcessTensor(
      const char* input_name, char* buffer, const size_t buffer_byte_size,
      const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id);

  // Process the batch input and return its shape. Returning error indicates
  // that the batch input can't be formed properly and the caller should abort
  // the whole batch.
  TRITONSERVER_Error* BatchInputShape(
      const BatchInput& batch_input, std::vector<int64_t>* shape);

  // Process the batch input and derive its value into 'buffer'. Returning
  // error indicates that the batch input can't be formed properly and
  // the caller should abort the whole batch.
  // 'buffer' is used to determine whether the input should be placed at the
  //   'buffer' provided by the caller. If 'buffer' == nullptr, the returned
  //   buffer will be managed by the BackendInputCollector object and
  //   has the same lifecycle as the BackendInputCollector object.
  // 'buffer_byte_size' is the byte size of 'buffer' if it is not nullptr.
  // 'allowed_input_types' is the ordered list of the memory type and id pairs
  //   that the returned buffer can be. It must only contain the memory type
  //   and id of 'buffer' if it is not nullptr.
  // 'dst_buffer' returns the contiguous buffer of the input tensor.
  // 'dst_memory_type' returns the memory type of 'dst_buffer'.
  // 'dst_memory_type_id' returns the memory type id of 'dst_buffer'.
  TRITONSERVER_Error* ProcessBatchInput(
      const BatchInput& batch_input, char* buffer,
      const size_t buffer_byte_size,
      const std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>>&
          allowed_input_types,
      const char** dst_buffer, size_t* dst_buffer_byte_size,
      TRITONSERVER_MemoryType* dst_memory_type, int64_t* dst_memory_type_id);

  // Finalize processing of all requests for all input tensors. Return
  // true if cudaMemcpyAsync is called, and the caller should call
  // cudaStreamSynchronize (or cudaEventSynchronize on 'event') before
  // using the data.
  bool Finalize();

 private:
  struct ContiguousBuffer {
    ContiguousBuffer() : start_request_idx_(0), end_request_idx_(0) {}
    MemoryDesc memory_desc_;
    size_t start_request_idx_;
    size_t end_request_idx_;
  };

  class InputIterator {
   public:
    InputIterator(
        TRITONBACKEND_Request** requests, const uint32_t request_count,
        std::vector<TRITONBACKEND_Response*>* responses, const char* input_name,
        const char* host_policy_name, const bool coalesce_request_input);

    // Return false if iterator reaches the end of inputs, 'input' is not set.
    bool GetNextContiguousInput(ContiguousBuffer* input);

   private:
    TRITONBACKEND_Request** requests_;
    const uint32_t request_count_;
    std::vector<TRITONBACKEND_Response*>* responses_;
    const char* input_name_;
    const char* host_policy_;
    const bool coalesce_request_input_;

    TRITONBACKEND_Input* curr_input_;
    size_t curr_request_idx_;
    size_t curr_buffer_idx_;
    uint32_t curr_buffer_cnt_;
    bool reach_end_;
  };

  // Return whether the entire input is in a contiguous buffer. If returns true,
  // the properties of the contiguous input buffer will also be returned.
  // Otherwise, only 'buffer_byte_size' will be set and return the total byte
  // size of the input.
  bool GetInputBufferIfContiguous(
      const char* input_name, const char** buffer, size_t* buffer_byte_size,
      TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id);
  bool FlushPendingPinned(
      char* tensor_buffer, const size_t tensor_buffer_byte_size,
      const TRITONSERVER_MemoryType tensor_memory_type,
      const int64_t tensor_memory_type_id);
  bool FlushPendingCopyKernel(
      char* tensor_buffer, const size_t tensor_buffer_byte_size,
      const TRITONSERVER_MemoryType tensor_memory_type,
      const int64_t tensor_memory_type_id);
  TRITONSERVER_Error* LaunchCopyKernel(
      char* tensor_buffer, const size_t tensor_buffer_byte_size,
      const TRITONSERVER_MemoryType tensor_memory_type,
      const int64_t tensor_memory_type_id);
  bool SetInputTensor(
      const char* input_name, const ContiguousBuffer& input,
      char* tensor_buffer, const size_t tensor_buffer_byte_size,
      const TRITONSERVER_MemoryType tensor_memory_type,
      const int64_t tensor_memory_type_id, const size_t tensor_buffer_offset,
      const TRITONSERVER_MemoryType use_pinned_memory_type,
      const bool use_kernel, const bool wait_buffer);
  template <typename T>
  TRITONSERVER_Error* SetElementCount(
      const std::string& source_input, char* buffer,
      const size_t buffer_byte_size);
  template <typename T>
  TRITONSERVER_Error* SetAccumulatedElementCount(
      const std::string& source_input, char* buffer,
      const size_t buffer_byte_size);
  template <typename T>
  TRITONSERVER_Error* SetBatchItemShape(
      const std::string& source_input, char* buffer,
      const size_t buffer_byte_size);

  bool need_sync_;
  TRITONBACKEND_Request** requests_;
  const uint32_t request_count_;
  std::vector<TRITONBACKEND_Response*>* responses_;
  TRITONBACKEND_MemoryManager* memory_manager_;
  const bool pinned_enabled_;
  const bool use_async_cpu_copy_;
  cudaStream_t stream_;
  cudaEvent_t event_;
  cudaEvent_t buffer_ready_event_;
  const size_t kernel_buffer_threshold_;

  size_t pending_pinned_byte_size_;
  size_t pending_pinned_offset_;
  std::list<ContiguousBuffer> pending_pinned_input_buffers_;

  // managed memories that need to live over the lifetime of this
  // BackendInputCollector object.
  std::list<std::unique_ptr<BackendMemory>> in_use_memories_;

  size_t pending_copy_kernel_buffer_byte_size_;
  size_t pending_copy_kernel_buffer_offset_;
  size_t pending_copy_kernel_input_buffer_counts_;
  std::list<ContiguousBuffer> pending_copy_kernel_input_buffers_;
  std::vector<std::unique_ptr<std::vector<int8_t*>>> input_ptr_buffer_host_;
  std::vector<std::unique_ptr<std::vector<size_t>>> byte_size_buffer_host_;
  std::vector<std::unique_ptr<std::vector<size_t>>>
      byte_size_offset_buffer_host_;

  // Pinned memory buffers and the corresponding request_inputs where
  // the final copy to the tensor is deferred until Finalize() after
  // waiting for all in-flight copies.
  struct DeferredPinned {
    DeferredPinned(
        char* pinned_memory, const size_t pinned_memory_size,
        char* tensor_buffer, const size_t tensor_buffer_offset,
        const TRITONSERVER_MemoryType tensor_memory_type,
        const int64_t tensor_memory_id,
        std::list<ContiguousBuffer>&& request_buffers,
        std::vector<TRITONBACKEND_Response*>* responses)
        : finalized_(false), pinned_memory_(pinned_memory),
          pinned_memory_size_(pinned_memory_size),
          tensor_buffer_(tensor_buffer),
          tensor_buffer_offset_(tensor_buffer_offset),
          tensor_memory_type_(tensor_memory_type),
          tensor_memory_id_(tensor_memory_id),
          requests_(std::move(request_buffers)), responses_(responses)
    {
    }

    bool Finalize(cudaStream_t stream);
    bool finalized_;
    // Holding reference to the pinned memory buffer, which is managed
    // by BackendInputCollector as 'pinned_memory'
    char* pinned_memory_;
    const size_t pinned_memory_size_;
    char* tensor_buffer_;
    const size_t tensor_buffer_offset_;
    const TRITONSERVER_MemoryType tensor_memory_type_;
    const int64_t tensor_memory_id_;
    std::list<ContiguousBuffer> requests_;
    std::vector<TRITONBACKEND_Response*>* responses_;
  };

  std::list<DeferredPinned> deferred_pinned_;
  // FIXME use future to maintain an issue-order queue to drop task count
  triton::common::SyncQueue<bool> completion_queue_;
  size_t async_task_count_;

  const char* host_policy_cstr_;
  const bool copy_on_stream_;
  const bool coalesce_request_input_;
};

}}  // namespace triton::backend
