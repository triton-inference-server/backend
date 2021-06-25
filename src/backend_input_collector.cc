// Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "triton/backend/backend_input_collector.h"

#include <atomic>
#include "triton/backend/backend_common.h"
#ifdef TRITON_ENABLE_GPU
#include "kernel.h"
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace backend {
//
// BackendInputCollector
//

bool
BackendInputCollector::GetInputBufferIfContiguous(
    const char* input_name, const char** buffer, size_t* buffer_byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  *buffer = nullptr;
  *buffer_byte_size = 0;
  const char* expected_next_buffer = nullptr;
  bool contiguous = true;
  for (size_t idx = 0; idx < request_count_; idx++) {
    auto& request = requests_[idx];
    auto& response = (*responses_)[idx];

    TRITONBACKEND_Input* input;
    RESPOND_AND_SET_NULL_IF_ERROR(
        &response, TRITONBACKEND_RequestInput(request, input_name, &input));
    uint64_t byte_size;
    uint32_t buffer_count;
    RESPOND_AND_SET_NULL_IF_ERROR(
        &response, TRITONBACKEND_InputPropertiesForHostPolicy(
                       input, host_policy_cstr_, nullptr, nullptr, nullptr,
                       nullptr, &byte_size, &buffer_count));
    for (size_t idx = 0; idx < buffer_count; ++idx) {
      const void* src_buffer;
      size_t src_byte_size;
      TRITONSERVER_MemoryType src_memory_type;
      int64_t src_memory_type_id;

      RESPOND_AND_SET_NULL_IF_ERROR(
          &response,
          TRITONBACKEND_InputBufferForHostPolicy(
              input, host_policy_cstr_, idx, &src_buffer, &src_byte_size,
              &src_memory_type, &src_memory_type_id));
      if (*buffer != nullptr) {
        if ((expected_next_buffer == src_buffer) &&
            (*memory_type == src_memory_type) &&
            (*memory_type_id == src_memory_type_id)) {
          expected_next_buffer += src_byte_size;
        } else {
          contiguous = false;
        }
        // Want to know total buffer byte size even if it is not contiguous
        *buffer_byte_size += src_byte_size;
      } else {
        *buffer = reinterpret_cast<const char*>(src_buffer);
        *memory_type = src_memory_type;
        *memory_type_id = src_memory_type_id;
        *buffer_byte_size = src_byte_size;
        expected_next_buffer = *buffer + src_byte_size;
      }
    }
  }
  return contiguous;
}

void
BackendInputCollector::ProcessTensor(
    const char* input_name, char* buffer, const size_t buffer_byte_size,
    const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id)
{
  // A value of CPU_PINNED indicates that pinned memory buffer is not
  // needed for this tensor. Any other value indicates that a pinned
  // memory buffer is needed when the target memory type matches
  // 'use_pinned_memory_type'.
  TRITONSERVER_MemoryType use_pinned_memory_type =
      TRITONSERVER_MEMORY_CPU_PINNED;
  if (pinned_enabled_) {
    use_pinned_memory_type = GetUsePinnedMemoryType(memory_type);
  }
  const bool use_kernel = (kernel_buffer_threshold_ != 0);

  size_t buffer_offset = 0;

  for (size_t idx = 0; idx < request_count_; idx++) {
    auto& request = requests_[idx];
    auto& response = (*responses_)[idx];

    // If there are pending copies from tensor buffer that is not
    // contiguous with 'response's part of that buffer, then need to
    // go ahead and perform the pending copies so that can start a new
    // contiguous region if necessary.
    if ((pending_pinned_byte_size_ > 0) &&
        (buffer_offset !=
         (pending_pinned_byte_size_ + pending_pinned_offset_))) {
      need_sync_ |= FlushPendingPinned(
          buffer, buffer_byte_size, memory_type, memory_type_id);
    }
    if ((pending_copy_kernel_buffer_byte_size_ > 0) &&
        (buffer_offset != (pending_copy_kernel_buffer_byte_size_ +
                           pending_copy_kernel_buffer_offset_))) {
      need_sync_ |= FlushPendingCopyKernel(
          buffer, buffer_byte_size, memory_type, memory_type_id);
    }

    TRITONBACKEND_Input* input;
    RESPOND_AND_SET_NULL_IF_ERROR(
        &response, TRITONBACKEND_RequestInput(request, input_name, &input));
    uint64_t byte_size;
    RESPOND_AND_SET_NULL_IF_ERROR(
        &response, TRITONBACKEND_InputPropertiesForHostPolicy(
                       input, host_policy_cstr_, nullptr, nullptr, nullptr,
                       nullptr, &byte_size, nullptr));
    if (response != nullptr) {
      need_sync_ |= SetFixedSizeInputTensor(
          input, buffer_offset, buffer, buffer_byte_size, memory_type,
          memory_type_id, use_pinned_memory_type, use_kernel, true, &response);
    }

    buffer_offset += byte_size;
  }

  // Done with the tensor, flush any pending pinned copies.
  need_sync_ |=
      FlushPendingPinned(buffer, buffer_byte_size, memory_type, memory_type_id);
  need_sync_ |= FlushPendingCopyKernel(
      buffer, buffer_byte_size, memory_type, memory_type_id);
#ifdef TRITON_ENABLE_GPU
  if (need_sync_ && (event_ != nullptr)) {
    cudaEventRecord(event_, stream_);
  }
#endif  // TRITON_ENABLE_GPU
}

TRITONSERVER_Error*
BackendInputCollector::ProcessTensor(
    const char* input_name, char* buffer, const size_t buffer_byte_size,
    const std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>>&
        allowed_input_types,
    const char** dst_buffer, size_t* dst_buffer_byte_size,
    TRITONSERVER_MemoryType* dst_memory_type, int64_t* dst_memory_type_id)
{
  if (buffer == nullptr) {
    if (allowed_input_types.size() == 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "'allowed_input_types' must contain at least one pair of memory type "
          "and id");
    }
    if (GetInputBufferIfContiguous(
            input_name, dst_buffer, dst_buffer_byte_size, dst_memory_type,
            dst_memory_type_id)) {
      // zero size buffer will be treated as contiguous as well,
      // but we want to invoke backend memory to have a valid address.
      if (*dst_buffer_byte_size != 0) {
        // If the buffer is contiguous, check if the caller expects its type
        for (const auto& allowed_type : allowed_input_types) {
          if ((*dst_memory_type == allowed_type.first) &&
              ((*dst_memory_type_id == allowed_type.second))) {
            return nullptr;  // success
          }
        }
      }
    }
    // A separate buffer is needed
    BackendMemory* backend_memory = nullptr;
    for (const auto& allowed_type : allowed_input_types) {
      std::vector<BackendMemory::AllocationType> alloc_types;
      const int64_t memory_type_id = allowed_type.second;
      switch (allowed_type.first) {
        case TRITONSERVER_MEMORY_GPU:
          alloc_types = {BackendMemory::AllocationType::GPU_POOL,
                         BackendMemory::AllocationType::GPU};
          break;
        case TRITONSERVER_MEMORY_CPU_PINNED:
          alloc_types = {BackendMemory::AllocationType::CPU_PINNED_POOL,
                         BackendMemory::AllocationType::CPU_PINNED};
          break;
        case TRITONSERVER_MEMORY_CPU:
          alloc_types = {BackendMemory::AllocationType::CPU};
          break;
      }
      auto err = BackendMemory::Create(
          memory_manager_, alloc_types, memory_type_id, *dst_buffer_byte_size,
          &backend_memory);
      if (err != nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("unable to create backend memory for type: ") +
             TRITONSERVER_MemoryTypeString(allowed_type.first) +
             " id: " + std::to_string(memory_type_id) + ": " +
             TRITONSERVER_ErrorMessage(err))
                .c_str());
        TRITONSERVER_ErrorDelete(err);
      } else {
        in_use_memories_.emplace_back(backend_memory);
        break;
      }
    }
    if (backend_memory == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("failed to allocate contiguous buffer for input '") +
           input_name + "'")
              .c_str());
    }
    buffer = backend_memory->MemoryPtr();
    *dst_buffer = backend_memory->MemoryPtr();
    *dst_buffer_byte_size = backend_memory->ByteSize();
    *dst_memory_type = backend_memory->MemoryType();
    *dst_memory_type_id = backend_memory->MemoryTypeId();
  } else {
    if (allowed_input_types.size() != 1) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "'allowed_input_types' must only contain the memory type and id of "
          "'buffer'");
    }
    *dst_buffer = buffer;
    *dst_buffer_byte_size = buffer_byte_size;
    *dst_memory_type = allowed_input_types[0].first;
    *dst_memory_type_id = allowed_input_types[0].second;
  }
  if (*dst_buffer_byte_size != 0) {
    ProcessTensor(
        input_name, buffer, *dst_buffer_byte_size, *dst_memory_type,
        *dst_memory_type_id);
  }
  return nullptr;  // success
}

bool
BackendInputCollector::Finalize()
{
#ifdef TRITON_ENABLE_GPU
  if ((!deferred_pinned_.empty()) && need_sync_) {
    if (event_ != nullptr) {
      cudaEventSynchronize(event_);
    } else {
      cudaStreamSynchronize(stream_);
    }
    need_sync_ = false;
  }
#endif  // TRITON_ENABLE_GPU

  // After the above sync all the GPU->pinned copies are complete. Any
  // deferred copies of pinned->CPU can now be done.
#ifdef TRITON_ENABLE_GPU
  if (buffer_ready_event_ != nullptr) {
    cudaEventSynchronize(buffer_ready_event_);
    buffer_ready_event_ = nullptr;
  }
#endif  // TRITON_ENABLE_GPU
  for (auto& def : deferred_pinned_) {
    if (!def.finalized_) {
      need_sync_ |= def.Finalize(stream_);
    }
  }
  for (size_t i = 0; i < async_task_count_; i++) {
    need_sync_ |= completion_queue_.Get();
  }

#ifdef TRITON_ENABLE_GPU
  // Record the new event location if deferred copies occur
  if ((!deferred_pinned_.empty()) && need_sync_ && (event_ != nullptr)) {
    cudaEventRecord(event_, stream_);
  }
#endif  // TRITON_ENABLE_GPU

  return need_sync_;
}

bool
BackendInputCollector::DeferredPinned::Finalize(cudaStream_t stream)
{
  bool cuda_used = false;
  auto err = CopyBuffer(
      "pinned buffer", TRITONSERVER_MEMORY_CPU_PINNED, 0, tensor_memory_type_,
      tensor_memory_id_, pinned_memory_size_, pinned_memory_,
      tensor_buffer_ + tensor_buffer_offset_, stream, &cuda_used);

  // If something goes wrong with the copy all the pending
  // responses fail...
  if (err != nullptr) {
    for (auto& pr : requests_) {
      auto response = pr.first;
      if (*response != nullptr) {
        LOG_IF_ERROR(
            TRITONBACKEND_ResponseSend(
                *response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
            "failed to send error response");
        *response = nullptr;
      }
    }
    TRITONSERVER_ErrorDelete(err);
  }
  return cuda_used;
}

bool
BackendInputCollector::SetFixedSizeInputTensor(
    TRITONBACKEND_Input* request_input, const size_t tensor_buffer_offset,
    char* tensor_buffer, const size_t tensor_buffer_byte_size,
    const TRITONSERVER_MemoryType tensor_memory_type,
    const int64_t tensor_memory_type_id,
    const TRITONSERVER_MemoryType use_pinned_memory_type, const bool use_kernel,
    const bool wait_buffer, TRITONBACKEND_Response** response)
{
  bool cuda_copy = false;

  const char* name;
  uint32_t buffer_count;
  RESPOND_AND_SET_NULL_IF_ERROR(
      response, TRITONBACKEND_InputPropertiesForHostPolicy(
                    request_input, host_policy_cstr_, &name, nullptr, nullptr,
                    nullptr, nullptr, &buffer_count));
  if (*response == nullptr) {
    return cuda_copy;
  }

  // First iterate through the buffers to ensure the byte size is proper
  size_t total_byte_size = 0;
  for (size_t idx = 0; idx < buffer_count; ++idx) {
    const void* src_buffer;
    size_t src_byte_size;
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;

    RESPOND_AND_SET_NULL_IF_ERROR(
        response, TRITONBACKEND_InputBufferForHostPolicy(
                      request_input, host_policy_cstr_, idx, &src_buffer,
                      &src_byte_size, &src_memory_type, &src_memory_type_id));
    total_byte_size += src_byte_size;
  }

  if ((tensor_buffer_offset + total_byte_size) > tensor_buffer_byte_size) {
    RESPOND_AND_SET_NULL_IF_ERROR(
        response,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "unexpected total byte size " +
                std::to_string(tensor_buffer_offset + total_byte_size) +
                " for input '" + name + "', expecting " +
                std::to_string(tensor_buffer_byte_size))
                .c_str()));
    return cuda_copy;
  } else if (response == nullptr) {
    return cuda_copy;
  }

  // Request input tensor data may be in multiple non-contiguous
  // buffers.
  size_t input_offset = 0;
  for (size_t idx = 0; idx < buffer_count; ++idx) {
    const void* src_buffer;
    size_t src_byte_size;
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;

    RESPOND_AND_SET_NULL_IF_ERROR(
        response, TRITONBACKEND_InputBufferForHostPolicy(
                      request_input, host_policy_cstr_, idx, &src_buffer,
                      &src_byte_size, &src_memory_type, &src_memory_type_id));
    if (*response == nullptr) {
      return cuda_copy;
    }

    // If the request buffer matches the memory type that should use an
    // intermediate pinned memory buffer for the transfer, then just
    // record the input as pending and increase the size required for
    // the intermediate pinned buffer. We only do this check for the
    // first buffer of an input and apply the same policy for all
    // buffers. So if an inputs data is split over different memory
    // types this may not be ideal but that should be a very rare
    // situation.
    if ((idx == 0) &&
        (use_pinned_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) &&
        (src_memory_type == use_pinned_memory_type)) {
      if (pending_pinned_byte_size_ == 0) {
        pending_pinned_offset_ = tensor_buffer_offset;
      }

      pending_pinned_byte_size_ += total_byte_size;
      pending_pinned_inputs_.push_back(std::make_pair(response, request_input));
      return cuda_copy;
    }
    // [FIXME] support other direction if prove to be faster, all kernel
    // handling code in this class asssumes the destination buffer is on device
    // If the request buffer and the destination buffer are accessible by all
    // GPUs (i.e. pinned, device), initiate the copy via copy CUDA kernel.
    // We only do this check for the
    // first buffer of an input and apply the same policy for all
    // buffers. So if an inputs data is split over different memory
    // types this may not be ideal but that should be a very rare
    // situation.
    // Currently checked direction:
    // pinned -> device
    // same device -> device
    // different device -> device
    if (use_kernel && (idx == 0) &&
        (src_memory_type != TRITONSERVER_MEMORY_CPU) &&
        (tensor_memory_type == TRITONSERVER_MEMORY_GPU)) {
      // [FIXME] Currently not allowing copy between devices as it requires
      // peer-to-peer access to be enabled. Peer-to-peer is enabled by default,
      // but server can still runs even if it fails to enable peer-to-peer.
      // Should provide a utility to check whether a device pair allows direct
      // access and use gather kernel accordingly
      if ((src_memory_type != TRITONSERVER_MEMORY_GPU) ||
          (src_memory_type_id == tensor_memory_type_id)) {
        if (pending_copy_kernel_buffer_byte_size_ == 0) {
          pending_copy_kernel_buffer_offset_ = tensor_buffer_offset;
        }

        pending_copy_kernel_buffer_byte_size_ += total_byte_size;
        pending_copy_kernel_input_buffer_counts_ += buffer_count;
        pending_copy_kernel_inputs_.push_back(
            std::make_pair(response, request_input));
        return cuda_copy;
      }
    }

#ifdef TRITON_ENABLE_GPU
    if (wait_buffer && (buffer_ready_event_ != nullptr)) {
      cudaEventSynchronize(buffer_ready_event_);
      buffer_ready_event_ = nullptr;
    }
#endif  // TRITON_ENABLE_GPU

    // Direct copy without intermediate pinned memory.
    bool cuda_used = false;
    RESPOND_AND_SET_NULL_IF_ERROR(
        response,
        CopyBuffer(
            name, src_memory_type, src_memory_type_id, tensor_memory_type,
            tensor_memory_type_id, src_byte_size, src_buffer,
            tensor_buffer + tensor_buffer_offset + input_offset, stream_,
            &cuda_used));
    cuda_copy |= cuda_used;
    if (*response == nullptr) {
      return cuda_copy;
    }

    input_offset += src_byte_size;
  }

  return cuda_copy;
}

bool
BackendInputCollector::FlushPendingPinned(
    char* tensor_buffer, const size_t tensor_buffer_byte_size,
    const TRITONSERVER_MemoryType tensor_memory_type,
    const int64_t tensor_memory_type_id)
{
  bool cuda_copy = false;

  // Will be copying from CPU->pinned->GPU or GPU->pinned->CPU

  // Attempt to allocate a pinned buffer to use for staging the
  // copy... if we fail to allocated the pinned buffer then we just
  // directly go CPU->GPU or GPU->CPU.
  char* pinned_memory = nullptr;
  int64_t pinned_memory_type_id = 0;
  TRITONSERVER_MemoryType pinned_memory_type;
  BackendMemory* backend_memory;
  if (pending_pinned_byte_size_ > 0) {
    TRITONSERVER_Error* err = BackendMemory::Create(
        memory_manager_,
        {BackendMemory::AllocationType::CPU_PINNED_POOL,
         BackendMemory::AllocationType::CPU_PINNED},
        0 /* memory_type_id */, pending_pinned_byte_size_, &backend_memory);
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
    } else {
      pinned_memory = backend_memory->MemoryPtr();
      pinned_memory_type = backend_memory->MemoryType();
      pinned_memory_type_id = backend_memory->MemoryTypeId();
    }
  }

  // If the pinned buffer wasn't actually allocated then just perform
  // a direct copy.
  if (pinned_memory == nullptr) {
    size_t offset = 0;
    for (auto& pr : pending_pinned_inputs_) {
      auto& response = pr.first;
      auto& request_input = pr.second;

      uint64_t byte_size;
      RESPOND_AND_SET_NULL_IF_ERROR(
          response, TRITONBACKEND_InputPropertiesForHostPolicy(
                        request_input, host_policy_cstr_, nullptr, nullptr,
                        nullptr, nullptr, &byte_size, nullptr));

      cuda_copy |= SetFixedSizeInputTensor(
          request_input, pending_pinned_offset_ + offset, tensor_buffer,
          tensor_buffer_byte_size, tensor_memory_type, tensor_memory_type_id,
          TRITONSERVER_MEMORY_CPU_PINNED, false, true, response);
      offset += byte_size;
    }
  }
  // We have a pinned buffer so copy the pending input buffer(s) into
  // the pinned memory.
  else {  // pinned_memory_type == TRITONSERVER_MEMORY_CPU_PINNED
    bool cuda_used = false;
    size_t offset = 0;
    if (!use_async_cpu_copy_) {
      for (auto& pr : pending_pinned_inputs_) {
        auto& response = pr.first;
        auto& request_input = pr.second;

        uint64_t byte_size;
        RESPOND_AND_SET_NULL_IF_ERROR(
            response, TRITONBACKEND_InputPropertiesForHostPolicy(
                          request_input, host_policy_cstr_, nullptr, nullptr,
                          nullptr, nullptr, &byte_size, nullptr));

        cuda_used |= SetFixedSizeInputTensor(
            request_input, offset, pinned_memory, pending_pinned_byte_size_,
            TRITONSERVER_MEMORY_CPU_PINNED, 0 /* memory_type_id */,
            TRITONSERVER_MEMORY_CPU_PINNED, false, false, response);
        offset += byte_size;
      }

      cuda_copy |= cuda_used;

      // If the copy was not async (i.e. if request input was in CPU so
      // a CPU->CPU-PINNED copy was performed above), then the pinned
      // buffer now holds the tensor contents and we can immediately
      // issue the copies from the pinned buffer to the tensor.
      //
      // Otherwise the GPU->CPU-PINNED async copies are in flight and we
      // simply remember the pinned buffer and the corresponding
      // request inputs so that we can do the pinned->CPU copies in
      // finalize after we have waited for all async copies to complete.
      if (!cuda_used) {
#ifdef TRITON_ENABLE_GPU
        if (buffer_ready_event_ != nullptr) {
          cudaEventSynchronize(buffer_ready_event_);
          buffer_ready_event_ = nullptr;
        }
#endif  // TRITON_ENABLE_GPU
        auto err = CopyBuffer(
            "pinned input buffer H2D", TRITONSERVER_MEMORY_CPU_PINNED,
            0 /* memory_type_id */, tensor_memory_type, tensor_memory_type_id,
            pending_pinned_byte_size_, pinned_memory,
            tensor_buffer + pending_pinned_offset_, stream_, &cuda_used);
        cuda_copy |= cuda_used;

        // If something goes wrong with the copy all the pending
        // responses fail...
        if (err != nullptr) {
          for (auto& pr : pending_pinned_inputs_) {
            auto& response = pr.first;
            if (*response != nullptr) {
              LOG_IF_ERROR(
                  TRITONBACKEND_ResponseSend(
                      *response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
                  "failed to send error response");
              *response = nullptr;
            }
          }
          TRITONSERVER_ErrorDelete(err);
        }
      } else {  // cuda_used
        deferred_pinned_.emplace_back(
            pinned_memory, pending_pinned_byte_size_, tensor_buffer,
            pending_pinned_offset_, tensor_memory_type, tensor_memory_type_id,
            std::move(pending_pinned_inputs_));
      }
    } else {
      async_task_count_++;
      deferred_pinned_.emplace_back(
          pinned_memory, pending_pinned_byte_size_, tensor_buffer,
          pending_pinned_offset_, tensor_memory_type, tensor_memory_type_id,
          std::move(pending_pinned_inputs_));
      auto& deferred_pinned = deferred_pinned_.back();
      // Mark finalized to avoid duplicated call to DeferredPinned::Finalized()
      // in BackendInputCollector::Finalize()
      deferred_pinned_.back().finalized_ = true;
      auto incomplete_count = new std::atomic<size_t>(std::min(
          deferred_pinned_.back().requests_.size(),
          triton::common::AsyncWorkQueue::WorkerCount()));
      auto pending_pinned_byte_size = pending_pinned_byte_size_;
      size_t stride = (deferred_pinned_.back().requests_.size() +
                       triton::common::AsyncWorkQueue::WorkerCount() - 1) /
                      triton::common::AsyncWorkQueue::WorkerCount();
      auto pending_it = deferred_pinned_.back().requests_.begin();
      while (pending_it != deferred_pinned_.back().requests_.end()) {
        auto end_it = pending_it;
        auto next_offset = offset;
        for (size_t idx = 0; idx < stride; idx++) {
          uint64_t byte_size;
          RESPOND_AND_SET_NULL_IF_ERROR(
              (*end_it).first,
              TRITONBACKEND_InputPropertiesForHostPolicy(
                  (*end_it).second, host_policy_cstr_, nullptr, nullptr,
                  nullptr, nullptr, &byte_size, nullptr));

          next_offset += byte_size;
          end_it++;
          if (end_it == deferred_pinned_.back().requests_.end()) {
            break;
          }
        }

        auto err =
            CommonErrorToTritonError(triton::common::AsyncWorkQueue::AddTask(
                [this, offset, pinned_memory, pinned_memory_type,
                 pending_pinned_byte_size, pinned_memory_type_id, pending_it,
                 end_it, incomplete_count, &deferred_pinned]() mutable {
                  for (; pending_it != end_it; pending_it++) {
                    TRITONBACKEND_Response** response = (*pending_it).first;
                    TRITONBACKEND_Input* request_input = (*pending_it).second;
                    SetFixedSizeInputTensor(
                        request_input, offset, pinned_memory,
                        pending_pinned_byte_size, pinned_memory_type,
                        pinned_memory_type_id, TRITONSERVER_MEMORY_CPU_PINNED,
                        false, false, response);
                    uint64_t byte_size;
                    RESPOND_AND_SET_NULL_IF_ERROR(
                        response,
                        TRITONBACKEND_InputPropertiesForHostPolicy(
                            request_input, host_policy_cstr_, nullptr, nullptr,
                            nullptr, nullptr, &byte_size, nullptr));

                    offset += byte_size;
                  }
                  // The last segmented task will start the next phase of
                  // the internal pinned buffer copy
                  if (incomplete_count->fetch_sub(1) == 1) {
#ifdef TRITON_ENABLE_GPU
                    if (buffer_ready_event_ != nullptr) {
                      cudaEventSynchronize(buffer_ready_event_);
                      buffer_ready_event_ = nullptr;
                    }
#endif  // TRITON_ENABLE_GPU
                    completion_queue_.Put(deferred_pinned.Finalize(stream_));
                    delete incomplete_count;
                  }
                }));
        if (err != nullptr) {
          for (; pending_it != end_it; pending_it++) {
            TRITONBACKEND_Response** response = (*pending_it).first;
            if (*response != nullptr) {
              LOG_IF_ERROR(
                  TRITONBACKEND_ResponseSend(
                      *response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
                  "failed to send error response");
            }
          }
        }
        TRITONSERVER_ErrorDelete(err);

        offset = next_offset;
        pending_it = end_it;
      }
    }
  }

  // Pending pinned copies are handled...
  pending_pinned_byte_size_ = 0;
  pending_pinned_offset_ = 0;
  pending_pinned_inputs_.clear();

  // Need to hold on to the allocated pinned buffer as there are still
  // copies in flight. Will delete it in finalize.
  if (pinned_memory != nullptr) {
    in_use_memories_.emplace_back(backend_memory);
  }

  return cuda_copy;
}

TRITONSERVER_Error*
BackendInputCollector::BatchInputShape(
    const BatchInput& batch_input, std::vector<int64_t>* shape)
{
  *shape = std::vector<int64_t>{0};
  switch (batch_input.BatchInputKind()) {
    case BatchInput::Kind::BATCH_ELEMENT_COUNT:
    case BatchInput::Kind::BATCH_ACCUMULATED_ELEMENT_COUNT: {
      (*shape)[0] = request_count_;
      break;
    }
    case BatchInput::Kind::BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO: {
      (*shape)[0] = request_count_ + 1;
      break;
    }
    case BatchInput::Kind::BATCH_MAX_ELEMENT_COUNT_AS_SHAPE: {
      const auto& source_input = batch_input.SourceInputs()[0];
      for (size_t req_idx = 0; req_idx < request_count_; req_idx++) {
        TRITONBACKEND_Input* input;
        RETURN_IF_ERROR(TRITONBACKEND_RequestInput(
            requests_[req_idx], source_input.c_str(), &input));
        const int64_t* shape_arr;
        uint32_t dims_count;
        RETURN_IF_ERROR(TRITONBACKEND_InputPropertiesForHostPolicy(
            input, host_policy_cstr_, nullptr, nullptr, &shape_arr, &dims_count,
            nullptr, nullptr));
        (*shape)[0] =
            std::max((*shape)[0], GetElementCount(shape_arr, dims_count));
      }
      break;
    }
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
BackendInputCollector::ProcessBatchInput(
    const BatchInput& batch_input, char* buffer, const size_t buffer_byte_size,
    const std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>>&
        allowed_input_types,
    const char** dst_buffer, size_t* dst_buffer_byte_size,
    TRITONSERVER_MemoryType* dst_memory_type, int64_t* dst_memory_type_id)
{
#ifdef TRITON_ENABLE_GPU
  if (buffer_ready_event_ != nullptr) {
    cudaEventSynchronize(buffer_ready_event_);
    buffer_ready_event_ = nullptr;
  }
#endif  // TRITON_ENABLE_GPU
  if (buffer == nullptr) {
    if (allowed_input_types.size() == 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "'allowed_input_types' must contain at least one pair of memory type "
          "and id");
    }
    // Calculate the byte size of the buffer
    std::vector<int64_t> shape;
    RETURN_IF_ERROR(BatchInputShape(batch_input, &shape));
    *dst_buffer_byte_size = GetByteSize(batch_input.DataType(), shape);
    BackendMemory* backend_memory = nullptr;
    for (const auto& allowed_type : allowed_input_types) {
      std::vector<BackendMemory::AllocationType> alloc_types;
      const int64_t memory_type_id = allowed_type.second;
      switch (allowed_type.first) {
        case TRITONSERVER_MEMORY_GPU:
          alloc_types = {BackendMemory::AllocationType::GPU_POOL,
                         BackendMemory::AllocationType::GPU};
          break;
        case TRITONSERVER_MEMORY_CPU_PINNED:
          alloc_types = {BackendMemory::AllocationType::CPU_PINNED_POOL,
                         BackendMemory::AllocationType::CPU_PINNED};
          break;
        case TRITONSERVER_MEMORY_CPU:
          alloc_types = {BackendMemory::AllocationType::CPU};
          break;
      }
      auto err = BackendMemory::Create(
          memory_manager_, alloc_types, memory_type_id, *dst_buffer_byte_size,
          &backend_memory);
      if (err != nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("unable to create backend memory for type: ") +
             TRITONSERVER_MemoryTypeString(allowed_type.first) +
             " id: " + std::to_string(memory_type_id) + ": " +
             TRITONSERVER_ErrorMessage(err))
                .c_str());
        TRITONSERVER_ErrorDelete(err);
      } else {
        in_use_memories_.emplace_back(backend_memory);
        break;
      }
    }
    if (backend_memory == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string(
               "failed to allocate contiguous buffer for batch input '") +
           batch_input.TargetNames()[0] + "'")
              .c_str());
    }
    buffer = backend_memory->MemoryPtr();
    *dst_buffer = backend_memory->MemoryPtr();
    *dst_buffer_byte_size = backend_memory->ByteSize();
    *dst_memory_type = backend_memory->MemoryType();
    *dst_memory_type_id = backend_memory->MemoryTypeId();
  } else {
    if (allowed_input_types.size() != 1) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "'allowed_input_types' must only contain the memory type and id of "
          "'buffer'");
    }
    *dst_buffer = buffer;
    *dst_buffer_byte_size = buffer_byte_size;
    *dst_memory_type = allowed_input_types[0].first;
    *dst_memory_type_id = allowed_input_types[0].second;
  }

  char* input_buffer = buffer;
  std::unique_ptr<BackendMemory> internal_buffer;
  // Need a CPU buffer for modifying the value
  if (*dst_memory_type == TRITONSERVER_MEMORY_GPU) {
    BackendMemory* ib = nullptr;
    RETURN_IF_ERROR(BackendMemory::Create(
        memory_manager_,
        {BackendMemory::AllocationType::CPU_PINNED_POOL,
         BackendMemory::AllocationType::CPU},
        0, *dst_buffer_byte_size, &ib));
    internal_buffer.reset(ib);
    input_buffer = internal_buffer->MemoryPtr();
  }
  const auto& data_type = batch_input.DataType();
  switch (batch_input.BatchInputKind()) {
    case BatchInput::Kind::BATCH_ELEMENT_COUNT: {
      const auto& source_input = batch_input.SourceInputs()[0];
      if (data_type == TRITONSERVER_TYPE_FP32) {
        SetElementCount<float>(
            source_input, input_buffer, *dst_buffer_byte_size);
      } else {
        SetElementCount<int32_t>(
            source_input, input_buffer, *dst_buffer_byte_size);
      }
      break;
    }
    case BatchInput::Kind::BATCH_ACCUMULATED_ELEMENT_COUNT: {
      const auto& source_input = batch_input.SourceInputs()[0];
      if (data_type == TRITONSERVER_TYPE_FP32) {
        SetAccumulatedElementCount<float>(
            source_input, input_buffer, *dst_buffer_byte_size);
      } else {
        SetAccumulatedElementCount<int32_t>(
            source_input, input_buffer, *dst_buffer_byte_size);
      }
      break;
    }
    case BatchInput::Kind::BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO: {
      const auto& source_input = batch_input.SourceInputs()[0];
      if (data_type == TRITONSERVER_TYPE_FP32) {
        *reinterpret_cast<float*>(input_buffer) = 0;
        SetAccumulatedElementCount<float>(
            source_input, input_buffer + sizeof(float),
            *dst_buffer_byte_size - sizeof(float));
      } else {
        *reinterpret_cast<int32_t*>(input_buffer) = 0;
        SetAccumulatedElementCount<int32_t>(
            source_input, input_buffer + sizeof(int32_t),
            *dst_buffer_byte_size - sizeof(int32_t));
      }
      break;
    }
    case BatchInput::Kind::BATCH_MAX_ELEMENT_COUNT_AS_SHAPE:
      // The batch input is described by the shape,
      // no data modification is needed
      return nullptr;  // success
  }
  if (*dst_memory_type == TRITONSERVER_MEMORY_GPU) {
    bool cuda_used;
    RETURN_IF_ERROR(CopyBuffer(
        "batch input buffer", internal_buffer->MemoryType(),
        internal_buffer->MemoryTypeId(), *dst_memory_type, *dst_memory_type_id,
        *dst_buffer_byte_size, input_buffer, buffer, stream_, &cuda_used));
    // Need to keep the backend memory alive in the case of async copy
    in_use_memories_.emplace_back(std::move(internal_buffer));
    need_sync_ |= cuda_used;
  }
  return nullptr;  // success
}

template <typename T>
TRITONSERVER_Error*
BackendInputCollector::SetElementCount(
    const std::string& source_input, char* buffer,
    const size_t buffer_byte_size)
{
  size_t buffer_offset = 0;
  for (size_t req_idx = 0; req_idx < request_count_; req_idx++) {
    if (buffer_offset + sizeof(T) > buffer_byte_size) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "unexpected total byte size for batch input");
    }

    TRITONBACKEND_Input* input;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInput(
        requests_[req_idx], source_input.c_str(), &input));
    const int64_t* shape;
    uint32_t dims_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputPropertiesForHostPolicy(
        input, host_policy_cstr_, nullptr, nullptr, &shape, &dims_count,
        nullptr, nullptr));
    *(reinterpret_cast<T*>(buffer) + req_idx) =
        GetElementCount(shape, dims_count);
    buffer_offset += sizeof(T);
  }
  // Set the rest of the buffer to 0
  for (; buffer_offset + sizeof(T) <= buffer_byte_size;
       buffer_offset += sizeof(T)) {
    *reinterpret_cast<T*>(buffer + buffer_offset) = 0;
  }
  return nullptr;  // success
}

template <typename T>
TRITONSERVER_Error*
BackendInputCollector::SetAccumulatedElementCount(
    const std::string& source_input, char* buffer,
    const size_t buffer_byte_size)
{
  size_t accumulated_element_count = 0;
  size_t buffer_offset = 0;
  for (size_t req_idx = 0; req_idx < request_count_; req_idx++) {
    if (buffer_offset + sizeof(T) > buffer_byte_size) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "unexpected total byte size for batch input");
    }

    TRITONBACKEND_Input* input;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInput(
        requests_[req_idx], source_input.c_str(), &input));
    const int64_t* shape;
    uint32_t dims_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputPropertiesForHostPolicy(
        input, host_policy_cstr_, nullptr, nullptr, &shape, &dims_count,
        nullptr, nullptr));
    accumulated_element_count += GetElementCount(shape, dims_count);
    *(reinterpret_cast<T*>(buffer) + req_idx) = accumulated_element_count;
    buffer_offset += sizeof(T);
  }
  // Set the rest of the buffer to 'accumulated_element_count'
  // (no increase in element count)
  for (; buffer_offset + sizeof(T) <= buffer_byte_size;
       buffer_offset += sizeof(T)) {
    *reinterpret_cast<T*>(buffer + buffer_offset) = accumulated_element_count;
  }
  return nullptr;  // success
}

bool
BackendInputCollector::FlushPendingCopyKernel(
    char* tensor_buffer, const size_t tensor_buffer_byte_size,
    const TRITONSERVER_MemoryType tensor_memory_type,
    const int64_t tensor_memory_type_id)
{
  if (pending_copy_kernel_inputs_.size() == 0) {
    return false;
  }

  bool cuda_copy = false;
  TRITONSERVER_Error* error = nullptr;
  // Only try to launch kernel if buffer count is large enough for
  // good GPU utilization
  if (pending_copy_kernel_input_buffer_counts_ >= kernel_buffer_threshold_) {
    error = LaunchCopyKernel(
        tensor_buffer, tensor_buffer_byte_size, tensor_memory_type,
        tensor_memory_type_id);
    cuda_copy = (error == nullptr);
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("gather kernel launched with status: ") +
         TRITONSERVER_ErrorMessage(error))
            .c_str());
  }
  // If kernel can't be launched then just perform a direct copy.
  if (error == nullptr) {
    size_t offset = 0;
    for (auto& pr : pending_copy_kernel_inputs_) {
      TRITONBACKEND_Response** response = pr.first;
      TRITONBACKEND_Input* request_input = pr.second;

      uint64_t byte_size;
      RESPOND_AND_SET_NULL_IF_ERROR(
          response, TRITONBACKEND_InputPropertiesForHostPolicy(
                        request_input, host_policy_cstr_, nullptr, nullptr,
                        nullptr, nullptr, &byte_size, nullptr));

      cuda_copy |= SetFixedSizeInputTensor(
          request_input, pending_copy_kernel_buffer_offset_ + offset,
          tensor_buffer, tensor_buffer_byte_size, tensor_memory_type,
          tensor_memory_type_id, TRITONSERVER_MEMORY_CPU_PINNED, false, true,
          response);
      offset += byte_size;
    }
  }

  // Pending kernel copies are handled...
  pending_copy_kernel_buffer_byte_size_ = 0;
  pending_copy_kernel_buffer_offset_ = 0;
  pending_copy_kernel_input_buffer_counts_ = 0;
  pending_copy_kernel_inputs_.clear();

  return cuda_copy;
}

TRITONSERVER_Error*
BackendInputCollector::LaunchCopyKernel(
    char* tensor_buffer, const size_t tensor_buffer_byte_size,
    const TRITONSERVER_MemoryType tensor_memory_type,
    const int64_t tensor_memory_type_id)
{
#ifdef TRITON_ENABLE_GPU
  input_ptr_buffer_host_.emplace_back(new std::vector<int8_t*>());
  byte_size_buffer_host_.emplace_back(new std::vector<size_t>());
  byte_size_offset_buffer_host_.emplace_back(new std::vector<size_t>());

  auto& input_ptr_buffer_host = *input_ptr_buffer_host_.back();
  auto& byte_size_buffer_host = *byte_size_buffer_host_.back();
  auto& byte_size_offset_buffer_host = *byte_size_offset_buffer_host_.back();

  input_ptr_buffer_host.reserve(pending_copy_kernel_input_buffer_counts_);
  byte_size_buffer_host.reserve(pending_copy_kernel_input_buffer_counts_);
  byte_size_offset_buffer_host.reserve(
      pending_copy_kernel_input_buffer_counts_);

  // placeholder for output parameters
  auto kernel_buffer_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t kernel_buffer_memory_id = 0;
  size_t buffer_byte_size = 0;

  size_t byte_size_offset = 0;
  for (const auto& response_input : pending_copy_kernel_inputs_) {
    auto input = response_input.second;
    uint32_t buffer_count;
    RESPOND_AND_SET_NULL_IF_ERROR(
        response_input.first, TRITONBACKEND_InputPropertiesForHostPolicy(
                                  input, host_policy_cstr_, nullptr, nullptr,
                                  nullptr, nullptr, nullptr, &buffer_count));
    for (size_t buffer_idx = 0; buffer_idx < buffer_count; ++buffer_idx) {
      input_ptr_buffer_host.emplace_back();
      RESPOND_AND_SET_NULL_IF_ERROR(
          response_input.first,
          TRITONBACKEND_InputBufferForHostPolicy(
              input, host_policy_cstr_, buffer_idx,
              (const void**)(&input_ptr_buffer_host.back()), &buffer_byte_size,
              &kernel_buffer_memory_type, &kernel_buffer_memory_id));

      byte_size_offset_buffer_host.emplace_back(byte_size_offset);
      byte_size_buffer_host.emplace_back(buffer_byte_size);
      byte_size_offset += buffer_byte_size;
    }
  }

  BackendMemory* backend_memory = nullptr;
  std::vector<BackendMemory::AllocationType> alloc_types;
  switch (tensor_memory_type) {
    case TRITONSERVER_MEMORY_GPU:
      alloc_types = {BackendMemory::AllocationType::GPU_POOL,
                     BackendMemory::AllocationType::GPU};
      break;
    case TRITONSERVER_MEMORY_CPU_PINNED:
      alloc_types = {BackendMemory::AllocationType::CPU_PINNED_POOL,
                     BackendMemory::AllocationType::CPU_PINNED};
      break;
    case TRITONSERVER_MEMORY_CPU:
      alloc_types = {BackendMemory::AllocationType::CPU};
      break;
  }

  // input_ptr_buffer
  size_t input_ptr_buffer_byte_size =
      pending_copy_kernel_input_buffer_counts_ * sizeof(int8_t*);
  auto err = BackendMemory::Create(
      memory_manager_, alloc_types, tensor_memory_type_id,
      input_ptr_buffer_byte_size, &backend_memory);
  if (err != nullptr) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("unable to create backend memory for type: ") +
         TRITONSERVER_MemoryTypeString(tensor_memory_type) +
         " id: " + std::to_string(tensor_memory_type_id) + ": " +
         TRITONSERVER_ErrorMessage(err))
            .c_str());
    TRITONSERVER_ErrorDelete(err);
  } else {
    in_use_memories_.emplace_back(backend_memory);
  }
  if (backend_memory == nullptr ||
      (backend_memory->MemoryType() != tensor_memory_type) ||
      (backend_memory->MemoryTypeId() != tensor_memory_type_id)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Failed to obtain memory buffer for copy kernel input");
  }
  char* input_ptr_buffer = backend_memory->MemoryPtr();

  // byte_size_buffer
  size_t byte_size_buffer_byte_size =
      pending_copy_kernel_input_buffer_counts_ * sizeof(size_t);
  err = BackendMemory::Create(
      memory_manager_, alloc_types, tensor_memory_type_id,
      byte_size_buffer_byte_size, &backend_memory);
  if (err != nullptr) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("unable to create backend memory for type: ") +
         TRITONSERVER_MemoryTypeString(tensor_memory_type) +
         " id: " + std::to_string(tensor_memory_type_id) + ": " +
         TRITONSERVER_ErrorMessage(err))
            .c_str());
    TRITONSERVER_ErrorDelete(err);
  } else {
    in_use_memories_.emplace_back(backend_memory);
  }
  if (backend_memory == nullptr ||
      (backend_memory->MemoryType() != tensor_memory_type) ||
      (backend_memory->MemoryTypeId() != tensor_memory_type_id)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Failed to obtain memory buffer for copy kernel input");
  }
  char* byte_size_buffer = backend_memory->MemoryPtr();

  // byte_size_offset_buffer
  size_t byte_size_offset_buffer_byte_size =
      pending_copy_kernel_input_buffer_counts_ * sizeof(size_t);
  err = BackendMemory::Create(
      memory_manager_, alloc_types, tensor_memory_type_id,
      byte_size_offset_buffer_byte_size, &backend_memory);
  if (err != nullptr) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("unable to create backend memory for type: ") +
         TRITONSERVER_MemoryTypeString(tensor_memory_type) +
         " id: " + std::to_string(tensor_memory_type_id) + ": " +
         TRITONSERVER_ErrorMessage(err))
            .c_str());
    TRITONSERVER_ErrorDelete(err);
  } else {
    in_use_memories_.emplace_back(backend_memory);
  }
  if (backend_memory == nullptr ||
      (backend_memory->MemoryType() != tensor_memory_type) ||
      (backend_memory->MemoryTypeId() != tensor_memory_type_id)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Failed to obtain memory buffer for copy kernel input");
  }
  char* byte_size_offset_buffer = backend_memory->MemoryPtr();

  cudaMemcpyAsync(
      input_ptr_buffer, input_ptr_buffer_host.data(),
      pending_copy_kernel_input_buffer_counts_ * sizeof(int8_t*),
      cudaMemcpyDefault, stream_);
  cudaMemcpyAsync(
      byte_size_buffer, byte_size_buffer_host.data(),
      pending_copy_kernel_input_buffer_counts_ * sizeof(size_t),
      cudaMemcpyDefault, stream_);
  cudaMemcpyAsync(
      byte_size_offset_buffer, byte_size_offset_buffer_host.data(),
      pending_copy_kernel_input_buffer_counts_ * sizeof(size_t),
      cudaMemcpyDefault, stream_);
  if (buffer_ready_event_ != nullptr) {
    cudaEventSynchronize(buffer_ready_event_);
    buffer_ready_event_ = nullptr;
  }
  RETURN_IF_CUDA_ERROR(
      RunGatherKernel(
          (const int8_t**)input_ptr_buffer, (const size_t*)byte_size_buffer,
          (const size_t*)byte_size_offset_buffer,
          (int8_t*)tensor_buffer + pending_copy_kernel_buffer_offset_,
          pending_copy_kernel_input_buffer_counts_, stream_),
      TRITONSERVER_ERROR_INTERNAL,
      std::string("Failed to launch gather kernel"));
  return nullptr;
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED,
      "Copy kernel can not be launched with TRITON_ENABLE_GPU=OFF");
#endif  // TRITON_ENABLE_GPU
}

}}  // namespace triton::backend
