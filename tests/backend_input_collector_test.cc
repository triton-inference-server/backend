// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::atomic<size_t> gEventRecords{0};
std::promise<void>* gReleaseOnRecord = nullptr;
void* gPinnedPool = nullptr;

void
Check(bool condition, const char* message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void
CheckCuda(cudaError_t error)
{
  Check(error == cudaSuccess, cudaGetErrorString(error));
}

struct Buffer {
  explicit Buffer(size_t bytes, bool pinned = false) : pinned_(pinned)
  {
    CheckCuda(
        pinned ? cudaMallocHost(&data_, bytes) : cudaMalloc(&data_, bytes));
  }
  ~Buffer() { pinned_ ? cudaFreeHost(data_) : cudaFree(data_); }
  void* data_ = nullptr;
  bool pinned_;
};

struct Stream {
  Stream()
  {
    CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }
  ~Stream() { cudaStreamDestroy(stream_); }
  cudaStream_t stream_ = nullptr;
};

struct Event {
  Event()
  {
    CheckCuda(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }
  ~Event() { cudaEventDestroy(event_); }
  cudaEvent_t event_ = nullptr;
};

struct Input {
  const void* data_;
  uint64_t bytes_;
  TRITONSERVER_MemoryType memory_type_;
};

struct Request {
  std::unordered_map<std::string, Input> inputs_;
};

// Exercise the public collector API using request/allocator test doubles.
// Copy submission, pinned staging and CUDA event semantics use the real code.
}  // namespace

extern "C" cudaError_t __real_cudaEventRecord(cudaEvent_t, cudaStream_t);
extern "C" cudaError_t
__wrap_cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
  gEventRecords.fetch_add(1);
  if (gReleaseOnRecord != nullptr) {
    auto* release = gReleaseOnRecord;
    gReleaseOnRecord = nullptr;
    release->set_value();
  }
  return __real_cudaEventRecord(event, stream);
}

extern "C" TRITONSERVER_Error*
TRITONBACKEND_RequestInput(
    TRITONBACKEND_Request* request, const char* name,
    TRITONBACKEND_Input** input)
{
  auto& inputs = reinterpret_cast<Request*>(request)->inputs_;
  auto found = inputs.find(name);
  if (found == inputs.end()) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_NOT_FOUND, name);
  }
  *input = reinterpret_cast<TRITONBACKEND_Input*>(&found->second);
  return nullptr;
}

extern "C" TRITONSERVER_Error*
TRITONBACKEND_InputPropertiesForHostPolicy(
    TRITONBACKEND_Input* input, const char*, const char**,
    TRITONSERVER_DataType*, const int64_t**, uint32_t*, uint64_t* bytes,
    uint32_t* count)
{
  auto& value = *reinterpret_cast<Input*>(input);
  if (bytes != nullptr) {
    *bytes = value.bytes_;
  }
  if (count != nullptr) {
    *count = 1;
  }
  return nullptr;
}

extern "C" TRITONSERVER_Error*
TRITONBACKEND_InputBufferForHostPolicy(
    TRITONBACKEND_Input* input, const char*, const uint32_t index,
    const void** data, uint64_t* bytes, TRITONSERVER_MemoryType* memory_type,
    int64_t* memory_type_id)
{
  if (index != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "buffer index");
  }
  auto& value = *reinterpret_cast<Input*>(input);
  *data = value.data_;
  *bytes = value.bytes_;
  *memory_type = value.memory_type_;
  *memory_type_id = 0;
  return nullptr;
}

extern "C" TRITONSERVER_Error*
TRITONBACKEND_MemoryManagerAllocate(
    TRITONBACKEND_MemoryManager*, void** buffer,
    const TRITONSERVER_MemoryType memory_type, const int64_t,
    const uint64_t bytes)
{
  if ((memory_type != TRITONSERVER_MEMORY_CPU_PINNED) ||
      (gPinnedPool == nullptr) || (bytes > sizeof(uint32_t))) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED, "unexpected allocation");
  }
  *buffer = gPinnedPool;
  return nullptr;
}

extern "C" TRITONSERVER_Error*
TRITONBACKEND_MemoryManagerFree(
    TRITONBACKEND_MemoryManager*, void*, const TRITONSERVER_MemoryType,
    const int64_t)
{
  return nullptr;
}

namespace {

// Independent input tensors must share one completion record, and that record
// must cover their copies even when the event is reused for another request.
void
CoalescesInputCompletion()
{
  constexpr size_t kINPUT_COUNT = 100;
  Buffer source(kINPUT_COUNT * sizeof(uint32_t), true);
  Buffer destination(kINPUT_COUNT * sizeof(uint32_t));
  Stream stream;
  Event event;
  auto* values = static_cast<uint32_t*>(source.data_);
  for (uint32_t iteration = 1; iteration <= 2; ++iteration) {
    Request request;
    for (size_t i = 0; i < kINPUT_COUNT; ++i) {
      values[i] = iteration * 1000 + i;
      request.inputs_.emplace(
          std::to_string(i),
          Input{values + i, sizeof(uint32_t), TRITONSERVER_MEMORY_CPU_PINNED});
    }
    auto* input_request = reinterpret_cast<TRITONBACKEND_Request*>(&request);
    std::vector<TRITONBACKEND_Response*> responses{
        reinterpret_cast<TRITONBACKEND_Response*>(&request)};
    triton::backend::BackendInputCollector collector(
        &input_request, 1, &responses, nullptr, false, stream.stream_,
        event.event_);
    gEventRecords.store(0);
    for (size_t i = 0; i < kINPUT_COUNT; ++i) {
      collector.ProcessTensor(
          std::to_string(i).c_str(),
          static_cast<char*>(destination.data_) + i * sizeof(uint32_t),
          sizeof(uint32_t), TRITONSERVER_MEMORY_GPU, 0);
    }
    const size_t records_before_finalize = gEventRecords.load();
    const bool need_sync = collector.Finalize();
    CheckCuda(cudaEventSynchronize(event.event_));
    std::vector<uint32_t> actual(kINPUT_COUNT);
    CheckCuda(cudaMemcpy(
        actual.data(), destination.data_, actual.size() * sizeof(uint32_t),
        cudaMemcpyDeviceToHost));
    Check(need_sync, "asynchronous input copies must require completion");
    Check(records_before_finalize == 0, "input records must be coalesced");
    Check(gEventRecords.load() == 1, "one record must cover all input copies");
    Check(
        std::equal(actual.begin(), actual.end(), values),
        "the completion event must cover every input tensor");
  }
}

// Host-only inputs must remain synchronous and must not record a CUDA event.
void
CollectsHostInputsWithoutEvents()
{
  uint32_t expected = 73;
  uint32_t actual = 0;
  Request request;
  request.inputs_.emplace(
      "input", Input{&expected, sizeof(expected), TRITONSERVER_MEMORY_CPU});
  auto* input_request = reinterpret_cast<TRITONBACKEND_Request*>(&request);
  std::vector<TRITONBACKEND_Response*> responses{
      reinterpret_cast<TRITONBACKEND_Response*>(&request)};
  Event event;
  triton::backend::BackendInputCollector collector(
      &input_request, 1, &responses, nullptr, false, nullptr, event.event_);
  gEventRecords.store(0);
  collector.ProcessTensor(
      "input", reinterpret_cast<char*>(&actual), sizeof(actual),
      TRITONSERVER_MEMORY_CPU, 0);
  Check(!collector.Finalize(), "host copies must not require synchronization");
  Check(actual == expected, "host input contents changed");
  Check(gEventRecords.load() == 0, "host-only inputs must not record an event");
}

// GPU-to-pinned-to-CPU collection must wait for the staging copy before reading
// it. The queued copy is held until an event is recorded, making a missing
// pre-synchronization record expose stale staging data deterministically.
void
FinalizesDeferredCpuCopy(bool use_event)
{
  constexpr uint32_t kVALUE = 0x12345678;
  Buffer source(sizeof(kVALUE));
  Buffer staging(sizeof(kVALUE), true);
  CheckCuda(cudaMemcpy(
      source.data_, &kVALUE, sizeof(kVALUE), cudaMemcpyHostToDevice));
  std::memset(staging.data_, 0, sizeof(kVALUE));
  Stream stream;
  Event event;
  std::promise<void> release;
  auto released = release.get_future().share();
  if (use_event) {
    CheckCuda(cudaLaunchHostFunc(
        stream.stream_,
        [](void* state) {
          static_cast<std::shared_future<void>*>(state)->wait();
        },
        &released));
    gReleaseOnRecord = &release;
  }
  Request request;
  request.inputs_.emplace(
      "input", Input{source.data_, sizeof(kVALUE), TRITONSERVER_MEMORY_GPU});
  auto* input_request = reinterpret_cast<TRITONBACKEND_Request*>(&request);
  std::vector<TRITONBACKEND_Response*> responses{
      reinterpret_cast<TRITONBACKEND_Response*>(&request)};
  uint32_t actual = 0;
  gPinnedPool = staging.data_;
  gEventRecords.store(0);
  {
    triton::backend::BackendInputCollector collector(
        &input_request, 1, &responses, nullptr, true, stream.stream_,
        use_event ? event.event_ : nullptr);
    collector.ProcessTensor(
        "input", reinterpret_cast<char*>(&actual), sizeof(actual),
        TRITONSERVER_MEMORY_CPU, 0);
    const bool need_sync = collector.Finalize();
    // Release the copy if a broken implementation omitted the readiness record.
    if (gReleaseOnRecord != nullptr) {
      gReleaseOnRecord = nullptr;
      release.set_value();
    }
    CheckCuda(cudaStreamSynchronize(stream.stream_));
    Check(!need_sync, "the deferred CPU destination must be ready on return");
    Check(
        actual == kVALUE,
        "deferred CPU copy read staging data before readiness");
    Check(
        gEventRecords.load() == (use_event ? 1 : 0),
        "deferred CPU copy must retain its required completion checkpoint");
  }
  gPinnedPool = nullptr;
}

}  // namespace

int
main()
{
  try {
    CoalescesInputCompletion();
    CollectsHostInputsWithoutEvents();
    FinalizesDeferredCpuCopy(true);
    FinalizesDeferredCpuCopy(false);
    std::cout << "PASS input coalescing, event reuse, deferred CPU copies, and "
                 "stream synchronization fallback\n";
    return 0;
  }
  catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
