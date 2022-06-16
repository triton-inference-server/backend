// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "bls_utils.h"

namespace triton { namespace backend { namespace bls {

TRITONSERVER_Error*
CPUAllocator(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  // For simplifying this example, this backend always uses CPU memory
  // regardless of the preferred memory type. You can make the actual memory
  // type and id that we allocate be the same as preferred memory type. You can
  // also provide a customized allocator to support different
  // preferred_memory_type, and reuse memory buffer when possible.
  *actual_memory_type = TRITONSERVER_MEMORY_CPU;
  *actual_memory_type_id = preferred_memory_type_id;

  // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
  // need to do any other book-keeping.
  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
    std::cout << "allocated " << byte_size << " bytes for result tensor "
              << tensor_name << std::endl;
  } else {
    void* allocated_ptr = nullptr;
    *actual_memory_type = TRITONSERVER_MEMORY_CPU;
    allocated_ptr = malloc(byte_size);

    // Pass the tensor name with buffer_userp so we can show it when
    // releasing the buffer.
    if (allocated_ptr != nullptr) {
      *buffer = allocated_ptr;
      *buffer_userp = new std::string(tensor_name);
      std::cout << "allocated " << byte_size << " bytes in "
                << TRITONSERVER_MemoryTypeString(*actual_memory_type)
                << " for result tensor " << tensor_name << std::endl;
    }
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  std::string* name = nullptr;
  if (buffer_userp != nullptr) {
    name = reinterpret_cast<std::string*>(buffer_userp);
  } else {
    name = new std::string("<unknown>");
  }

  std::cout << "Releasing buffer " << buffer << " of size " << byte_size
            << " in " << TRITONSERVER_MemoryTypeString(memory_type)
            << " for result '" << *name << "'" << std::endl;
  switch (memory_type) {
    case TRITONSERVER_MEMORY_CPU:
      free(buffer);
      break;
    default:
      std::cerr << "error: unexpected buffer allocated in CUDA managed memory"
                << std::endl;
      break;
  }

  delete name;

  return nullptr;  // Success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  if (request != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_InferenceRequestDelete(request),
        "Failed to delete inference request.");
  }
}

void
InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  // The following logic only works for non-decoupled models as for decoupled
  // models it may send multiple responses for a request or not send any
  // responses for a request. Need to modify this function if the model is using
  // decoupled API.
  if (response != nullptr) {
    // Send 'response' to the future.
    std::promise<TRITONSERVER_InferenceResponse*>* p =
        reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
    p->set_value(response);
    delete p;
  }
}

TRITONSERVER_Error*
PrepareInferenceRequest(
    TRITONSERVER_Server* server, TRITONBACKEND_Request* bls_request,
    TRITONSERVER_InferenceRequest** irequest, const std::string model_name)
{
  // Get request_id, correlation_id, and flags from the current request
  // for preparing a new inference request that we will send to 'addsub_python'
  // or 'addsub_tf' model later.
  const char* request_id;
  uint64_t correlation_id;
  uint32_t flags;
  RETURN_IF_ERROR(TRITONBACKEND_RequestId(bls_request, &request_id));
  RETURN_IF_ERROR(
      TRITONBACKEND_RequestCorrelationId(bls_request, &correlation_id));
  RETURN_IF_ERROR(TRITONBACKEND_RequestFlags(bls_request, &flags));

  // Create an inference request object. The inference request object
  // is where we set the name of the model we want to use for
  // inference and the input tensors.
  RETURN_IF_ERROR(TRITONSERVER_InferenceRequestNew(
      irequest, server, model_name.c_str(), 1 /* model_version */));
  // Set request_id, correlation_id, and flags for the new request.
  RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetId(*irequest, request_id));
  RETURN_IF_ERROR(
      TRITONSERVER_InferenceRequestSetCorrelationId(*irequest, correlation_id));
  RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetFlags(*irequest, flags));
  RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetReleaseCallback(
      *irequest, InferRequestComplete, nullptr /* request_release_userp */));

  return nullptr;  // success
}

TRITONSERVER_Error*
PrepareInferenceInput(
    TRITONBACKEND_Request* bls_request,
    TRITONSERVER_InferenceRequest** irequest)
{
  // Get the properties of the two inputs from the current request.
  // Then, add the two input tensors and append the input data to the new
  // request.
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(bls_request, &input_count));

  TRITONBACKEND_Input* input;
  const char* name;
  TRITONSERVER_DataType datatype;
  const int64_t* shape;
  uint32_t dims_count;
  size_t data_byte_size;
  TRITONSERVER_MemoryType data_memory_type;
  int64_t data_memory_id;
  const char* data_buffer;

  for (size_t count = 0; count < input_count; count++) {
    RETURN_IF_ERROR(TRITONBACKEND_RequestInputByIndex(
        bls_request, count /* index */, &input));
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &name, &datatype, &shape, &dims_count, nullptr, nullptr));
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        input, 0 /* idx */, reinterpret_cast<const void**>(&data_buffer),
        &data_byte_size, &data_memory_type, &data_memory_id));
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAddInput(
        *irequest, name, datatype, shape, dims_count));
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAppendInputData(
        *irequest, name, &data_buffer[0], data_byte_size, data_memory_type,
        data_memory_id));
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
PrepareInferenceOutput(
    TRITONBACKEND_Request* bls_request,
    TRITONSERVER_InferenceRequest** irequest)
{
  // Indicate the output tensors to be calculated and returned
  // for the inference request.
  uint32_t output_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestOutputCount(bls_request, &output_count));
  const char* output_name;
  for (size_t count = 0; count < output_count; count++) {
    RETURN_IF_ERROR(TRITONBACKEND_RequestOutputName(
        bls_request, count /* index */, &output_name));
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAddRequestedOutput(
        *irequest, output_name));
  }

  return nullptr;  // success
}

void
ConstructFinalResponse(
    TRITONBACKEND_Response** response,
    std::vector<std::future<TRITONSERVER_InferenceResponse*>> futures)
{
  // Prepare two TRITONSERVER_InferenceResponse* objects for 'addsub_python' and
  // 'addsub_tf' repectively.
  std::vector<TRITONSERVER_InferenceResponse*> completed_responses = {nullptr,
                                                                      nullptr};

  const char* output_name;
  TRITONSERVER_DataType output_datatype;
  const int64_t* output_shape;
  uint64_t dims_count;
  size_t output_byte_size;
  TRITONSERVER_MemoryType output_memory_type;
  int64_t output_memory_id;
  const void* output_base;
  void* userp;
  for (size_t icount = 0; icount < 2; icount++) {
    try {
      // Retrieve the corresponding TRITONSERVER_InferenceResponse object from
      // 'futures'. The InferResponseComplete function sets the std::promise
      // so that this thread will block until the response is returned.
      completed_responses[icount] = futures[icount].get();
      THROW_IF_TRITON_ERROR(
          TRITONSERVER_InferenceResponseError(completed_responses[icount]));

      // Retrieve outputs from 'completed_responses'.
      // Extract OUTPUT0 from the 'addsub_python' and OUTPUT1 from the
      // 'addsub_tf' model to form the final inference response object.
      // Get all the information about the output tensor.
      RESPOND_AND_SET_NULL_IF_ERROR(
          response, TRITONSERVER_InferenceResponseOutput(
                        completed_responses[icount], icount, &output_name,
                        &output_datatype, &output_shape, &dims_count,
                        &output_base, &output_byte_size, &output_memory_type,
                        &output_memory_id, &userp));

      // Create an output tensor in the final response with
      // the information retrieved above.
      TRITONBACKEND_Output* output;
      RESPOND_AND_SET_NULL_IF_ERROR(
          response, TRITONBACKEND_ResponseOutput(
                        *response, &output, output_name, output_datatype,
                        output_shape, dims_count));

      // Get a buffer that holds the tensor data for the output.
      // We request a buffer in CPU memory but we have to handle any returned
      // type. If we get back a buffer in GPU memory we just fail the request.
      void* output_buffer;
      output_memory_type = TRITONSERVER_MEMORY_CPU;
      RESPOND_AND_SET_NULL_IF_ERROR(
          response, TRITONBACKEND_OutputBuffer(
                        output, &output_buffer, output_byte_size,
                        &output_memory_type, &output_memory_id));
      if (output_memory_type == TRITONSERVER_MEMORY_GPU) {
        RESPOND_AND_SET_NULL_IF_ERROR(
            response, TRITONSERVER_ErrorNew(
                          TRITONSERVER_ERROR_INTERNAL,
                          "failed to create output buffer in CPU memory"));
      }

      // Fill the BLS output buffer with output data returned by internal
      // requests.
      memcpy(output_buffer, output_base, output_byte_size);

      LOG_IF_ERROR(
          TRITONSERVER_InferenceResponseDelete(completed_responses[icount]),
          "Failed to delete inference response.");
    }
    catch (const BLSBackendException& bls_exception) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, bls_exception.what());

      if (completed_responses[icount] != nullptr) {
        LOG_IF_ERROR(
            TRITONSERVER_InferenceResponseDelete(completed_responses[icount]),
            "Failed to delete inference response.");
      }
    }
  }
}

BLSExecutor::BLSExecutor(TRITONSERVER_Server* server) : server_(server)
{
  // When triton needs a buffer to hold an output tensor, it will ask
  // us to provide the buffer. In this way we can have any buffer
  // management and sharing strategy that we want. To communicate to
  // triton the functions that we want it to call to perform the
  // allocations, we create a "response allocator" object. We pass
  // this response allocate object to triton when requesting
  // inference. We can reuse this response allocate object for any
  // number of inference requests.
  allocator_ = nullptr;
  THROW_IF_TRITON_ERROR(TRITONSERVER_ResponseAllocatorNew(
      &allocator_, CPUAllocator, ResponseRelease, nullptr /* start_fn */));
}

TRITONSERVER_Error*
BLSExecutor::Execute(
    TRITONSERVER_InferenceRequest* irequest,
    std::future<TRITONSERVER_InferenceResponse*>* future)
{
  // Perform inference by calling TRITONSERVER_ServerInferAsync. This
  // call is asychronous and therefore returns immediately. The
  // completion of the inference and delivery of the response is done
  // by triton by calling the "response complete" callback functions
  // (InferResponseComplete in this case).
  auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
  *future = p->get_future();

  RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetResponseCallback(
      irequest, allocator_, nullptr /* response_allocator_userp */,
      InferResponseComplete, reinterpret_cast<void*>(p)));

  RETURN_IF_ERROR(
      TRITONSERVER_ServerInferAsync(server_, irequest, nullptr /* trace */));

  return nullptr;  // success
}

}}}  // namespace triton::backend::bls
