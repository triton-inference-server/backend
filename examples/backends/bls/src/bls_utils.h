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

#include <future>
#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace bls {

#define THROW_IF_TRITON_ERROR(X)                                       \
  do {                                                                 \
    TRITONSERVER_Error* tie_err__ = (X);                               \
    if (tie_err__ != nullptr) {                                        \
      throw BLSBackendException(TRITONSERVER_ErrorMessage(tie_err__)); \
    }                                                                  \
  } while (false)

//
// BLSBackendException
//
// Exception thrown if error occurs in BLSBackend.
//
struct BLSBackendException : std::exception {
  BLSBackendException(const std::string& message) : message_(message) {}

  const char* what() const throw() { return message_.c_str(); }

  std::string message_;
};

//
// ModelExecutor
//
// Execute inference request on a model.
//
class ModelExecutor {
 public:
  ModelExecutor(){};

  // Performs async inference request.
  TRITONSERVER_Error* Execute(
      TRITONSERVER_Server* server, TRITONSERVER_ResponseAllocator* allocator,
      TRITONSERVER_InferenceRequest* irequest,
      std::future<TRITONSERVER_InferenceResponse*>* future);
};

//
// BLSExecutor
//
// Includes the custom BLS logic for this backend.
//
class BLSExecutor {
 public:
  BLSExecutor(TRITONSERVER_Server* server);

  // Prepares the inference request that will be used internally.
  TRITONSERVER_Error* PrepareInferenceRequest(
      TRITONBACKEND_Request* bls_request,
      TRITONSERVER_InferenceRequest** irequest, const std::string model_name);

  // Prepares the input for the internal inference request.
  TRITONSERVER_Error* PrepareInferenceInput(
      TRITONBACKEND_Request* bls_request,
      TRITONSERVER_InferenceRequest* irequest);

  // Prepares the output for the internal inference request.
  TRITONSERVER_Error* PrepareInferenceOutput(
      TRITONBACKEND_Request* bls_request,
      TRITONSERVER_InferenceRequest* irequest);

  // Performs the whole BLS pipeline.
  void Execute(
      TRITONBACKEND_Request* bls_request, TRITONBACKEND_Response** response);

  // Constructs the final response.
  void ConstructFinalResponse(
      TRITONBACKEND_Response** response,
      std::vector<std::future<TRITONSERVER_InferenceResponse*>> futures);

 private:
  // The server object that encapsulates all the functionality of the Triton
  // server and allows access to the Triton server API.
  TRITONSERVER_Server* server_;

  // The allocator object that will be used for allocation.
  TRITONSERVER_ResponseAllocator* allocator_;
};

}}}  // namespace triton::backend::bls
