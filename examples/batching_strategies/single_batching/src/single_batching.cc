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

#include <iostream>
#include "triton/core/tritonbackend.h"

namespace triton { namespace core { namespace single_batching {

//
// Minimal custom  batching strategy that demonstrates the
// TRITONBACKEND_ModelBatch API. This custom batching strategy dynamically
// creates batches up to 1 request.
//

/////////////

extern "C" {

/// Check whether a request should be added to the pending model batch.
/// \param model The backend model for which Triton is forming a batch.
/// \param request The request to be added to the pending batch.
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch. When the callback returns, this should reflect
/// the latest batch information.
/// \param should_include The pointer to be updated on whether the request was
/// included in the batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatchIncludeRequest(
    TRITONBACKEND_Model* model, TRITONBACKEND_Request* request, void* userp,
    bool* should_include)
{
  // Check if the batch is empty.
  // If so, include this request. Otherwise, do not.
  bool* empty = static_cast<bool*>(userp);
  if (*empty) {
    *should_include = true;
    *empty = false;
  } else {
    *should_include = false;
  }

  return nullptr;  // success
}

/// Callback to be invoked when Triton has begun forming a batch.
/// \param model The backend model for which Triton is forming a batch.
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatchInitialize(
    TRITONBACKEND_Model* model, void** userp, void** cache_userp)
{
  // Userp will point to a boolean indicating whether the batch is empty.
  *userp = new bool(true);
  return nullptr;  // success
}

/// Callback to be invoked when Triton has completed forming a batch.
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatchFinalize(void* userp)
{
  delete static_cast<bool*>(userp);
  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::core::single_batching
