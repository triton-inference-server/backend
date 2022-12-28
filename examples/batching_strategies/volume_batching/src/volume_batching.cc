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

#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr
#include "triton/common/triton_json.h"

namespace triton { namespace core { namespace volume_batching {

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
  // Default should_include to false in case function returns error.
  *should_include = false;

  // Get current remaining batch volume.
  unsigned int* remaining_volume = static_cast<unsigned int*>(userp);

  // Get request's volume in bytes.
  unsigned int pending_volume = 0;

  uint32_t input_count;
  auto err = TRITONBACKEND_RequestInputCount(request, &input_count);
  if (err)
    return err;

  TRITONBACKEND_Input* input;
  size_t data_byte_size;

  for (size_t count = 0; count < input_count; count++) {
    auto err =
        TRITONBACKEND_RequestInputByIndex(request, count /* index */, &input);
    if (err)
      return err;
    err = TRITONBACKEND_InputProperties(
        input, nullptr, nullptr, nullptr, nullptr, &data_byte_size, nullptr);
    if (err)
      return err;
    pending_volume += static_cast<unsigned int>(data_byte_size);
  }

  // Print remaining volume for debugging purposes.
  std::cout << "Pending volume : " << pending_volume << std::endl;
  std::cout << "Remaining volume : " << *remaining_volume << std::endl;

  // Check if there is enough remaining volume for this request.
  // If so, include this request. Otherwise, do not.
  if (pending_volume <= *remaining_volume) {
    *should_include = true;
    *remaining_volume = *remaining_volume - pending_volume;
  } else {
    *should_include = false;
  }

  return nullptr;  // success
}

/// Callback to be invoked when Triton has begun forming a batch.
/// \param model The backend model for which Triton is forming a batch.
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch.
/// \param cache_userp The read-only placeholder for backend to retrieve
// information about the batching strategy for this model.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatchInitialize(
    TRITONBACKEND_Model* model, void** userp, void** cache_userp)
{
  // Userp will point to an unsigned integer representing the remaining volume
  // in bytes for this batch.

  *userp = new unsigned int(*static_cast<unsigned int*>(*cache_userp));
  return nullptr;  // success
}

/// Callback to be invoked when Triton has finished forming a batch.
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatchFinalize(void* userp)
{
  delete static_cast<unsigned int*>(userp);
  return nullptr;  // success
}

/// Callback to be invoked when Triton loads model.
/// This will hold a cached user pointer that can be read during custom
/// batching. \param model The backend model for which Triton is forming a
/// batch. \param cache_userp The placeholder for backend to store and retrieve
/// information about the batching strategy for this model. \return a
/// TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatchCacheInitialize(
    TRITONBACKEND_Model* model, void** cache_userp)
{
  // Cache_userp will point to an unsigned integer representing the maximum
  // volume in bytes for each batch.

  // Read the user-specified bytes from the model config.
  TRITONSERVER_Message* config_message;
  TRITONBACKEND_ModelConfig(model, 1 /* config_version */, &config_message);

  const char* buffer;
  size_t byte_size;

  uint64_t max_volume_bytes = 0;
  std::string max_volume_bytes_str;

  auto err =
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size);
  if (err)
    return err;

  triton::common::TritonJson::Value model_config, params, volume_param;
  err = model_config.Parse(buffer, byte_size);
  TRITONSERVER_MessageDelete(config_message);

  if (!model_config.Find("parameters", &params)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        "Unable to find parameters in model config");
  }

  std::vector<std::string> param_keys;

  if (!params.Find("MAX_BATCH_VOLUME_BYTES", &volume_param)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        "Unable to find MAX_BATCH_VOLUME_BYTES parameter in model config");
  }
  err = volume_param.MemberAsString("string_value", &max_volume_bytes_str);
  if (err)
    return err;

  try {
    max_volume_bytes = static_cast<uint64_t>(std::stoul(max_volume_bytes_str));
  }
  catch (const std::invalid_argument& ia) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("failed to convert '") + max_volume_bytes_str +
         "' to unsigned int64")
            .c_str());
  }

  *cache_userp = new unsigned int(max_volume_bytes);
}

/// Callback to be invoked when Triton unloads model.
/// \param cache_userp The placeholder for backend to store and retrieve
/// information about the batching strategy for this model.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatchCacheFinalize(void* cache_userp)
{
  delete static_cast<unsigned int*>(cache_userp);
}

}  // extern "C"

}}}  // namespace triton::core::volume_batching
