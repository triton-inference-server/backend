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

#include "triton/backend/backend_model.h"

#include "triton/backend/backend_common.h"

namespace triton { namespace backend {

//
// BackendModel
//
BackendModel::BackendModel(
    TRITONBACKEND_Model* triton_model, const bool allow_optional)
    : triton_model_(triton_model), allow_optional_(allow_optional)
{
  const char* model_name;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelName(triton_model, &model_name));
  name_ = model_name;

  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelVersion(triton_model, &version_));

  const char* repository_path = nullptr;
  TRITONBACKEND_ArtifactType repository_artifact_type;
  THROW_IF_BACKEND_MODEL_ERROR(TRITONBACKEND_ModelRepository(
      triton_model, &repository_artifact_type, &repository_path));
  if (repository_artifact_type != TRITONBACKEND_ARTIFACT_FILESYSTEM) {
    throw BackendModelException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("unsupported repository artifact type for model '") +
         model_name + "'")
            .c_str()));
  }
  repository_path_ = repository_path;

  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelServer(triton_model, &triton_server_));
  TRITONBACKEND_Backend* backend;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelBackend(triton_model, &backend));
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_BackendMemoryManager(backend, &triton_memory_manager_));

  THROW_IF_BACKEND_MODEL_ERROR(ParseModelConfig());
}

TRITONSERVER_Error*
BackendModel::ParseModelConfig()
{
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model_, 1 /* config_version */, &config_message));

  // Get the model configuration as a json string from
  // config_message. We use TritonJson, which is a wrapper that
  // returns nice errors (currently the underlying implementation is
  // rapidjson... but others could be added).
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  TRITONSERVER_Error* err = model_config_.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  int64_t mbs = 0;
  RETURN_IF_ERROR(model_config_.MemberAsInt("max_batch_size", &mbs));
  max_batch_size_ = mbs;

  enable_pinned_input_ = false;
  enable_pinned_output_ = false;
  {
    common::TritonJson::Value optimization;
    if (model_config_.Find("optimization", &optimization)) {
      common::TritonJson::Value pinned_memory;
      if (optimization.Find("input_pinned_memory", &pinned_memory)) {
        RETURN_IF_ERROR(
            pinned_memory.MemberAsBool("enable", &enable_pinned_input_));
      }
      if (optimization.Find("output_pinned_memory", &pinned_memory)) {
        RETURN_IF_ERROR(
            pinned_memory.MemberAsBool("enable", &enable_pinned_output_));
      }
    }
  }

  RETURN_IF_ERROR(
      BatchInput::ParseFromModelConfig(model_config_, &batch_inputs_));
  RETURN_IF_ERROR(
      BatchOutput::ParseFromModelConfig(model_config_, &batch_outputs_));
  for (const auto& batch_output : batch_outputs_) {
    for (const auto& name : batch_output.TargetNames()) {
      batch_output_map_.emplace(name, &batch_output);
    }
  }
  triton::common::TritonJson::Value config_inputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &config_inputs));
  for (size_t i = 0; i < config_inputs.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(config_inputs.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    triton::common::TritonJson::Value input_property_json;
    bool allow_ragged_batch = false;
    if (io.Find("allow_ragged_batch", &input_property_json)) {
      RETURN_IF_ERROR(input_property_json.AsBool(&allow_ragged_batch));
    }
    if (allow_ragged_batch) {
      ragged_inputs_.emplace(io_name);
    }
    bool optional = false;
    if (io.Find("optional", &input_property_json)) {
      RETURN_IF_ERROR(input_property_json.AsBool(&optional));
    }
    if (optional) {
      if (allow_optional_) {
        optional_inputs_.emplace(io_name);
      } else {
        RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("'optional' is set to true for input '") + io_name +
             "' while the backend model doesn't support optional input")
                .c_str()));
      }
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
BackendModel::SetModelConfig()
{
  triton::common::TritonJson::WriteBuffer json_buffer;
  RETURN_IF_ERROR(ModelConfig().Write(&json_buffer));

  TRITONSERVER_Message* message;
  RETURN_IF_ERROR(TRITONSERVER_MessageNewFromSerializedJson(
      &message, json_buffer.Base(), json_buffer.Size()));
  RETURN_IF_ERROR(TRITONBACKEND_ModelSetConfig(
      triton_model_, 1 /* config_version */, message));
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(message));

  // Triton core can normalize the missing config settings
  // in the above call. We must retrieve the updated model
  // configuration from the core.
  RETURN_IF_ERROR(ParseModelConfig());

  return nullptr;
}

TRITONSERVER_Error*
BackendModel::SupportsFirstDimBatching(bool* supports)
{
  *supports = max_batch_size_ > 0;
  return nullptr;
}

const BatchOutput*
BackendModel::FindBatchOutput(const std::string& output_name) const
{
  const auto it = batch_output_map_.find(output_name);
  return ((it == batch_output_map_.end()) ? nullptr : it->second);
}

}}  // namespace triton::backend
