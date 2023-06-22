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

#include <map>
#include <set>
#include <string>

#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend {

//
// BackendModel
//
// Common functionality for a backend model. This class is provided as
// a convenience; backends are not required to use this class.
//
class BackendModel {
 public:
  BackendModel(
      TRITONBACKEND_Model* triton_model, const bool allow_optional = false);
  virtual ~BackendModel() = default;

  // Get the handle to the TRITONBACKEND server hosting this model.
  TRITONSERVER_Server* TritonServer() { return triton_server_; }

  // Get the handle to the memory manager for this model.
  TRITONBACKEND_MemoryManager* TritonMemoryManager()
  {
    return triton_memory_manager_;
  }

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Model* TritonModel() { return triton_model_; }

  // Get the name and version of the model.
  const std::string& Name() const { return name_; }
  uint64_t Version() const { return version_; }
  const std::string& RepositoryPath() const { return repository_path_; }

  // The model configuration.
  common::TritonJson::Value& ModelConfig() { return model_config_; }

  // Sets the updated model configuration to the core.
  TRITONSERVER_Error* SetModelConfig();

  // Parses information out of the model configuration.
  TRITONSERVER_Error* ParseModelConfig();

  // Maximum batch size supported by the model. A value of 0
  // indicates that the model does not support batching.
  int MaxBatchSize() const { return max_batch_size_; }

  // Set the max batch size for the model. When a backend
  // auto-completes a configuration it may set or change the maximum
  // batch size.
  void SetMaxBatchSize(const int b) { max_batch_size_ = b; }

  // Does this model support batching in the first dimension?
  TRITONSERVER_Error* SupportsFirstDimBatching(bool* supports);

  // Use indirect pinned memory buffer when copying an input or output
  // tensor to/from the model.
  bool EnablePinnedInput() const { return enable_pinned_input_; }
  bool EnablePinnedOutput() const { return enable_pinned_output_; }

  const std::vector<BatchInput>& BatchInputs() const { return batch_inputs_; }
  const std::vector<BatchOutput>& BatchOutputs() const
  {
    return batch_outputs_;
  }
  const BatchOutput* FindBatchOutput(const std::string& output_name) const;
  bool IsInputRagged(const std::string& input_name) const
  {
    return (ragged_inputs_.find(input_name) != ragged_inputs_.end());
  }
  bool IsInputOptional(const std::string& input_name) const
  {
    return (optional_inputs_.find(input_name) != optional_inputs_.end());
  }

 protected:
  TRITONSERVER_Server* triton_server_;
  TRITONBACKEND_MemoryManager* triton_memory_manager_;
  TRITONBACKEND_Model* triton_model_;
  std::string name_;
  uint64_t version_;
  std::string repository_path_;
  bool allow_optional_;

  common::TritonJson::Value model_config_;
  int max_batch_size_;
  bool enable_pinned_input_;
  bool enable_pinned_output_;
  std::vector<BatchInput> batch_inputs_;
  std::vector<BatchOutput> batch_outputs_;
  std::map<std::string, const BatchOutput*> batch_output_map_;
  std::set<std::string> ragged_inputs_;
  std::set<std::string> optional_inputs_;
};

//
// BackendModelException
//
// Exception thrown if error occurs while constructing an
// BackendModel.
//
struct BackendModelException {
  BackendModelException(TRITONSERVER_Error* err) : err_(err) {}
  TRITONSERVER_Error* err_;
};

#define THROW_IF_BACKEND_MODEL_ERROR(X)                        \
  do {                                                         \
    TRITONSERVER_Error* tie_err__ = (X);                       \
    if (tie_err__ != nullptr) {                                \
      throw triton::backend::BackendModelException(tie_err__); \
    }                                                          \
  } while (false)

}}  // namespace triton::backend
