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

#include "bls.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

//
// Backend that demonstrates using in-process C-API to execute inferences
// within the backend.
//
// Two particular models, 'addsub_python' and 'addsub_tf', must be loaded on
// the server for a successful inference execution on this backend.
//
// The model configuration should be set as follows in order to be in line with
// the 'addsub_python' and 'addsub_tf' models. This backend does not support
// batching. These limitations are only for this specific backend. You can
// implement your custom BLS backend with less limitations.
//
// Model Configuration:
//   - Input 'INPUT0' must have shape [16] and datatype must be TYPE_FP32.
//
//   - Input 'INPUT1' must have shape [16] and datatype must be TYPE_FP32.
//
//   - For each response, output 'OUTPUT0' must have shape [16] and
//     datatype TYPE_FP32.
//
//   - For each response, output 'OUTPUT1' must have shape [16] and
//     datatype TYPE_FP32.
//
// This backend will send two requests on the 'addsub_python' and 'addsub_tf'
// models. After the inference requests are completed, this backend
// will extract OUTPUT0 from the 'addsub_python' and OUTPUT1 from the
// 'addsub_tf' model to construct the final inference response object using
// these tensors.

namespace triton { namespace backend { namespace bls {

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

 private:
  ModelState(TRITONBACKEND_Model* triton_model) : BackendModel(triton_model) {}
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // We have the json DOM for the model configuration...
  common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  // max_batch_size must be 0 because this backend does not support
  // batching
  int64_t max_batch_size;
  RETURN_IF_ERROR(model_config_.MemberAsInt("max_batch_size", &max_batch_size));
  RETURN_ERROR_IF_FALSE(
      max_batch_size == 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("bls backend only supports models with max_batch_size == 0"));

  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));

  // There must be 2 inputs and 2 outputs.
  RETURN_ERROR_IF_FALSE(
      inputs.ArraySize() == 2, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected 2 inputs, got ") +
          std::to_string(inputs.ArraySize()));
  RETURN_ERROR_IF_FALSE(
      outputs.ArraySize() == 2, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected 2 outputs, got ") +
          std::to_string(outputs.ArraySize()));

  // Here we rely on the model configuation listing the inputs and
  // outputs in a specific order, which we shouldn't really require...
  common::TritonJson::Value input0, input1, output0, output1;
  RETURN_IF_ERROR(inputs.IndexAsObject(0, &input0));
  RETURN_IF_ERROR(inputs.IndexAsObject(1, &input1));
  RETURN_IF_ERROR(outputs.IndexAsObject(0, &output0));
  RETURN_IF_ERROR(outputs.IndexAsObject(1, &output1));

  // Check tensor names
  std::string in0_name, in1_name, out0_name, out1_name;
  RETURN_IF_ERROR(input0.MemberAsString("name", &in0_name));
  RETURN_IF_ERROR(input1.MemberAsString("name", &in1_name));
  RETURN_IF_ERROR(output0.MemberAsString("name", &out0_name));
  RETURN_IF_ERROR(output1.MemberAsString("name", &out1_name));

  RETURN_ERROR_IF_FALSE(
      in0_name == "INPUT0", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected first input tensor name to be INPUT0, got ") +
          in0_name);
  RETURN_ERROR_IF_FALSE(
      in1_name == "INPUT1", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected second input tensor name to be INPUT1, got ") +
          in1_name);
  RETURN_ERROR_IF_FALSE(
      out0_name == "OUTPUT0", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected first output tensor name to be OUTPUT0, got ") +
          out0_name);
  RETURN_ERROR_IF_FALSE(
      out1_name == "OUTPUT1", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected second output tensor name to be OUTPUT1, got ") +
          out1_name);

  // Check shapes
  std::vector<int64_t> in0_shape, in1_shape, out0_shape, out1_shape;
  RETURN_IF_ERROR(backend::ParseShape(input0, "dims", &in0_shape));
  RETURN_IF_ERROR(backend::ParseShape(input1, "dims", &in1_shape));
  RETURN_IF_ERROR(backend::ParseShape(output0, "dims", &out0_shape));
  RETURN_IF_ERROR(backend::ParseShape(output1, "dims", &out1_shape));

  RETURN_ERROR_IF_FALSE(
      in0_shape.size() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected INPUT0 shape to have one dimension, got ") +
          backend::ShapeToString(in0_shape));
  RETURN_ERROR_IF_FALSE(
      in1_shape.size() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected INPUT1 shape to have one dimension, got ") +
          backend::ShapeToString(in1_shape));
  RETURN_ERROR_IF_FALSE(
      out0_shape.size() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected OUTPUT0 shape to have one dimension, got ") +
          backend::ShapeToString(out0_shape));
  RETURN_ERROR_IF_FALSE(
      out1_shape.size() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected OUTPUT1 shape to have one dimension, got ") +
          backend::ShapeToString(out1_shape));

  // Check datatypes
  std::string in0_dtype, in1_dtype, out0_dtype, out1_dtype;
  RETURN_IF_ERROR(input0.MemberAsString("data_type", &in0_dtype));
  RETURN_IF_ERROR(input1.MemberAsString("data_type", &in1_dtype));
  RETURN_IF_ERROR(output0.MemberAsString("data_type", &out0_dtype));
  RETURN_IF_ERROR(output1.MemberAsString("data_type", &out1_dtype));

  RETURN_ERROR_IF_FALSE(
      in0_dtype == "TYPE_FP32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected INPUT0 datatype to be TYPE_FP32, got ") +
          in0_dtype);
  RETURN_ERROR_IF_FALSE(
      in1_dtype == "TYPE_FP32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected INPUT1 datatype to be TYPE_FP32, got ") +
          in1_dtype);
  RETURN_ERROR_IF_FALSE(
      out0_dtype == "TYPE_FP32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected OUTPUT0 datatype to be TYPE_FP32, got ") +
          out0_dtype);
  RETURN_ERROR_IF_FALSE(
      out1_dtype == "TYPE_FP32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected OUTPUT1 datatype to be TYPE_FP32, got ") +
          out1_dtype);

  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance)
  {
  }
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to BLS backend for '" + Name() + "'")
                  .c_str()));
      return;
    }
  }

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  ModelState* model_state = reinterpret_cast<ModelState*>(Model());

  // The way we collect these batch timestamps is not entirely
  // accurate. Normally, in a performant backend you would execute all
  // the requests at the same time, and so there would be a single
  // compute-start / compute-end time-range. But here we execute each
  // request separately so there is no single range. As a result we
  // just show the entire execute time as being the compute time as
  // well.
  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  // Create a BLSExecutor object. To separate from standard backend
  // implementation, the BLS logic is placed inside class BLSExecutor.
  BLSExecutor bls_executor(model_state->TritonServer());

  for (size_t r = 0; r < request_count; r++) {
    bls_executor.Execute(requests[r], &responses[r]);
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // we successful or not.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send BLS backend response");
    }
  }

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // Report the entire batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          TritonModelInstance(), 1 /*total_batch_size*/, exec_start_ns,
          compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: model ") + Name() +
       " released " + std::to_string(request_count) + " requests")
          .c_str());
}

/////////////

extern "C" {

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ValidateModelConfig());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: instance "
                   "initialization successful ") +
       name + " (device " + std::to_string(device_id) + ")")
          .c_str());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state =
      reinterpret_cast<ModelState*>(instance_state->Model());

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::bls
