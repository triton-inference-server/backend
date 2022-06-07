// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "triton/common/error.h"
#include "triton/core/tritonbackend.h"

#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr
#include "triton/common/triton_json.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace backend {

#define IGNORE_ERROR(X)                   \
  do {                                    \
    TRITONSERVER_Error* ie_err__ = (X);   \
    if (ie_err__ != nullptr) {            \
      TRITONSERVER_ErrorDelete(ie_err__); \
    }                                     \
  } while (false)

#define LOG_IF_ERROR(X, MSG)                                                   \
  do {                                                                         \
    TRITONSERVER_Error* lie_err__ = (X);                                       \
    if (lie_err__ != nullptr) {                                                \
      IGNORE_ERROR(TRITONSERVER_LogMessage(                                    \
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,                           \
          (std::string(MSG) + ": " + TRITONSERVER_ErrorCodeString(lie_err__) + \
           " - " + TRITONSERVER_ErrorMessage(lie_err__))                       \
              .c_str()));                                                      \
      TRITONSERVER_ErrorDelete(lie_err__);                                     \
    }                                                                          \
  } while (false)

#define LOG_MESSAGE(LEVEL, MSG)                                  \
  do {                                                           \
    LOG_IF_ERROR(                                                \
        TRITONSERVER_LogMessage(LEVEL, __FILE__, __LINE__, MSG), \
        ("failed to log message: "));                            \
  } while (false)


#define RETURN_ERROR_IF_FALSE(P, C, MSG)              \
  do {                                                \
    if (!(P)) {                                       \
      return TRITONSERVER_ErrorNew(C, (MSG).c_str()); \
    }                                                 \
  } while (false)

#define RETURN_ERROR_IF_TRUE(P, C, MSG)               \
  do {                                                \
    if ((P)) {                                        \
      return TRITONSERVER_ErrorNew(C, (MSG).c_str()); \
    }                                                 \
  } while (false)

#define RETURN_IF_ERROR(X)               \
  do {                                   \
    TRITONSERVER_Error* rie_err__ = (X); \
    if (rie_err__ != nullptr) {          \
      return rie_err__;                  \
    }                                    \
  } while (false)

#ifdef TRITON_ENABLE_GPU
#define LOG_IF_CUDA_ERROR(X, MSG)                                    \
  do {                                                               \
    cudaError_t lice_err__ = (X);                                    \
    if (lice_err__ != cudaSuccess) {                                 \
      IGNORE_ERROR(TRITONSERVER_LogMessage(                          \
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,                 \
          (std::string(MSG) + ": " + cudaGetErrorString(lice_err__)) \
              .c_str()));                                            \
    }                                                                \
  } while (false)

#define RETURN_IF_CUDA_ERROR(X, C, MSG)                                \
  do {                                                                 \
    cudaError_t rice_err__ = (X);                                      \
    if (rice_err__ != cudaSuccess) {                                   \
      return TRITONSERVER_ErrorNew(                                    \
          C, ((MSG) + ": " + cudaGetErrorString(rice_err__)).c_str()); \
    }                                                                  \
  } while (false)
#endif  // TRITON_ENABLE_GPU

#define RESPOND_AND_SET_NULL_IF_ERROR(RESPONSE_PTR, X)               \
  do {                                                               \
    TRITONSERVER_Error* rarie_err__ = (X);                           \
    if (rarie_err__ != nullptr) {                                    \
      if (*RESPONSE_PTR != nullptr) {                                \
        LOG_IF_ERROR(                                                \
            TRITONBACKEND_ResponseSend(                              \
                *RESPONSE_PTR, TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                rarie_err__),                                        \
            "failed to send error response");                        \
        *RESPONSE_PTR = nullptr;                                     \
      }                                                              \
      TRITONSERVER_ErrorDelete(rarie_err__);                         \
    }                                                                \
  } while (false)

#define RESPOND_ALL_AND_SET_NULL_IF_ERROR(RESPONSES, RESPONSES_COUNT, X) \
  do {                                                                   \
    TRITONSERVER_Error* raasnie_err__ = (X);                             \
    if (raasnie_err__ != nullptr) {                                      \
      for (size_t ridx = 0; ridx < RESPONSES_COUNT; ++ridx) {            \
        if (RESPONSES[ridx] != nullptr) {                                \
          LOG_IF_ERROR(                                                  \
              TRITONBACKEND_ResponseSend(                                \
                  RESPONSES[ridx], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                  raasnie_err__),                                        \
              "failed to send error response");                          \
          RESPONSES[ridx] = nullptr;                                     \
        }                                                                \
      }                                                                  \
      TRITONSERVER_ErrorDelete(raasnie_err__);                           \
    }                                                                    \
  } while (false)

#define RESPOND_ALL_AND_SET_TRUE_IF_ERROR(RESPONSES, RESPONSES_COUNT, BOOL, X) \
  do {                                                                         \
    TRITONSERVER_Error* raasnie_err__ = (X);                                   \
    if (raasnie_err__ != nullptr) {                                            \
      BOOL = true;                                                             \
      for (size_t ridx = 0; ridx < RESPONSES_COUNT; ++ridx) {                  \
        if (RESPONSES[ridx] != nullptr) {                                      \
          LOG_IF_ERROR(                                                        \
              TRITONBACKEND_ResponseSend(                                      \
                  RESPONSES[ridx], TRITONSERVER_RESPONSE_COMPLETE_FINAL,       \
                  raasnie_err__),                                              \
              "failed to send error response");                                \
          RESPONSES[ridx] = nullptr;                                           \
        }                                                                      \
      }                                                                        \
      TRITONSERVER_ErrorDelete(raasnie_err__);                                 \
    }                                                                          \
  } while (false)

#ifdef TRITON_ENABLE_STATS
#define TIMESPEC_TO_NANOS(TS) ((TS).tv_sec * 1000000000 + (TS).tv_nsec)
#define SET_TIMESTAMP(TS_NS)                                         \
  {                                                                  \
    TS_NS = std::chrono::duration_cast<std::chrono::nanoseconds>(    \
                std::chrono::steady_clock::now().time_since_epoch()) \
                .count();                                            \
  }
#define DECL_TIMESTAMP(TS_NS) \
  uint64_t TS_NS;             \
  SET_TIMESTAMP(TS_NS);
#else
#define DECL_TIMESTAMP(TS_NS)
#define SET_TIMESTAMP(TS_NS)
#endif  // TRITON_ENABLE_STATS

#ifndef TRITON_ENABLE_GPU
using cudaStream_t = void*;
#endif  // !TRITON_ENABLE_GPU

/// Convenience deleter for TRITONBACKEND_ResponseFactory.
struct ResponseFactoryDeleter {
  void operator()(TRITONBACKEND_ResponseFactory* f)
  {
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseFactoryDelete(f),
        "failed deleting response factory");
  }
};

// A representation of the BatchInput message in model config
class BatchInput {
 public:
  enum class Kind {
    BATCH_ELEMENT_COUNT,
    BATCH_ACCUMULATED_ELEMENT_COUNT,
    BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO,
    BATCH_MAX_ELEMENT_COUNT_AS_SHAPE,
    BATCH_ITEM_SHAPE,
    BATCH_ITEM_SHAPE_FLATTEN
  };
  static TRITONSERVER_Error* ParseFromModelConfig(
      triton::common::TritonJson::Value& config,
      std::vector<BatchInput>* batch_inputs);
  const std::vector<std::string>& TargetNames() const { return target_names_; }
  TRITONSERVER_DataType DataType() const { return data_type_; }
  Kind BatchInputKind() const { return kind_; }
  std::string BatchInputKindString() const { return kind_str_; }
  const std::vector<std::string>& SourceInputs() const
  {
    return source_inputs_;
  }

 private:
  TRITONSERVER_Error* Init(triton::common::TritonJson::Value& bi_config);
  Kind kind_;
  std::string kind_str_;
  std::vector<std::string> target_names_;
  TRITONSERVER_DataType data_type_;
  std::vector<std::string> source_inputs_;
};

// A representation of the BatchOutput message in model config
class BatchOutput {
 public:
  enum class Kind { BATCH_SCATTER_WITH_INPUT_SHAPE };
  static TRITONSERVER_Error* ParseFromModelConfig(
      triton::common::TritonJson::Value& config,
      std::vector<BatchOutput>* batch_outputs);
  const std::vector<std::string>& TargetNames() const { return target_names_; }
  TRITONSERVER_DataType DataType() const { return data_type_; }
  const std::vector<int64_t>& OutputShape() const { return shape_; }
  Kind BatchOutputKind() const { return kind_; }
  const std::vector<std::string>& SourceInputs() const
  {
    return source_inputs_;
  }

 private:
  Kind kind_;
  std::vector<std::string> target_names_;
  TRITONSERVER_DataType data_type_;
  std::vector<int64_t> shape_;
  std::vector<std::string> source_inputs_;
};

struct CopyParams {
  CopyParams(void* dst, const void* src, const size_t byte_size)
      : dst_(dst), src_(src), byte_size_(byte_size)
  {
  }

  void* dst_;
  const void* src_;
  const size_t byte_size_;
};

/// The value for a dimension in a shape that indicates that that
/// dimension can take on any size.
constexpr int WILDCARD_DIM = -1;

constexpr char kTensorRTExecutionAccelerator[] = "tensorrt";
constexpr char kOpenVINOExecutionAccelerator[] = "openvino";
constexpr char kGPUIOExecutionAccelerator[] = "gpu_io";
constexpr char kAutoMixedPrecisionExecutionAccelerator[] =
    "auto_mixed_precision";

TRITONSERVER_MemoryType GetUsePinnedMemoryType(
    TRITONSERVER_MemoryType ref_buffer_type);

TRITONSERVER_Error* CommonErrorToTritonError(triton::common::Error error);

TRITONSERVER_Error_Code StatusCodeToTritonCode(
    triton::common::Error::Code error_code);

/// Parse an array in a JSON object into the corresponding shape. The
/// array must be composed of integers.
///
/// \param io The JSON object containing the member array.
/// \param name The name of the array member in the JSON object.
/// \param shape Returns the shape.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* ParseShape(
    common::TritonJson::Value& io, const std::string& name,
    std::vector<int64_t>* shape);

/// Return the string representation of a shape.
///
/// \param dims The shape dimensions.
/// \param dims_count The number of dimensions.
/// \return The string representation.
std::string ShapeToString(const int64_t* dims, const size_t dims_count);

/// Return the string representation of a shape.
///
/// \param shape The shape as a vector of dimensions.
/// \return The string representation.
std::string ShapeToString(const std::vector<int64_t>& shape);

/// Return the number of elements of a shape.
///
/// \param dims The shape dimensions.
/// \param dims_count The number of dimensions.
/// \return The number of elements.
int64_t GetElementCount(const int64_t* dims, const size_t dims_count);

/// Return the number of elements of a shape.
///
/// \param shape The shape as a vector of dimensions.
/// \return The number of elements.
int64_t GetElementCount(const std::vector<int64_t>& shape);

/// Get the size, in bytes, of a tensor based on datatype and
/// shape.
/// \param dtype The data-type.
/// \param dims The shape.
/// \return The size, in bytes, of the corresponding tensor, or -1 if
/// unable to determine the size.
int64_t GetByteSize(
    const TRITONSERVER_DataType& dtype, const std::vector<int64_t>& dims);

/// Get an input tensor's contents into a buffer. This overload expects
/// both 'buffer' and buffers of the input to be in CPU.
///
/// \param request The inference request.
/// \param input_name The name of the input buffer.
/// \param buffer The buffer where the input tensor content is copied into.
/// \param buffer_byte_size Acts as both input and output. On input
/// gives the size of 'buffer', in bytes. The function will fail if
/// the buffer is not large enough to hold the input tensor
/// contents. Returns the size of the input tensor data returned in
/// 'buffer'.
/// \param host_policy_name The host policy name to look up the input buffer.
/// Default input buffer will be used if nullptr is provided.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* ReadInputTensor(
    TRITONBACKEND_Request* request, const std::string& input_name, char* buffer,
    size_t* buffer_byte_size, const char* host_policy_name = nullptr);

/// Get an input tensor's contents into a buffer. This overload of
/// 'ReadInputTensor' supports input buffers that can be in any memory.
///
/// \param request The inference request.
/// \param input_name The name of the input buffer.
/// \param buffer The buffer where the input tensor content is copied into.
/// \param buffer_byte_size Acts as both input and output. On input
/// gives the size of 'buffer', in bytes. The function will fail if
/// the buffer is not large enough to hold the input tensor
/// contents. Returns the size of the input tensor data returned in
/// 'buffer'.
/// \param host_policy_name The host policy name to look up the input buffer.
/// Default input buffer will be used if nullptr is provided.
/// \param memory_type The memory type of the buffer provided.
/// \param memory_type_id The memory type id of the buffer provided.
/// \param cuda_stream specifies the stream to be associated with, and 0 can be
/// passed for default stream.
/// \param cuda_used returns whether a CUDA memory copy is initiated. If true,
/// the caller should synchronize on the given 'cuda_stream' to ensure data copy
/// is completed.
/// \param copy_on_stream whether the memory copies should be performed in cuda
/// host functions on the 'cuda_stream'.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* ReadInputTensor(
    TRITONBACKEND_Request* request, const std::string& input_name, char* buffer,
    size_t* buffer_byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, cudaStream_t cuda_stream, bool* cuda_used,
    const char* host_policy_name = nullptr, const bool copy_on_stream = false);

/// Validate that an input matches one of the allowed input names.
/// \param io The model input.
/// \param allowed The set of allowed input names.
/// \return The error status. A non-OK status indicates the input
/// is not valid.
TRITONSERVER_Error* CheckAllowedModelInput(
    common::TritonJson::Value& io, const std::set<std::string>& allowed);

/// Validate that an output matches one of the allowed output names.
/// \param io The model output.
/// \param allowed The set of allowed output names.
/// \return The error status. A non-OK status indicates the output
/// is not valid.
TRITONSERVER_Error* CheckAllowedModelOutput(
    common::TritonJson::Value& io, const std::set<std::string>& allowed);

/// Get the tensor name, false value, and true value for a boolean
/// sequence batcher control kind. If 'required' is true then must
/// find a tensor for the control. If 'required' is false, return
/// 'tensor_name' as empty-string if the control is not mapped to any
/// tensor.
///
/// \param batcher The JSON object of the sequence batcher.
/// \param model_name The name of the model.
/// \param control_kind The kind of control tensor to look for.
/// \param required Whether the tensor must be specified.
/// \param tensor_name Returns the name of the tensor.
/// \param tensor_datatype Returns the data type of the tensor.
/// \param fp32_false_value Returns the float value for false if
/// the tensor type is FP32.
/// \param fp32_true_value Returns the float value for true if
/// the tensor type is FP32.
/// \param int32_false_value Returns the int value for false if
/// the tensor type is INT32.
/// \param int32_true_value Returns the int value for true if
/// the tensor type is INT32.
/// \param bool_false_value Returns the bool value for false if
/// the tensor type is BOOL.
/// \param bool_true_value Returns the bool value for true if
/// the tensor type is BOOL.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* GetBooleanSequenceControlProperties(
    common::TritonJson::Value& batcher, const std::string& model_name,
    const std::string& control_kind, const bool required,
    std::string* tensor_name, std::string* tensor_datatype,
    float* fp32_false_value, float* fp32_true_value, int32_t* int32_false_value,
    int32_t* int32_true_value, bool* bool_false_value, bool* bool_true_value);

/// Get the tensor name and datatype for a non-boolean sequence
/// batcher control kind. If 'required' is true then must find a
/// tensor for the control. If 'required' is false, return
/// 'tensor_name' as empty-string if the control is not mapped to any
/// tensor. 'tensor_datatype' returns the required datatype for the
/// control.
///
/// \param batcher The JSON object of the sequence batcher.
/// \param model_name The name of the model.
/// \param control_kind The kind of control tensor to look for.
/// \param required Whether the tensor must be specified.
/// \param tensor_name Returns the name of the tensor.
/// \param tensor_datatype Returns the data type of the tensor.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* GetTypedSequenceControlProperties(
    common::TritonJson::Value& batcher, const std::string& model_name,
    const std::string& control_kind, const bool required,
    std::string* tensor_name, std::string* tensor_datatype);

/// Create and send an error response for a set of requests. This
/// function takes ownership of 'response_err' and so the caller must
/// not access or delete it after this call returns.
///
/// \param requests The requests.
/// \param request_count The number of 'requests'.
/// \param response_err The error to send to each request.
/// \param release_request If true, the requests will be released after
/// sending the error responses and the request pointers are set to
/// nullptr.
void RequestsRespondWithError(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    TRITONSERVER_Error* response_err, const bool release_request = true);

/// Send an error response for a set of responses. This function takes
/// ownership of 'response_err' and so the caller must not access or
/// delete it after this call returns.
///
/// \param responses The responses.
/// \param response_count The number of 'responses'.
/// \param response_err The error to send.
void SendErrorForResponses(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count, TRITONSERVER_Error* response_err);

/// Copy buffer from 'src' to 'dst' for given 'byte_size'. The buffer location
/// is identified by the memory type and id, and the corresponding copy will be
/// initiated.
/// \param msg The message to be prepended in error message.
/// \param src_memory_type The memory type of the source buffer.
/// \param src_memory_type_id The memory type id of the source buffer.
/// \param dst_memory_type The memory type of the destination buffer.
/// \param dst_memory_type_id The memory type id of the destination buffer.
/// \param byte_size The byte size of the source buffer.
/// \param src The pointer to the source buffer.
/// \param dst The pointer to the destination buffer.
/// \param cuda_stream specifies the stream to be associated with, and 0 can be
/// passed for default stream.
/// \param cuda_used returns whether a CUDA memory copy is initiated. If true,
/// the caller should synchronize on the given 'cuda_stream' to ensure data copy
/// is completed.
/// \param copy_on_stream whether the memory copies should be performed in cuda
/// host functions on the 'cuda_stream'.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* CopyBuffer(
    const std::string& msg, const TRITONSERVER_MemoryType src_memory_type,
    const int64_t src_memory_type_id,
    const TRITONSERVER_MemoryType dst_memory_type,
    const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
    void* dst, cudaStream_t cuda_stream, bool* cuda_used,
    const bool copy_on_stream = false);

/// Does a file or directory exist?
/// \param path The path to check for existance.
/// \param exists Returns true if file/dir exists
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* FileExists(const std::string& path, bool* exists);

/// Read a text file into a string.
/// \param path The path of the file.
/// \param contents Returns the contents of the file.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* ReadTextFile(
    const std::string& path, std::string* contents);

/// Is a path a directory?
/// \param path The path to check.
/// \param is_dir Returns true if path represents a directory
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* IsDirectory(const std::string& path, bool* is_dir);

/// Join path segments into a longer path
/// \param segments The path segments.
/// \return the path formed by joining the segments.
std::string JoinPath(std::initializer_list<std::string> segments);

/// Returns the content in the model version path and the path to the content as
/// key-value pair.
/// \param model_repository_path The path to the model repository.
/// \param version The version of the model.
/// \param ignore_directories Whether the directories will be ignored.
/// \param ignore_files Whether the files will be ignored.
/// \param model_paths Returns the content in the model version path and
/// the path to the content.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* ModelPaths(
    const std::string& model_repository_path, uint64_t version,
    const bool ignore_directories, const bool ignore_files,
    std::unordered_map<std::string, std::string>* model_paths);

/// Create a CUDA stream appropriate for GPU<->CPU data transfer
/// operations for a given GPU device. The caller takes ownership of
/// the stream. 'stream' returns nullptr if GPU support is disabled.
///
/// \param device_id The ID of the GPU.
/// \param priority The stream priority. Use 0 for normal priority.
/// \param stream Returns the created stream.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* CreateCudaStream(
    const int device_id, const int cuda_stream_priority, cudaStream_t* stream);

/// Parse the string as long long integer.
///
/// \param value The string.
/// \param parse_value The long long integral value of the string.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* ParseLongLongValue(
    const std::string& value, int64_t* parsed_value);

/// Parse the string as unsigned long long integer.
///
/// \param value The string.
/// \param parse_value The unsigned long long integral value of the string.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* ParseUnsignedLongLongValue(
    const std::string& value, uint64_t* parsed_value);

/// Parse the string as boolean.
///
/// \param value The string.
/// \param parse_value The boolean value of the string.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* ParseBoolValue(
    const std::string& value, bool* parsed_value);

/// Parse the string as integer.
///
/// \param value The string.
/// \param parse_value The integral value of the string.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* ParseIntValue(const std::string& value, int* parsed_value);

/// Parse the string as double.
///
/// \param value The string.
/// \param parse_value The double value of the string.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* ParseDoubleValue(
    const std::string& value, double* parsed_value);

/// Return the value of the specified key in a JSON object.
///
/// \param params The JSON object containing the key-value mapping.
/// \param key The key to look up the value in the JSON object.
/// \param value Returns the value.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* GetParameterValue(
    triton::common::TritonJson::Value& params, const std::string& key,
    std::string* value);

/// Return the Triton server data type of the data type string specified
/// in model config JSON.
///
/// \param data_type_str The string representation of the data type.
/// \return the Triton server data type.
TRITONSERVER_DataType ModelConfigDataTypeToTritonServerDataType(
    const std::string& data_type_str);

/// Try to parse the requested parameter.
///
/// \param params The param in model config
/// \param mkey Key in the model config.
/// \param value The parsed string value.
/// \param default_value Default value to use when key is not found.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* TryParseModelStringParameter(
    triton::common::TritonJson::Value& params, const std::string& mkey,
    std::string* value, const std::string& default_value);

/// Try to parse the requested parameter.
///
/// \param params The param in model config
/// \param mkey Key in the model config.
/// \param value The parsed int value.
/// \param default_value Default value to use when key is not found.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* TryParseModelStringParameter(
    triton::common::TritonJson::Value& params, const std::string& mkey,
    int* value, const int& default_value);

/// Try to parse the requested parameter.
///
/// \param params The param in model config
/// \param mkey Key in the model config.
/// \param value The parsed bool value.
/// \param default_value Default value to use when key is not found.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* TryParseModelStringParameter(
    triton::common::TritonJson::Value& params, const std::string& mkey,
    bool* value, const bool& default_value);

/// Try to parse the requested parameter.
///
/// \param params The param in model config
/// \param mkey Key in the model config.
/// \param value The parsed uint64 value.
/// \param default_value Default value to use when key is not found.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* TryParseModelStringParameter(
    triton::common::TritonJson::Value& params, const std::string& mkey,
    uint64_t* value, const uint64_t& default_value);

/// Get a string representation of a tensor buffer.
///
/// \param str Returns the string.
/// \param buffer The base pointer to the tensor buffer.
/// \param buffer_byte_size The size of the buffer in bytes.
/// \param datatype The type of the tensor
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error* BufferAsTypedString(
    std::string& str, const char* buffer, size_t buffer_byte_size,
    TRITONSERVER_DataType datatype);

/// Get the ID of the request as a string formatted for logging.
///
/// \param request Request of which to get the ID.
/// \return a formatted string for logging the request ID.
std::string GetRequestId(TRITONBACKEND_Request* request);

}}  // namespace triton::backend
