// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "triton/backend/backend_common.h"

#ifdef _WIN32
// suppress the min and max definitions in Windef.h.
#define NOMINMAX
#include <Windows.h>

// _CRT_INTERNAL_NONSTDC_NAMES 1 before including Microsoft provided C Runtime
// library to expose declarations without "_" prefix to match POSIX style.
#define _CRT_INTERNAL_NONSTDC_NAMES 1
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif
#include <sys/stat.h>
#include <algorithm>
#include <cerrno>
#include <fstream>
#include <functional>
#include <memory>

#ifdef _WIN32
// <sys/stat.h> in Windows doesn't define S_ISDIR macro
#if !defined(S_ISDIR) && defined(S_IFMT) && defined(S_IFDIR)
#define S_ISDIR(m) (((m)&S_IFMT) == S_IFDIR)
#endif
#define F_OK 0
#endif

namespace triton { namespace backend {

TRITONSERVER_MemoryType
GetUsePinnedMemoryType(TRITONSERVER_MemoryType ref_buffer_type)
{
  // The following matrix is used for both input and output.
  // src   \ dest | non-pinned    | pinned     | device
  // non-pinned   | memcpy        | memcpy     | buffer needed
  // pinned       | memcpy        | memcpy     | cudaMemcpy
  // device       | buffer needed | cudaMemcpy | cudaMemcpy
  if (ref_buffer_type == TRITONSERVER_MEMORY_CPU_PINNED) {
    return TRITONSERVER_MEMORY_CPU_PINNED;
  }

  return (ref_buffer_type == TRITONSERVER_MEMORY_CPU) ? TRITONSERVER_MEMORY_GPU
                                                      : TRITONSERVER_MEMORY_CPU;
}

TRITONSERVER_Error_Code
StatusCodeToTritonCode(triton::common::Error::Code error_code)
{
  switch (error_code) {
    case triton::common::Error::Code::UNKNOWN:
      return TRITONSERVER_ERROR_UNKNOWN;
    case triton::common::Error::Code::INTERNAL:
      return TRITONSERVER_ERROR_INTERNAL;
    case triton::common::Error::Code::NOT_FOUND:
      return TRITONSERVER_ERROR_NOT_FOUND;
    case triton::common::Error::Code::INVALID_ARG:
      return TRITONSERVER_ERROR_INVALID_ARG;
    case triton::common::Error::Code::UNAVAILABLE:
      return TRITONSERVER_ERROR_UNAVAILABLE;
    case triton::common::Error::Code::UNSUPPORTED:
      return TRITONSERVER_ERROR_UNSUPPORTED;
    case triton::common::Error::Code::ALREADY_EXISTS:
      return TRITONSERVER_ERROR_ALREADY_EXISTS;

    default:
      break;
  }

  return TRITONSERVER_ERROR_UNKNOWN;
}

TRITONSERVER_Error*
CommonErrorToTritonError(triton::common::Error error)
{
  return TRITONSERVER_ErrorNew(
      StatusCodeToTritonCode(error.ErrorCode()), error.Message().c_str());
}

TRITONSERVER_Error*
ParseShape(
    common::TritonJson::Value& io, const std::string& name,
    std::vector<int64_t>* shape)
{
  common::TritonJson::Value shape_array;
  RETURN_IF_ERROR(io.MemberAsArray(name.c_str(), &shape_array));
  for (size_t i = 0; i < shape_array.ArraySize(); ++i) {
    int64_t d = 0;
    RETURN_IF_ERROR(shape_array.IndexAsInt(i, &d));
    shape->push_back(d);
  }

  return nullptr;  // success
}

std::string
ShapeToString(const int64_t* dims, const size_t dims_count)
{
  bool first = true;

  std::string str("[");
  for (size_t i = 0; i < dims_count; ++i) {
    const int64_t dim = dims[i];
    if (!first) {
      str += ",";
    }
    str += std::to_string(dim);
    first = false;
  }

  str += "]";
  return str;
}

std::string
ShapeToString(const std::vector<int64_t>& shape)
{
  return ShapeToString(shape.data(), shape.size());
}

int64_t
GetElementCount(const int64_t* dims, const size_t dims_count)
{
  bool first = true;
  int64_t cnt = 0;
  for (size_t i = 0; i < dims_count; i++) {
    if (dims[i] == WILDCARD_DIM) {
      return -1;
    }

    if (first) {
      cnt = dims[i];
      first = false;
    } else {
      cnt *= dims[i];
    }
  }

  return cnt;
}

int64_t
GetElementCount(const std::vector<int64_t>& shape)
{
  return GetElementCount(shape.data(), shape.size());
}

int64_t
GetByteSize(
    const TRITONSERVER_DataType& dtype, const std::vector<int64_t>& dims)
{
  size_t dt_size = TRITONSERVER_DataTypeByteSize(dtype);
  if (dt_size == 0) {
    return -1;
  }

  int64_t cnt = GetElementCount(dims);
  if (cnt == -1) {
    return -1;
  }

  return cnt * dt_size;
}

TRITONSERVER_Error*
ReadInputTensor(
    TRITONBACKEND_Request* request, const std::string& input_name, char* buffer,
    size_t* buffer_byte_size, const char* host_policy_name)
{
  TRITONBACKEND_Input* input;
  RETURN_IF_ERROR(
      TRITONBACKEND_RequestInput(request, input_name.c_str(), &input));

  uint64_t input_byte_size;
  uint32_t input_buffer_count;
  RETURN_IF_ERROR(TRITONBACKEND_InputPropertiesForHostPolicy(
      input, host_policy_name, nullptr, nullptr, nullptr, nullptr,
      &input_byte_size, &input_buffer_count));
  RETURN_ERROR_IF_FALSE(
      input_byte_size <= *buffer_byte_size, TRITONSERVER_ERROR_INVALID_ARG,
      std::string(
          "buffer too small for input tensor '" + input_name + "', " +
          std::to_string(*buffer_byte_size) + " < " +
          std::to_string(input_byte_size)));

  size_t output_buffer_offset = 0;
  for (uint32_t b = 0; b < input_buffer_count; ++b) {
    const void* input_buffer = nullptr;
    uint64_t input_buffer_byte_size = 0;
    TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t input_memory_type_id = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputBufferForHostPolicy(
        input, host_policy_name, b, &input_buffer, &input_buffer_byte_size,
        &input_memory_type, &input_memory_type_id));
    RETURN_ERROR_IF_FALSE(
        input_memory_type != TRITONSERVER_MEMORY_GPU,
        TRITONSERVER_ERROR_INTERNAL,
        std::string("expected input tensor in CPU memory"));

    memcpy(buffer + output_buffer_offset, input_buffer, input_buffer_byte_size);
    output_buffer_offset += input_buffer_byte_size;
  }

  *buffer_byte_size = input_byte_size;

  return nullptr;  // success
}

TRITONSERVER_Error*
CheckAllowedModelInput(
    common::TritonJson::Value& io, const std::set<std::string>& allowed)
{
  std::string io_name;
  RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
  if (allowed.find(io_name) == allowed.end()) {
    std::string astr;
    for (const auto& a : allowed) {
      if (!astr.empty()) {
        astr.append(", ");
      }
      astr.append(a);
    }

    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected inference input '" + io_name +
            "', allowed inputs are: " + astr)
            .c_str());
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
CheckAllowedModelOutput(
    common::TritonJson::Value& io, const std::set<std::string>& allowed)
{
  std::string io_name;
  RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
  if (allowed.find(io_name) == allowed.end()) {
    std::string astr;
    for (const auto& a : allowed) {
      if (!astr.empty()) {
        astr.append(", ");
      }
      astr.append(a);
    }

    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected inference output '" + io_name +
            "', allowed outputs are: " + astr)
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
GetBooleanSequenceControlProperties(
    common::TritonJson::Value& batcher, const std::string& model_name,
    const std::string& control_kind, const bool required,
    std::string* tensor_name, std::string* tensor_datatype,
    float* fp32_false_value, float* fp32_true_value, int32_t* int32_false_value,
    int32_t* int32_true_value)
{
  // Make sure same tensor is not configured for multiple controls
  std::set<std::string> seen_tensors;

  // Make sure the control kind is not mentioned multiple times.
  bool seen_control = false;

  common::TritonJson::Value control_inputs;
  if (batcher.Find("control_input", &control_inputs)) {
    for (size_t ci_idx = 0; ci_idx < control_inputs.ArraySize(); ci_idx++) {
      common::TritonJson::Value control_input;
      RETURN_IF_ERROR(control_inputs.IndexAsObject(ci_idx, &control_input));
      std::string input_name;
      RETURN_IF_ERROR(control_input.MemberAsString("name", &input_name));
      if (input_name.empty()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string(
                 "sequence batching control tensor must have a name for ") +
             model_name)
                .c_str());
      }

      if (seen_tensors.find(input_name) != seen_tensors.end()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("sequence batching control tensor '") + input_name +
             "' is specified for multiple control kinds for " + model_name)
                .c_str());
      }

      seen_tensors.insert(input_name);
      common::TritonJson::Value controls;
      if (control_input.Find("control", &controls)) {
        for (size_t c_idx = 0; c_idx < controls.ArraySize(); c_idx++) {
          common::TritonJson::Value c;
          RETURN_IF_ERROR(controls.IndexAsObject(c_idx, &c));
          std::string kind_str;
          RETURN_IF_ERROR(c.MemberAsString("kind", &kind_str));
          if (kind_str == control_kind) {
            if (seen_control) {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string(
                       "sequence batching specifies multiple " + control_kind +
                       " tensors for " + model_name)
                       .c_str()));
            }

            *tensor_name = input_name;
            seen_control = true;

            common::TritonJson::Value int32_false_true, fp32_false_true;
            bool found_int32 =
                (c.Find("int32_false_true", &int32_false_true) &&
                 (int32_false_true.ArraySize() > 0));
            bool found_fp32 =
                (c.Find("fp32_false_true", &fp32_false_true) &&
                 (fp32_false_true.ArraySize() > 0));
            if (found_fp32 && found_int32) {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string(
                       "sequence batching specifies both 'int32_false_true' "
                       "and "
                       "'fp32_false_true' for " +
                       control_kind + " for " + model_name))
                      .c_str());
            }
            if (!(found_int32 || found_fp32)) {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string(
                       "sequence batching must specify either "
                       "'int32_false_true' or "
                       "'fp32_false_true' for " +
                       control_kind + " for " + model_name))
                      .c_str());
            }
            if (found_int32) {
              if (int32_false_true.ArraySize() != 2) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    (std::string(
                         "sequence batching control 'int32_false_true' must "
                         "have "
                         "exactly 2 entries for " +
                         control_kind + " for " + model_name))
                        .c_str());
              }
              if (tensor_datatype != nullptr) {
                *tensor_datatype = "TYPE_INT32";
              }
              if (int32_false_value != nullptr) {
                int64_t value;
                RETURN_IF_ERROR(int32_false_true.IndexAsInt(0, &value));
                *int32_false_value = value;
              }
              if (int32_true_value != nullptr) {
                int64_t value;
                RETURN_IF_ERROR(int32_false_true.IndexAsInt(1, &value));
                *int32_true_value = value;
              }
            } else {
              if (fp32_false_true.ArraySize() != 2) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    (std::string(
                         "sequence batching control 'fp32_false_true' must "
                         "have exactly "
                         "2 entries for " +
                         control_kind + " for " + model_name))
                        .c_str());
              }

              if (tensor_datatype != nullptr) {
                *tensor_datatype = "TYPE_FP32";
              }
              if (fp32_false_value != nullptr) {
                double value = 0.0;
                RETURN_IF_ERROR(fp32_false_true.IndexAsDouble(0, &value));
                *fp32_false_value = value;
              }
              if (fp32_true_value != nullptr) {
                double value = 0.0;
                RETURN_IF_ERROR(fp32_false_true.IndexAsDouble(1, &value));
                *fp32_true_value = value;
              }
            }
          }
        }
      }
    }
  }

  if (!seen_control) {
    if (required) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string(
               "sequence batching control tensor must specify a " +
               control_kind + " value for " + model_name))
              .c_str());
    }

    tensor_name->clear();
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
GetTypedSequenceControlProperties(
    common::TritonJson::Value& batcher, const std::string& model_name,
    const std::string& control_kind, const bool required,
    std::string* tensor_name, std::string* tensor_datatype)
{
  // Make sure same tensor is not configured for multiple controls
  std::set<std::string> seen_tensors;

  // Make sure the control kind is not mentioned multiple times.
  bool seen_control = false;

  common::TritonJson::Value control_inputs;
  if (batcher.Find("control_input", &control_inputs)) {
    for (size_t ci_idx = 0; ci_idx < control_inputs.ArraySize(); ci_idx++) {
      common::TritonJson::Value control_input;
      RETURN_IF_ERROR(control_inputs.IndexAsObject(ci_idx, &control_input));
      std::string input_name;
      RETURN_IF_ERROR(control_input.MemberAsString("name", &input_name));
      if (input_name.empty()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string(
                 "sequence batching control tensor must have a name for ") +
             model_name)
                .c_str());
      }
      if (seen_tensors.find(input_name) != seen_tensors.end()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("sequence batching control tensor '") + input_name +
             "' is specified for multiple control kinds for " + model_name)
                .c_str());
      }

      seen_tensors.insert(input_name);
      common::TritonJson::Value controls;
      if (control_input.Find("control", &controls)) {
        for (size_t c_idx = 0; c_idx < controls.ArraySize(); c_idx++) {
          common::TritonJson::Value c;
          RETURN_IF_ERROR(controls.IndexAsObject(c_idx, &c));
          std::string kind_str;
          RETURN_IF_ERROR(c.MemberAsString("kind", &kind_str));
          if (kind_str == control_kind) {
            if (seen_control) {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string(
                       "sequence batching specifies multiple " + control_kind +
                       " tensors for " + model_name)
                       .c_str()));
            }

            *tensor_name = input_name;
            if (tensor_datatype != nullptr) {
              c.MemberAsString("data_type", tensor_datatype);
            }

            seen_control = true;

            common::TritonJson::Value int32_false_true, fp32_false_true;
            bool found_int32 =
                (c.Find("int32_false_true", &int32_false_true) &&
                 (int32_false_true.ArraySize() > 0));
            bool found_fp32 =
                (c.Find("fp32_false_true", &fp32_false_true) &&
                 (fp32_false_true.ArraySize() > 0));
            if (found_int32 || found_fp32) {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string(
                       "sequence batching must not specify either "
                       "'int32_false_true' "
                       "nor 'fp32_false_true' for " +
                       control_kind + " for " + model_name))
                      .c_str());
            }
          }
        }
      }
    }
  }

  if (!seen_control) {
    if (required) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string(
               "sequence batching control tensor must specify a " +
               control_kind + " value for " + model_name))
              .c_str());
    }

    tensor_name->clear();
  }

  return nullptr;  // success
}

void
RequestsRespondWithError(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    TRITONSERVER_Error* response_err, const bool release_request)
{
  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err != nullptr) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "fail to create response");
      TRITONSERVER_ErrorDelete(err);
    } else {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, response_err),
          "fail to send error response");
    }

    if (release_request) {
      LOG_IF_ERROR(
          TRITONBACKEND_RequestRelease(
              requests[i], TRITONSERVER_REQUEST_RELEASE_ALL),
          "fail to release request");
    }
  }

  TRITONSERVER_ErrorDelete(response_err);
}

void
SendErrorForResponses(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count, TRITONSERVER_Error* response_err)
{
  for (size_t i = 0; i < response_count; i++) {
    TRITONBACKEND_Response* response = (*responses)[i];
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, response_err),
          "fail to send error response");
      (*responses)[i] = nullptr;
    }
  }

  TRITONSERVER_ErrorDelete(response_err);
}

TRITONSERVER_Error*
CopyBuffer(
    const std::string& msg, const TRITONSERVER_MemoryType src_memory_type,
    const int64_t src_memory_type_id,
    const TRITONSERVER_MemoryType dst_memory_type,
    const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
    void* dst, cudaStream_t cuda_stream, bool* cuda_used)
{
  *cuda_used = false;

  // For CUDA memcpy, all host to host copy will be blocked in respect to the
  // host, so use memcpy() directly. In this case, need to be careful on whether
  // the src buffer is valid.
  if ((src_memory_type != TRITONSERVER_MEMORY_GPU) &&
      (dst_memory_type != TRITONSERVER_MEMORY_GPU)) {
    memcpy(dst, src, byte_size);
  } else {
#ifdef TRITON_ENABLE_GPU
    // [TODO] use cudaMemcpyDefault if UVM is supported for the device
    auto copy_kind = cudaMemcpyDeviceToDevice;
    if (src_memory_type != TRITONSERVER_MEMORY_GPU) {
      copy_kind = cudaMemcpyHostToDevice;
    } else if (dst_memory_type != TRITONSERVER_MEMORY_GPU) {
      copy_kind = cudaMemcpyDeviceToHost;
    }

    if ((src_memory_type_id != dst_memory_type_id) &&
        (copy_kind == cudaMemcpyDeviceToDevice)) {
      RETURN_IF_CUDA_ERROR(
          cudaMemcpyPeerAsync(
              dst, dst_memory_type_id, src, src_memory_type_id, byte_size,
              cuda_stream),
          TRITONSERVER_ERROR_INTERNAL, msg + ": failed to perform CUDA copy");
    } else {
      RETURN_IF_CUDA_ERROR(
          cudaMemcpyAsync(dst, src, byte_size, copy_kind, cuda_stream),
          TRITONSERVER_ERROR_INTERNAL, msg + ": failed to perform CUDA copy");
    }

    *cuda_used = true;
#else
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string(msg + ": try to use CUDA copy while GPU is not supported")
            .c_str());
#endif  // TRITON_ENABLE_GPU
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
GetDirectoryContents(const std::string& path, std::set<std::string>* contents)
{
#ifdef _WIN32
  WIN32_FIND_DATA entry;
  HANDLE dir = FindFirstFile(path.c_str(), &entry);
  if (dir == INVALID_HANDLE_VALUE) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("failed to open directory: ") + path).c_str());
  }
  if ((entry.cFileName != ".") && (entry.cFileName != "..")) {
    contents->insert(entry.cFileName);
  }
  while (FindNextFileA(dir, &entry)) {
    if ((entry.cFileName != ".") && (entry.cFileName != "..")) {
      contents->insert(entry.cFileName);
    }
  }

  FindClose(dir);
#else
  DIR* dir = opendir(path.c_str());
  if (dir == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("failed to open directory: ") + path).c_str());
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string entryname = entry->d_name;
    if ((entryname != ".") && (entryname != "..")) {
      contents->insert(entryname);
    }
  }

  closedir(dir);
#endif
  return nullptr;  // success
}

TRITONSERVER_Error*
FileExists(const std::string& path, bool* exists)
{
  *exists = (access(path.c_str(), F_OK) == 0);
  return nullptr;  // success
}

TRITONSERVER_Error*
ReadTextFile(const std::string& path, std::string* contents)
{
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("failed to open/read file '" + path + "': " + strerror(errno))
            .c_str());
  }

  in.seekg(0, std::ios::end);
  contents->resize(in.tellg());
  in.seekg(0, std::ios::beg);
  in.read(&(*contents)[0], contents->size());
  in.close();

  return nullptr;  // success
}

TRITONSERVER_Error*
IsDirectory(const std::string& path, bool* is_dir)
{
  *is_dir = false;

  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("failed to stat file ") + path).c_str());
  }

  *is_dir = S_ISDIR(st.st_mode);
  return nullptr;  // success
}

std::string
JoinPath(std::initializer_list<std::string> segments)
{
  std::string joined;

  for (const auto& seg : segments) {
    if (joined.empty()) {
      joined = seg;
    } else if (!seg.empty() && (seg[0] == '/')) {  // IsAbsolutePath(seg)
      if (joined[joined.size() - 1] == '/') {
        joined.append(seg.substr(1));
      } else {
        joined.append(seg);
      }
    } else {  // !IsAbsolutePath(seg)
      if (joined[joined.size() - 1] != '/') {
        joined.append("/");
      }
      joined.append(seg);
    }
  }

  return joined;
}

TRITONSERVER_Error*
ModelPaths(
    const std::string& model_repository_path, uint64_t version,
    const bool ignore_directories, const bool ignore_files,
    std::unordered_map<std::string, std::string>* model_paths)
{
  std::set<std::string> model_files;
  // Read all the files in 'path' and filter by type for different requirements
  auto path = JoinPath({model_repository_path, std::to_string(version)});
  RETURN_IF_ERROR(GetDirectoryContents(path, &model_files));
  if (ignore_directories) {
    // Erase directory entries...
    for (auto iter = model_files.begin(); iter != model_files.end();) {
      bool is_dir;
      RETURN_IF_ERROR(IsDirectory(JoinPath({path, *iter}), &is_dir));
      if (is_dir) {
        iter = model_files.erase(iter);
      } else {
        ++iter;
      }
    }
  }
  if (ignore_files) {
    // Erase non-directory entries...
    for (auto iter = model_files.begin(); iter != model_files.end();) {
      bool is_dir;
      RETURN_IF_ERROR(IsDirectory(JoinPath({path, *iter}), &is_dir));
      if (!is_dir) {
        iter = model_files.erase(iter);
      } else {
        ++iter;
      }
    }
  }

  for (const auto& filename : model_files) {
    const auto model_path = JoinPath({path, filename});
    model_paths->emplace(
        std::piecewise_construct, std::make_tuple(filename),
        std::make_tuple(model_path));
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
CreateCudaStream(
    const int device_id, const int cuda_stream_priority, cudaStream_t* stream)
{
  *stream = nullptr;

#ifdef TRITON_ENABLE_GPU
  // Make sure that correct device is set before creating stream and
  // then restore the device to what was set by the caller.
  int current_device;
  auto cuerr = cudaGetDevice(&current_device);
  bool overridden = false;
  if (cuerr == cudaSuccess) {
    overridden = (current_device != device_id);
    if (overridden) {
      cuerr = cudaSetDevice(device_id);
    }
  }

  if (cuerr == cudaSuccess) {
    cuerr = cudaStreamCreateWithPriority(
        stream, cudaStreamDefault, cuda_stream_priority);
  }

  if (overridden) {
    cudaSetDevice(current_device);
  }

  if (cuerr != cudaSuccess) {
    *stream = nullptr;
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unable to create stream: ") + cudaGetErrorString(cuerr))
            .c_str());
  }
#endif  // TRITON_ENABLE_GPU

  return nullptr;  // success
}

TRITONSERVER_Error*
ParseLongLongValue(const std::string& value, int64_t* parsed_value)
{
  try {
    *parsed_value = std::stoll(value);
  }
  catch (const std::invalid_argument& ia) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("failed to convert '") + value +
         "' to long long integral number")
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ParseBoolValue(const std::string& value, bool* parsed_value)
{
  std::string lvalue = value;
  std::transform(
      lvalue.begin(), lvalue.end(), lvalue.begin(),
      [](unsigned char c) { return std::tolower(c); });

  if ((lvalue == "true") || (lvalue == "on") || (lvalue == "1")) {
    *parsed_value = true;
    return nullptr;  // success
  }
  if ((lvalue == "false") || (lvalue == "off") || (lvalue == "0")) {
    *parsed_value = false;
    return nullptr;  // success
  }

  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INVALID_ARG,
      (std::string("failed to convert '") + value + "' to boolean").c_str());
}

TRITONSERVER_Error*
ParseIntValue(const std::string& value, int* parsed_value)
{
  try {
    *parsed_value = std::stoi(value);
  }
  catch (const std::invalid_argument& ia) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("failed to convert '") + value + "' to integral number")
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ParseDoubleValue(const std::string& value, double* parsed_value)
{
  try {
    *parsed_value = std::stod(value);
  }
  catch (const std::invalid_argument& ia) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("failed to convert '") + value + "' to double number")
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
GetParameterValue(
    triton::common::TritonJson::Value& params, const std::string& key,
    std::string* value)
{
  triton::common::TritonJson::Value json_value;
  RETURN_ERROR_IF_FALSE(
      params.Find(key.c_str(), &json_value), TRITONSERVER_ERROR_NOT_FOUND,
      std::string("model configuration is missing the parameter ") + key);
  RETURN_IF_ERROR(json_value.MemberAsString("string_value", value));
  return nullptr;  // success
}

TRITONSERVER_Error*
BatchInput::ParseFromModelConfig(
    triton::common::TritonJson::Value& config,
    std::vector<BatchInput>* batch_inputs)
{
  batch_inputs->clear();
  triton::common::TritonJson::Value bis;
  RETURN_IF_ERROR(config.MemberAsArray("batch_input", &bis));
  for (size_t i = 0; i < bis.ArraySize(); ++i) {
    batch_inputs->emplace_back();
    auto& batch_input = batch_inputs->back();
    triton::common::TritonJson::Value bi;
    RETURN_IF_ERROR(bis.IndexAsObject(i, &bi));
    {
      triton::common::TritonJson::Value bi_target_names;
      RETURN_IF_ERROR(bi.MemberAsArray("target_name", &bi_target_names));
      for (size_t i = 0; i < bi_target_names.ArraySize(); ++i) {
        std::string tn;
        RETURN_IF_ERROR(bi_target_names.IndexAsString(i, &tn));
        batch_input.target_names_.emplace_back(std::move(tn));
      }
    }
    {
      std::string bi_kind;
      RETURN_IF_ERROR(bi.MemberAsString("kind", &bi_kind));
      if (bi_kind == "BATCH_ELEMENT_COUNT") {
        batch_input.kind_ = Kind::BATCH_ELEMENT_COUNT;
      } else if (bi_kind == "BATCH_ACCUMULATED_ELEMENT_COUNT") {
        batch_input.kind_ = Kind::BATCH_ACCUMULATED_ELEMENT_COUNT;
      } else if (bi_kind == "BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO") {
        batch_input.kind_ = Kind::BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO;
      } else if (bi_kind == "BATCH_MAX_ELEMENT_COUNT_AS_SHAPE") {
        batch_input.kind_ = Kind::BATCH_MAX_ELEMENT_COUNT_AS_SHAPE;
      } else {
        RETURN_ERROR_IF_FALSE(
            false, TRITONSERVER_ERROR_INVALID_ARG,
            std::string("unexpected batch input kind '" + bi_kind + "'"));
      }
    }
    {
      std::string bi_dtype;
      RETURN_IF_ERROR(bi.MemberAsString("data_type", &bi_dtype));
      batch_input.data_type_ =
          ModelConfigDataTypeToTritonServerDataType(bi_dtype);
      RETURN_ERROR_IF_TRUE(
          batch_input.data_type_ == TRITONSERVER_TYPE_INVALID,
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("unexpected batch input data type '" + bi_dtype + "'"));
    }
    {
      triton::common::TritonJson::Value bi_source_inputs;
      RETURN_IF_ERROR(bi.MemberAsArray("source_input", &bi_source_inputs));
      for (size_t i = 0; i < bi_source_inputs.ArraySize(); ++i) {
        std::string si;
        RETURN_IF_ERROR(bi_source_inputs.IndexAsString(i, &si));
        batch_input.source_inputs_.emplace_back(std::move(si));
      }
    }
  }

  return nullptr;  // success
}

TRITONSERVER_DataType
ModelConfigDataTypeToTritonServerDataType(const std::string& data_type_str)
{
  // Must start with "TYPE_".
  if (data_type_str.rfind("TYPE_", 0) != 0) {
    return TRITONSERVER_TYPE_INVALID;
  }

  const std::string dtype = data_type_str.substr(strlen("TYPE_"));

  if (dtype == "BOOL") {
    return TRITONSERVER_TYPE_BOOL;
  } else if (dtype == "UINT8") {
    return TRITONSERVER_TYPE_UINT8;
  } else if (dtype == "UINT16") {
    return TRITONSERVER_TYPE_UINT16;
  } else if (dtype == "UINT32") {
    return TRITONSERVER_TYPE_UINT32;
  } else if (dtype == "UINT64") {
    return TRITONSERVER_TYPE_UINT64;
  } else if (dtype == "INT8") {
    return TRITONSERVER_TYPE_INT8;
  } else if (dtype == "INT16") {
    return TRITONSERVER_TYPE_INT16;
  } else if (dtype == "INT32") {
    return TRITONSERVER_TYPE_INT32;
  } else if (dtype == "INT64") {
    return TRITONSERVER_TYPE_INT64;
  } else if (dtype == "FP16") {
    return TRITONSERVER_TYPE_FP16;
  } else if (dtype == "FP32") {
    return TRITONSERVER_TYPE_FP32;
  } else if (dtype == "FP64") {
    return TRITONSERVER_TYPE_FP64;
  } else if (dtype == "STRING") {
    return TRITONSERVER_TYPE_BYTES;
  }

  return TRITONSERVER_TYPE_INVALID;
}

TRITONSERVER_Error*
BatchOutput::ParseFromModelConfig(
    triton::common::TritonJson::Value& config,
    std::vector<BatchOutput>* batch_outputs)
{
  batch_outputs->clear();
  triton::common::TritonJson::Value bos;
  RETURN_IF_ERROR(config.MemberAsArray("batch_output", &bos));
  for (size_t i = 0; i < bos.ArraySize(); ++i) {
    batch_outputs->emplace_back();
    auto& batch_output = batch_outputs->back();
    triton::common::TritonJson::Value bo;
    RETURN_IF_ERROR(bos.IndexAsObject(i, &bo));
    {
      triton::common::TritonJson::Value bo_target_names;
      RETURN_IF_ERROR(bo.MemberAsArray("target_name", &bo_target_names));
      for (size_t i = 0; i < bo_target_names.ArraySize(); ++i) {
        std::string tn;
        RETURN_IF_ERROR(bo_target_names.IndexAsString(i, &tn));
        batch_output.target_names_.emplace_back(std::move(tn));
      }
    }
    {
      std::string bo_kind;
      RETURN_IF_ERROR(bo.MemberAsString("kind", &bo_kind));
      if (bo_kind == "BATCH_SCATTER_WITH_INPUT_SHAPE") {
        batch_output.kind_ = Kind::BATCH_SCATTER_WITH_INPUT_SHAPE;
        // Keep track of the output info for later cross reference with input
        int64_t mbs = 0;
        RETURN_IF_ERROR(config.MemberAsInt("max_batch_size", &mbs));
        if (mbs != 0) {
          batch_output.shape_.push_back(-1);
        }
        triton::common::TritonJson::Value ios;
        RETURN_IF_ERROR(config.MemberAsArray("output", &ios));
        for (size_t i = 0; i < ios.ArraySize(); i++) {
          triton::common::TritonJson::Value io;
          RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
          std::string io_name;
          RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
          if (io_name == batch_output.target_names_[0]) {
            std::string io_dtype;
            RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
            batch_output.data_type_ =
                ModelConfigDataTypeToTritonServerDataType(io_dtype);
            // If a reshape is provided for the input then use that when
            // validating that the model matches what is expected.
            triton::common::TritonJson::Value reshape;
            if (io.Find("reshape", &reshape)) {
              RETURN_IF_ERROR(
                  ParseShape(reshape, "shape", &batch_output.shape_));
            } else {
              RETURN_IF_ERROR(ParseShape(io, "dims", &batch_output.shape_));
            }
            break;
          }
        }
      } else {
        RETURN_ERROR_IF_FALSE(
            false, TRITONSERVER_ERROR_INVALID_ARG,
            std::string("unexpected batch output kind '" + bo_kind + "'"));
      }
    }
    {
      triton::common::TritonJson::Value bo_source_inputs;
      RETURN_IF_ERROR(bo.MemberAsArray("source_input", &bo_source_inputs));
      for (size_t i = 0; i < bo_source_inputs.ArraySize(); ++i) {
        std::string si;
        RETURN_IF_ERROR(bo_source_inputs.IndexAsString(i, &si));
        batch_output.source_inputs_.emplace_back(std::move(si));
      }
    }
  }

  return nullptr;  // success
}

}}  // namespace triton::backend
