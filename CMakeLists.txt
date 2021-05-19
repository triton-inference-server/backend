# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)

project(tritonbackend LANGUAGES C CXX)

#
# Options
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend utilities" ON)
option(TRITON_ENABLE_MALI_GPU "Enable Arm MALI GPU support in backend utilities" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend utilities" ON)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core)

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
  find_package(CUDA REQUIRED)
  message(STATUS "Using CUDA ${CUDA_VERSION}")
  set(CUDA_NVCC_FLAGS -std=c++11)

  if(CUDA_VERSION VERSION_GREATER "10.1" OR CUDA_VERSION VERSION_EQUAL "10.1")
    add_definitions(-DTRITON_ENABLE_CUDA_GRAPH=1)
  else()
    message(WARNING "CUDA ${CUDA_VERSION} does not support CUDA graphs.")
  endif()
endif() # TRITON_ENABLE_GPU

#
# Backend library containing useful source and utilities
#
set(SRC_FILES 
  "src/backend_common.cc"
  "src/backend_input_collector.cc"
  "src/backend_memory.cc"
  "src/backend_model_instance.cc"
  "src/backend_model.cc"
  "src/backend_output_responder.cc"
)

if(${TRITON_ENABLE_GPU})
  set(SRC_FILES ${SRC_FILES} "src/kernel.h")
endif() # TRITON_ENABLE_GPU

add_library(
  triton-backend-utils
  ${SRC_FILES}
)

if(${TRITON_ENABLE_GPU})
  set(HOST_COMPILER_FLAGS "")
  if (WIN32)
    set(HOST_COMPILER_FLAGS "/MD")
  else()
    set(HOST_COMPILER_FLAGS "-fPIC")
  endif()

  set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
  cuda_add_library(
    kernel-library-new
    src/kernel.cu src/kernel.h
    OPTIONS -arch compute_53
    OPTIONS -code compute_53,sm_53,sm_60,sm_61,sm_62,sm_70,sm_72,sm_75
    OPTIONS -Xcompiler ${HOST_COMPILER_FLAGS}
  )
endif() # TRITON_ENABLE_GPU

add_library(
  TritonBackend::triton-backend-utils ALIAS triton-backend-utils
)

target_include_directories(
  triton-backend-utils
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  message("Using MSVC as compiler, default target on Windows 10. "
      "If the target system is not Windows 10, please update _WIN32_WINNT "
      "to corresponding value.")
endif()
target_compile_features(triton-backend-utils PRIVATE cxx_std_11)
target_compile_options(
  triton-backend-utils
  PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc>
)

# TRITON_ENABLE_GPU exposed in header so set PUBLIC
if(${TRITON_ENABLE_GPU})
target_compile_definitions(
  triton-backend-utils
  PUBLIC TRITON_ENABLE_GPU=1
)
endif() # TRITON_ENABLE_GPU

# TRITON_ENABLE_STATS exposed in header so set PUBLIC
if(${TRITON_ENABLE_STATS})
target_compile_definitions(
  triton-backend-utils
  PUBLIC TRITON_ENABLE_STATS=1
)
endif() # TRITON_ENABLE_STATS

set_target_properties(
  triton-backend-utils PROPERTIES
  WINDOWS_EXPORT_ALL_SYMBOLS TRUE
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME tritonbackendutils
)

target_link_libraries(
  triton-backend-utils
  PUBLIC
    triton-core-backendapi         # from repo-core
    triton-core-serverapi          # from repo-core
    triton-common-async-work-queue # from repo-common
    triton-common-json             # from repo-common
)

if(${TRITON_ENABLE_GPU})
  target_link_libraries(
    triton-backend-utils
    PUBLIC
      CUDA::cudart
    PRIVATE
    kernel-library-new
  )
endif() # TRITON_ENABLE_GPU

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonBackend)

install(
  TARGETS
    triton-backend-utils
  EXPORT
    triton-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(${TRITON_ENABLE_GPU})
  install(
    TARGETS
      kernel-library-new
    EXPORT
      triton-backend-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif() # TRITON_ENABLE_GPU

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  EXPORT
    triton-backend-targets
  FILE
    TritonBackendTargets.cmake
  NAMESPACE
    TritonBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonBackendTargets.cmake
  NAMESPACE TritonBackend::
)

export(PACKAGE TritonBackend)
