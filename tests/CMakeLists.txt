# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.31.8)
project(backend_input_collector_tests LANGUAGES C CXX)

# Standalone CUDA regression test; does not change normal backend builds.
include(CTest)
find_package(CUDAToolkit REQUIRED)
add_subdirectory(.. backend)
find_library(TRITONSERVER_LIBRARY tritonserver HINTS /opt/tritonserver/lib REQUIRED)

add_executable(backend_input_collector_test backend_input_collector_test.cc)
target_link_libraries(
  backend_input_collector_test PRIVATE
  triton-backend-utils CUDA::cudart ${TRITONSERVER_LIBRARY}
)
target_compile_features(backend_input_collector_test PRIVATE cxx_std_20)
target_link_options(backend_input_collector_test PRIVATE -Wl,--wrap=cudaEventRecord)
add_test(NAME backend_input_collector COMMAND backend_input_collector_test)
set_tests_properties(backend_input_collector PROPERTIES TIMEOUT 60)
