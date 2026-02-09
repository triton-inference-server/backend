# Copyright 2025-2026, NVIDIA CORPORATION. All rights reserved.
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

function(set_cuda_architectures_list)
    # Check if CUDA_ARCH_LIST environment variable is set
    if(DEFINED ENV{CUDA_ARCH_LIST})
        # Parse CUDA_ARCH_LIST: split by spaces, skip PTX, validate each code
        set(raw_input "$ENV{CUDA_ARCH_LIST}")
        string(REGEX REPLACE "PTX" "" raw_input "${raw_input}")
        string(REPLACE " " ";" arch_list "${raw_input}")

        set(cuda_arch_result_list "")
        foreach(arch IN LISTS arch_list)
            string(STRIP "${arch}" arch)
            if(arch STREQUAL "")
                continue()
            endif()
            # Normalize: remove dots so 10.0 -> 100, 12.0 -> 120
            string(REGEX REPLACE "\\." "" arch_num "${arch}")
            if(NOT arch_num MATCHES "^[0-9]+$")
                continue()
            endif()
            # Code >= 100 (10.x, 11.x, 12.x): use family code, no -real
            if(arch_num GREATER_EQUAL 100)
                math(EXPR arch_major "${arch_num} / 10")
                set(arch_entry "${arch_major}0f")
            else()
                set(arch_entry "${arch_num}-real")
            endif()
            list(APPEND cuda_arch_result_list "${arch_entry}")
        endforeach()
        # If last element is below 100 (has -real), leave it without -real
        list(LENGTH cuda_arch_result_list result_len)
        if(result_len GREATER 0)
            math(EXPR last_index "${result_len} - 1")
            list(GET cuda_arch_result_list ${last_index} last_entry)
            string(REGEX REPLACE "-real$" "" last_entry_stripped "${last_entry}")
            if(NOT last_entry_stripped STREQUAL last_entry)
                list(REMOVE_AT cuda_arch_result_list ${last_index})
                list(APPEND cuda_arch_result_list "${last_entry_stripped}")
            endif()
        endif()
        list(JOIN cuda_arch_result_list ";" cuda_arch_input)

        set(CUDA_ARCHITECTURES "${cuda_arch_input}" PARENT_SCOPE)

        message(STATUS "CUDA_ARCH_LIST found, defined CUDA_ARCHITECTURES: $ENV{CUDA_ARCH_LIST}")
    else()
        # Set default value if CUDA_ARCH_LIST is not present
        set(CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;100f;120f" PARENT_SCOPE)
        message(STATUS "CUDA_ARCH_LIST not found, using default values for CUDA_ARCHITECTURES: ${CUDA_ARCHITECTURES}")
    endif()
endfunction()

# Call the function to validate and set CUDA_ARCHITECTURES
set_cuda_architectures_list()
message(STATUS "Defined CUDA_ARCHITECTURES: ${CUDA_ARCHITECTURES}")
