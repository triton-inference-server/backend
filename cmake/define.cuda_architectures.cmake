# Copyright 2020-2025, NVIDIA CORPORATION. All rights reserved.
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
        # Parse the existing CUDA_ARCH_LIST
        set(cuda_arch_input "$ENV{CUDA_ARCH_LIST}")
        string(REPLACE " " ";" cuda_arch_list "${cuda_arch_input}")

        # Convert each architecture to the required format
        set(converted_archs "")
        list(LENGTH cuda_arch_list arch_count)
        math(EXPR last_index "${arch_count} - 1")

        foreach(arch_index RANGE ${last_index})
            list(GET cuda_arch_list ${arch_index} arch)

            # Remove any trailing characters and convert to integer format
            string(REGEX REPLACE "\\..*$" "" major_version "${arch}")
            string(REGEX REPLACE "^.*\\." "" minor_version "${arch}")

            # Handle cases where there's no decimal point
            if(minor_version STREQUAL "")
                set(minor_version "0")
            endif()

            # Convert to the required format (e.g., 7.5 -> 75-real)
            # Last architecture should not have "real" suffix
            if(arch_index EQUAL last_index)
                set(converted_arch "${major_version}${minor_version}")
            else()
                set(converted_arch "${major_version}${minor_version}-real")
            endif()

            list(APPEND converted_archs "${converted_arch}")
        endforeach()

        # Join the list with semicolons
        string(REPLACE ";" ";" converted_arch_string "${converted_archs}")
        set(CUDA_ARCHITECTURES "${converted_arch_string}" )

        message(STATUS "CUDA_ARCH_LIST found, defined CUDA_ARCHITECTURES: ${CUDA_ARCHITECTURES}")
    else()
        # Set default value if CUDA_ARCH_LIST is not present
        set(CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;100-real;120" )
        message(STATUS "CUDA_ARCH_LIST not found, using default values for CUDA_ARCHITECTURES: ${CUDA_ARCHITECTURES}")
    endif()
endfunction()

# Call the function to validate and set CUDA_ARCHITECTURES
set_cuda_architectures_list()
