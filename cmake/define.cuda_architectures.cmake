# Copyright 2025, NVIDIA CORPORATION. All rights reserved.
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
        string(REGEX REPLACE "PTX" "" cuda_arch_input "${cuda_arch_input}")
        string(REGEX REPLACE " " "-real;" cuda_arch_input "${cuda_arch_input}")
        string(REGEX REPLACE "-real;\$" "" cuda_arch_input "${cuda_arch_input}")
        string(REGEX REPLACE "\\." "" cuda_arch_input "${cuda_arch_input}")

        set(CUDA_ARCHITECTURES "${cuda_arch_input}" PARENT_SCOPE)

        message(STATUS "CUDA_ARCH_LIST found, defined CUDA_ARCHITECTURES: $ENV{CUDA_ARCH_LIST}")
    else()
        # Set default value if CUDA_ARCH_LIST is not present
        set(CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;100-real;103-real;120" PARENT_SCOPE)
        message(STATUS "CUDA_ARCH_LIST not found, using default values for CUDA_ARCHITECTURES: ${CUDA_ARCHITECTURES}")
    endif()
endfunction()

# Call the function to validate and set CUDA_ARCHITECTURES
set_cuda_architectures_list()
message(STATUS "Defined CUDA_ARCHITECTURES: ${CUDA_ARCHITECTURES}")
