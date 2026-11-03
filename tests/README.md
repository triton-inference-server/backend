<!--
# Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
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
-->
# Input collector regression test

This standalone test requires Linux, a CUDA GPU, the CUDA development toolkit,
CMake 3.31.8 or newer, and a Triton runtime (`libtritonserver.so`). It builds the
backend utilities from this checkout. Request-buffer and pinned-pool APIs are test
doubles; the collector, data copies, and CUDA synchronization are real.

```sh
CUDA_ARCH_LIST=8.9 cmake -S tests -B build/tests \
  -DTRITONSERVER_LIBRARY=/opt/tritonserver/lib/libtritonserver.so
cmake --build build/tests --target backend_input_collector_test
ctest --test-dir build/tests --output-on-failure
```

Select the CUDA architecture for the test GPU. The test checks completion across
100 independent inputs, event reuse, GPU-to-pinned-to-CPU staging, and the
no-event stream synchronization fallback. A missing staging completion record is
tested using a gated CUDA host callback, without timing-dependent sleeps.
