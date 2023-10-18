<!--
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Python Based Backends

Python based backend is a special type of Triton's backends, which does
not require any C++ code. However, this type of backends depends on
[Python backend](https://github.com/triton-inference-server/python_backend)
and requires the following artifacts being present:
`libtriton_python.so`, `triton_python_backend_stub`,
and `triton_python_backend_utils.py`.

## Usage
To implement and use a Python based backend, make sure to follow these steps.
* Implement the
[`TritonPythonModel` interface](https://github.com/triton-inference-server/python_backend#usage),
which could be re-used as a backend by multiple models.
This script should be named `model.py`.
* Create a folder for your backend under
the backends directory (ex: /opt/tritonserver/backends)
with the corresponding backend name, containing the `model.py`.
For example, for a backend named `my_python_based_backend`,
Triton would expect to find the full path
`/opt/tritonserver/backends/my_python_based_backend/model.py`.
* Make sure that `libtriton_python.so`, `triton_python_backend_stub`,
and `triton_python_backend_utils.py` are present either under
`/opt/tritonserver/backends/my_python_based_backend/` or
`/opt/tritonserver/backends/python/`.
* Specify `my_python_based_backend` as a backend in `config.pbtxt`
for any model, that should use this backend.

```
...
backend: "my_python_based_backend"
...
```

Since Triton uses Python backend under the hood, it is expected,
to see `python` backend entry in server logs, even when Python backend
is not explicitly used.

```
I1013 21:52:45.756456 18668 server.cc:619]
+-------------------------+-------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| Backend                 | Path                                                        | Config                                                                                                              |
+-------------------------+-------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| python                  | /opt/tritonserver/backends/python/libtriton_python.so       | {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-compute-capability" |
|                         |                                                             | :"6.000000","default-max-batch-size":"4"}}                                                                          |
| my_python_based_backend | /opt/tritonserver/backends/my_python_based_backend/model.py | {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-compute-capability" |
|                         |                                                             | :"6.000000","default-max-batch-size":"4"}}                                                                          |
+-------------------------+-------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
```

## Background

In some use cases, it is sufficient to implement
[`TritonPythonModel` interface](https://github.com/triton-inference-server/python_backend#usage)
only once and re-use it across multiple models. As an example, please refer
to the [vLLM backend](https://github.com/triton-inference-server/vllm_backend),
which provides a common python script to serve models supported by vLLM.

Triton Inference Server can handle this special case and treats common
`model.py` script as a Python-based backend. In the scenario, when model
relies on a custom Python-based backend, Triton loads `libtriton_python.so`
first, this ensures that Triton knows how to send requests to the backend
for execution and the backend knows how to communicate with Triton. Then,
Triton makes sure to use common `model.py` from the backend's repository,
and not look for it in the model repository.

While the only required function is `execute`, it is typically helpful
to enhance your implementation by adding ` initialize`, `finalize`,
and any other helper functions. Users are also encouraged to make use of the
[`auto_complete_config`](https://github.com/triton-inference-server/python_backend#auto_complete_config)
function to define standardized input and output properties upfront.
