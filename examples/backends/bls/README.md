<!--
# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# *BLS* Triton Backend

The [*BLS*](../bls) backend demonstrates using in-process C-API to
execute inferences within the backend. This backend serves as an example to
backend developers for implementing their own custom pipeline in C++.
For Python use cases, please refer to 
[Business Logic Scripting](https://github.com/triton-inference-server/python_backend#business-logic-scripting)
section in Python backend.

The source code for the *bls* backend is contained in
[src](./src).

* [backend.cc](./src/backend.cc) contains the main backend
implementation. The content of this file is not BLS specific. It only includes
the required Triton backend functions that is standard for any backend
implementation. The BLS logic is set off in the
[`TRITONBACKEND_ModelInstanceExecute`](./src/backend.cc#L316).
function.

* [bls.h](./src/bls.h) is where the BLS (class `BLSExecutor`) of
this example is located. You can refer to this file to see how to interact with
Triton in-process C-API to build the custom execution pipeline.

* [bls_utils.h](./src/bls_utils.h) is where all the utilities that
are not BLS dependent are located.

The source code contains extensive documentation describing the operation of
the backend and the use of the
[Triton Backend API](../../../README.md#triton-backend-api) and the
[Triton Server API](https://github.com/triton-inference-server/server/blob/main/docs/inference_protocols.md#in-process-triton-server-api).
Before reading the source code, make sure you understand
the concepts associated with Triton backend abstractions
[TRITONBACKEND_Backend](../../../README.md#tritonbackend_backend),
[TRITONBACKEND_Model](../../../README.md#tritonbackend_model), and
[TRITONBACKEND_ModelInstance](../../../README.md#tritonbackend_modelinstance).

The *bls* backend will send two requests on the 'addsub_python' and 'addsub_tf'
models. After the inference requests are completed, this backend will extract
OUTPUT0 from the 'addsub_python' and OUTPUT1 from the 'addsub_tf' model to
construct the final inference response object using these tensors.

There are some self-imposed limitations that were made for the simplicity of
this example:
1. This backend does not support batching.
1. This backend does not support decoupled models.
1. This backend does not support GPU tensors.
1. The model configuraion should be strictly set as the comments described in
[backend.cc](./src/backend.cc).

You can implement your custom backend that is not limited to the limitations
mentioned above.

## Building the *BLS* Backend

[backends/bls/CMakeLists.txt](CMakeLists.txt)
shows the recommended build and install script for a Triton
backend. Building and installing is the same as decribed in [Building
the *Minimal* Backend](../../README.md#building-the-minimal-backend).

## Running Triton with the *BLS* Backend

After adding the *bls* backend to the Triton server as
described in [Backend Shared
Library](../../../README.md#backend-shared-library), you can run Triton and
have it load the models in
[model_repos/bls_models](../../model_repos/bls_models). Assuming you have created a
*tritonserver* Docker image by adding the *bls* backend to Triton, the
following command will run Triton:

```
$ docker run --rm -it --net=host -v/path/to/model_repos/bls_models:/models tritonserver --model-repository=/models
```

The console output will show similar to the following indicating that
the *bls_fp32*, *addsub_python* and *addsub_tf* models from the bls_models repository have
loaded correctly.

```
I0616 09:34:47.767433 19214 server.cc:629] 
+---------------+---------+--------+
| Model         | Version | Status |
+---------------+---------+--------+
| addsub_python | 1       | READY  |
| addsub_tf     | 1       | READY  |
| bls_fp32      | 1       | READY  |
+---------------+---------+--------+
```

## Testing the *BLS* Backend

The [clients](../../clients) directory holds example clients. The
[bls_client](../../clients/bls_client) Python script demonstrates sending an
inference requests to the *bls* backend. With Triton running as
described in [Running Triton with the *BLS* Backend](#running-triton-with-the-bls-backend),
execute the client:

```
$ clients/bls_client
```

You should see an output similar to the output below:

```
INPUT0 ([0.42935285 0.51512766 0.43625894 ... 0.6670954  0.17747518 0.7976901 ]) + INPUT1 ([6.7752063e-01 2.4223252e-01 6.7743927e-01 ... 4.1531715e-01 2.5451833e-01 7.9097062e-01]) = OUTPUT0 ([1.1068735  0.75736016 1.1136982 ... 1.0824126  0.4319935  1.5886607 ])
INPUT0 ([0.42935285 0.51512766 0.43625894 ... 0.6670954  0.17747518 0.7976901 ]) - INPUT1 ([6.7752063e-01 2.4223252e-01 6.7743927e-01 ... 4.1531715e-01 2.5451833e-01 7.9097062e-01]) = OUTPUT1 ([-0.24816778  0.27289516 -0.24118033 ... 0.25177827 -0.07704315  0.00671947])

PASS
```
