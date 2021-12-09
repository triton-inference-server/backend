<!--
# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Triton Example Backends

To learn how to create a Triton backend, and to see a best-practices
baseline onto which you can add your own backend log, follow the
[Tutorial](#tutorial).

Triton also provides a couple of example backends that demonstrate
specific aspects of the backend API not covered by the
[Tutorial](#tutorial).

* The
[*repeat*](https://github.com/triton-inference-server/repeat_backend)
backend shows a more advanced example of how a backend can produce
multiple responses per request.

* The
[*stateful*](https://github.com/triton-inference-server/stateful_backend)
backend shows an example of how a backend can manage model state
tensors on the server-side for the [sequence
batcher](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#sequence-batcher)
to avoid transferring state tensors between client and server. Triton
also implements [Implicit State
Management](https://github.com/triton-inference-server/server/blob/main/docs/architecture.md#implicit-state-management)
which allows backends to behave in a stateless manner and leave the
state management to Triton.

## Tutorial

The [Triton Backend API](../README.md#triton-backend-api) exposes a
large number of features. The backend utilities and classes provide
many functions commonly used when creating a backend. But to create a
functional backend it is not necessary to use most of the backend API
or utilities. The tutorial starts with an implementation that shows a
minimal backend and then adds on recommended and optional
enhancements. The tutorial implementations follow best practices for
Triton backends and so can be used as templates for your own backend.

### Minimal Triton Backend

The source code for the *minimal* backend is contained in
[minimal.cc](backends/minimal/src/minimal.cc). The source code
contains extensive documentation describing the operation of the
backend and the use of the [Triton Backend
API](../README.md#triton-backend-api) and the backend
utilities. Before reading the source code, make sure you understand
the concepts associated with Triton backend abstractions
[TRITONBACKEND_Backend](../README.md#tritonbackend-backend),
[TRITONBACKEND_Model](../README.md#tritonbackend-model), and
[TRITONBACKEND_ModelInstance](../README.md#tritonbackend-modelinstance).

The *minimal* backend is complete but for clarity leaves out some
important aspects of writing a full-featured backend. When creating
your own backend use the [Recommended Triton
Backend](#recommended-triton-backend) as a starting point.

#### Building the Backend

[backends/minimal/CMakeLists.txt](backends/minimal/CMakeLists.txt)
shows the recommended build and install script for a Triton
backend. To build the minimal backend and install in a local directory
use the following commands.

```
$ cd backends/minimal
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
$ make install
```

The following required Triton repositories will be pulled and used in
the build. By default the "main" branch/tag will be used for each repo
but the listed CMake argument can be used to override.

* triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]
* triton-inference-server/core: -DTRITON_CORE_REPO_TAG=[tag]
* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]

If you are building on a release branch (or on a development branch
that is based off of a release branch), then you must set these cmake
arguments to point to that release branch as well. For example, if you
are building the r21.10 identity_backend branch then you need to use
the following additional cmake flags:

```
-DTRITON_BACKEND_REPO_TAG=r21.10
-DTRITON_CORE_REPO_TAG=r21.10
-DTRITON_COMMON_REPO_TAG=r21.10
```

After building the install directory will contain a backends/minimal
directory that contains the minimal backend. Instructions for adding
this backend to the Triton server are described in [Backend Shared
Library](../README.md#backend-shared-library).

#### Running Triton with the Minimal Backend

After adding the *minimal* backend to the Triton server as described
in [Backend Shared Library](../README.md#backend-shared-library), you
can run Triton and have it load the models in
[model_repos/minimal_models](model_repos/minimal_models). Assuming you
have created a *tritonserver* Docker image by adding the *minimal*
backend to Triton, the following command will run Triton:at the cmd
prompt inside your Triton Docker container:

```
$ docker run --rm -it --net=host -v/path/to/model_repos/minimal_models:/models tritonserver --model-repository=/models
```

The console output will show similar to the following indicating that
the *batching* and *nonbatching* models from the minimal_models
repository have loaded correctly. Note that the model repository has
two models that both use the *minimal* backend. A backend can support
any number of diffent models.

```
I1215 23:46:00.250284 68 server.cc:589]
+-------------+---------+--------+
| Model       | Version | Status |
+-------------+---------+--------+
| batching    | 1       | READY  |
| nonbatching | 1       | READY  |
+-------------+---------+--------+
```

The models are identical except that the *batching* model enabled the
[dynamic
batcher](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#dynamic-batcher)
and supports batch sizes up to 8. Note that the *batching* model sets
the [batch
delay](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#delayed-batching)
to 5 seconds so that the example client described below can
demonstrate how the *minimal* backend receives a batch of requests.

#### Testing the Backend

The [clients](clients) directory holds example clients. The
[minimal_client](clients/minimal_client) Python script demonstrates
sending a couple of inference requests to the *minimal* backend. With
Triton running as described in [Running Triton with the Minimal
Backend](#running-triton-with-the-minimal-backend), execute the
client:

```
$ clients/minimal_client
```

The minimal_client first sends a single request to nonbatching
model. From the output you can see that the input value is returned in
the output.

```
=========
Sending request to nonbatching model: IN0 = [1 2 3 4]
Response: {'model_name': 'nonbatching', 'model_version': '1', 'outputs': [{'name': 'OUT0', 'datatype': 'INT32', 'shape': [4], 'parameters': {'binary_data_size': 16}}]}
OUT0 = [1 2 3 4]
```

In the Triton console output you can see the log message printed by
the *minimal* backend that indicates that it received a batch
containing the single request.

```
I1216 23:34:28.779692 417 minimal.cc:378] model nonbatching: requests in batch 1
I1216 23:34:28.779713 417 minimal.cc:386] batched IN0 value: [ 1, 2, 3, 4 ]
```

The minimal_client next sends 2 requests at the same time to the
batching model. Triton will dynamically batch those requests into a
single batch and send that single batch to the *minimal* backend.

```
=========
Sending request to batching model: IN0 = [[10 11 12 13]]
Sending request to batching model: IN0 = [[20 21 22 23]]
Response: {'model_name': 'batching', 'model_version': '1', 'outputs': [{'name': 'OUT0', 'datatype': 'INT32', 'shape': [1, 4], 'parameters': {'binary_data_size': 16}}]}
OUT0 = [[10 11 12 13]]
Response: {'model_name': 'batching', 'model_version': '1', 'outputs': [{'name': 'OUT0', 'datatype': 'INT32', 'shape': [1, 4], 'parameters': {'binary_data_size': 16}}]}
OUT0 = [[20 21 22 23]]
```

In the Triton console output you can see the log message indicating
that the *minimal* backend received a batch containing both requests.

```
I1216 23:39:46.479728 460 minimal.cc:376] model batching: requests in batch 2
I1216 23:39:46.479770 460 minimal.cc:384] batched IN0 value: [ 10, 11, 12, 13, 20, 21, 22, 23 ]
```

### Recommended Triton Backend

Under construction.

### Enhancements

Under construction.
