<!--
# Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
*minimal* backend and then adds on recommended and optional
enhancements. The tutorial implementations follow best practices for
Triton backends and so can be used as templates for your own backend.

### *Minimal* Triton Backend

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

The *minimal* backend does not do any interesting operation, it simply
copies a single input tensor to a single output tensor, but it does
demonstrate the basic organization required for a Triton backend.

The *minimal* backend is complete but for clarity leaves out some
important aspects of writing a full-featured backend that are
described in [*Recommended* Triton
Backend](#recommended-triton-backend). When creating your own backend
use the [*Recommended* Triton Backend](#recommended-triton-backend) as
a starting point.

#### Building the *Minimal* Backend

[backends/minimal/CMakeLists.txt](backends/minimal/CMakeLists.txt)
shows the recommended build and install script for a Triton
backend. To build the *minimal* backend and install in a local directory
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
directory that contains the *minimal* backend. Instructions for adding
this backend to the Triton server are described in [Backend Shared
Library](../README.md#backend-shared-library).

#### Running Triton with the *Minimal* Backend

After adding the *minimal* backend to the Triton server as described
in [Backend Shared Library](../README.md#backend-shared-library), you
can run Triton and have it load the models in
[model_repos/minimal_models](model_repos/minimal_models). Assuming you
have created a *tritonserver* Docker image by adding the *minimal*
backend to Triton, the following command will run Triton:

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

#### Testing the *Minimal* Backend

The [clients](clients) directory holds example clients. The
[minimal_client](clients/minimal_client) Python script demonstrates
sending a couple of inference requests to the *minimal* backend. With
Triton running as described in [Running Triton with the *Minimal*
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
I1221 18:14:12.964836 86 minimal.cc:348] model nonbatching: requests in batch 1
I1221 18:14:12.964857 86 minimal.cc:356] batched IN0 value: [ 1, 2, 3, 4 ]
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
I1221 18:14:17.965982 86 minimal.cc:348] model batching: requests in batch 2
I1221 18:14:17.966035 86 minimal.cc:356] batched IN0 value: [ 10, 11, 12, 13, 20, 21, 22, 23 ]
```

### *Recommended* Triton Backend

The source code for the *recommended* backend is contained in
[recommended.cc](backends/recommended/src/recommended.cc). The source
code contains extensive documentation describing the operation of the
backend and the use of the [Triton Backend
API](../README.md#triton-backend-api) and the backend
utilities. Before reading the source code, make sure you understand
the concepts associated with Triton backend abstractions
[TRITONBACKEND_Backend](../README.md#tritonbackend-backend),
[TRITONBACKEND_Model](../README.md#tritonbackend-model), and
[TRITONBACKEND_ModelInstance](../README.md#tritonbackend-modelinstance).

The *recommended* backend improves the [*minimal*
backend](#minimal-triton-backend) to include the following features
which should be present in any robust backend implementation:

* Enhances the backend to support models with input/output tensors
  that have datatypes other than INT32.

* Enhances the backend to support models with input/output tensors
  that have any shape.

* Uses the Triton backend metric APIs to record statistics about
  requests executing in the backend. These metrics can then we queried
  using the Triton
  [metrics](https://github.com/triton-inference-server/server/blob/main/docs/metrics.md)
  and
  [statistics](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_statistics.md)
  APIs.

* Additional error checking to ensure that the backend's version is
  compatible with Triton and that each model's configuration is
  compatible with the backend.

As with the *minimal* backend, the *recommended* backend just returns
the input tensor value in the output tensor. Because of the additions
described above, the *recommended* backend can serve as a starting
point for your backend.

#### Building the *Recommended* Backend

[backends/recommended/CMakeLists.txt](backends/recommended/CMakeLists.txt)
shows the recommended build and install script for a Triton
backend. Building and installing is the same as decribed in [Building
the *Minimal* Backend](#building-the-minimal-backend).

#### Running Triton with the *Recommended* Backend

After adding the *recommended* backend to the Triton server as
described in [Backend Shared
Library](../README.md#backend-shared-library), you can run Triton and
have it load the models in
[model_repos/recommended_models](model_repos/recommended_models). Assuming
you have created a *tritonserver* Docker image by adding the
*recommended* backend to Triton, the following command will run
Triton:

```
$ docker run --rm -it --net=host -v/path/to/model_repos/recommended_models:/models tritonserver --model-repository=/models
```

The console output will show similar to the following indicating that
the *batching* model from the recommended_models repository have
loaded correctly.

```
I1215 23:46:00.250284 68 server.cc:589]
+-------------+---------+--------+
| Model       | Version | Status |
+-------------+---------+--------+
| batching    | 1       | READY  |
+-------------+---------+--------+
```

#### Testing the *Recommended* Backend

The [clients](clients) directory holds example clients. The
[recommended_client](clients/recommended_client) Python script
demonstrates sending a couple of inference requests to the
*recommended* backend. With Triton running as described in [Running
Triton with the *Recommended*
Backend](#running-triton-with-the-recommended-backend), execute the
client:

```
$ clients/recommended_client
```

The recommended_client next sends 2 requests at the same time to the
batching model, similar to what was done above with the *minimal*
backend. Triton will dynamically batch those requests into a single
batch and send that single batch to the *recommended* backend. In this
model, batching is supported, the datatype is FP32 and the tensor
shape is [ -1, 4, 4 ].

```
=========
Sending request to batching model: input = [[[1.  1.1 1.2 1.3]
  [2.  2.1 2.2 2.3]
  [3.  3.1 3.2 3.3]
  [4.  4.1 4.2 4.3]]]
Sending request to batching model: input = [[[10.  10.1 10.2 10.3]
  [20.  20.1 20.2 20.3]
  [30.  30.1 30.2 30.3]
  [40.  40.1 40.2 40.3]]]
Response: {'model_name': 'batching', 'model_version': '1', 'outputs': [{'name': 'OUTPUT', 'datatype': 'FP32', 'shape': [1, 4, 4], 'parameters': {'binary_data_size': 64}}]}
OUTPUT = [[[1.  1.1 1.2 1.3]
  [2.  2.1 2.2 2.3]
  [3.  3.1 3.2 3.3]
  [4.  4.1 4.2 4.3]]]
Response: {'model_name': 'batching', 'model_version': '1', 'outputs': [{'name': 'OUTPUT', 'datatype': 'FP32', 'shape': [1, 4, 4], 'parameters': {'binary_data_size': 64}}]}
OUTPUT = [[[10.  10.1 10.2 10.3]
  [20.  20.1 20.2 20.3]
  [30.  30.1 30.2 30.3]
  [40.  40.1 40.2 40.3]]]
```

In the Triton console output you can see the log message indicating
that the *recommended* backend received a batch containing both
requests.

```
I1221 18:30:52.223226 127 recommended.cc:604] model batching: requests in batch 2
I1221 18:30:52.223313 127 recommended.cc:613] batched INPUT value: [ 1.000000, 1.100000, 1.200000, 1.300000, 2.000000, 2.100000, 2.200000, 2.300000, 3.000000, 3.100000, 3.200000, 3.300000, 4.000000, 4.100000, 4.200000, 4.300000, 10.000000, 10.100000, 10.200000, 10.300000, 20.000000, 20.100000, 20.200001, 20.299999, 30.000000, 30.100000, 30.200001, 30.299999, 40.000000, 40.099998, 40.200001, 40.299999 ]
```

Because the *recommended* backend can support models that have
input/output tensors with any datatype and shape, you can edit the
model configuration and the client to experiment with these options.

To see the metrics collected for these two inference requests, use the following command to access Triton's metrics endpoint.

```
$ curl localhost:8002/metrics
```

The output will be metric values in Prometheus data format. The
[metrics
documentation](https://github.com/triton-inference-server/server/blob/main/docs/metrics.md)
gives a description of these metric values.

```
# HELP nv_inference_request_success Number of successful inference requests, all batch sizes
# TYPE nv_inference_request_success counter
nv_inference_request_success{model="batching",version="1"} 2.000000
# HELP nv_inference_request_failure Number of failed inference requests, all batch sizes
# TYPE nv_inference_request_failure counter
nv_inference_request_failure{model="batching",version="1"} 0.000000
# HELP nv_inference_count Number of inferences performed
# TYPE nv_inference_count counter
nv_inference_count{model="batching",version="1"} 2.000000
# HELP nv_inference_exec_count Number of model executions performed
# TYPE nv_inference_exec_count counter
nv_inference_exec_count{model="batching",version="1"} 1.000000
...
```

You can also see the collected statistics using the [statistics
endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_statistics.md).

```
$ curl localhost:8000/v2/models/batching/stats
{"model_stats":[{"name":"batching","version":"1","last_inference":1640111452223,"inference_count":2,"execution_count":1,"inference_stats":{"success":{"count":2,"ns":9997025869},"fail":{"count":0,"ns":0},"queue":{"count":2,"ns":9996491319},"compute_input":{"count":2,"ns":95288},"compute_infer":{"count":2,"ns":232202},"compute_output":{"count":2,"ns":195850}},"batch_stats":[{"batch_size":2,"compute_input":{"count":1,"ns":47644},"compute_infer":{"count":1,"ns":116101},"compute_output":{"count":1,"ns":97925}}]}]}
```

### *BLS* Triton Backend

Please see the [doucumentation](backends/bls/README.md) of *BLS* Backend.

### Enhancements

This section describes several optional features that you can add to
enhance the capabilities of your backend.

#### Automatically Model Configuration Generation

[Automatic model configuration
generation](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#auto-generated-model-configuration)
is enabled by the backend implementing the appropriate logic (for
example, in a function called AutoCompleteConfig) during
TRITONBACKEND_ModelInitialize. For the *recommended* backend you would
add a call to AutoCompleteConfig in the ModelState constructor just
before the call to ValidateModelConfig. The AutoCompleteConfig
function can update the model configuration with input tensor, output
tensor, and max-batch-size configuration; and then update the
configuration using TRITONBACKEND_ModelSetConfig. Examples can be
found in [ONNXRuntime
backend](https://github.com/triton-inference-server/onnxruntime_backend),
[TensorFlow
backend](https://github.com/triton-inference-server/tensorflow_backend)
and other backends.

#### Add Key-Value Parameters to a Response

A backend can add a key-value pair to a response any time after the
response is created and before it is sent. The parameter key must be a
string and the parameter value can be a string, integer or
boolean. The following example shows the TRITONBACKEND API used to set
response parameters. Error checking code is not shown to improve
clarity.

```
TRITONBACKEND_ResponseSetStringParameter(response, "param0", "an example string parameter");
TRITONBACKEND_ResponseSetIntParameter(responses[r], "param1", 42);
TRITONBACKEND_ResponseSetBoolParameter(responses[r], "param2", false);
```

#### Access Model Artifacts in the Model Repository

A backend can access any of the files in a model's area of the model
registry. These files are typically needed during
TRITONBACKEND_ModelInitialize but can be accessed at other times as
well. The TRITONBACKEND_ModelRepository API gives the location of the
model's repository. For example, the following code can be run during
TRITONBACKEND_ModelInitialize to write the location to the log.

```
// Can get location of the model artifacts. Normally we would need
// to check the artifact type to make sure it was something we can
// handle... but we are just going to log the location so we don't
// need the check. We would use the location if we wanted to load
// something from the model's repo.
TRITONBACKEND_ArtifactType artifact_type;
const char* clocation;
RETURN_IF_ERROR(
    TRITONBACKEND_ModelRepository(model, &artifact_type, &clocation));
LOG_MESSAGE(
    TRITONSERVER_LOG_INFO,
    (std::string("Repository location: ") + clocation).c_str());
```

The framework backends (for example, TensorRT, ONNXRuntime,
TensorFlow, PyTorch) read the actual model file from the model
repository using this API. See those backends for examples of how it
can be used.
