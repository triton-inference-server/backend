<!--
# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Triton Inference Server Backend

A Triton *backend* is the implementation that executes a model. A
backend can be a wrapper around a deep-learning framework, like
PyTorch, TensorFlow, TensorRT or ONNX Runtime. Or a backend can be
custom C/C++ logic performing any operation (for example, image
pre-processing).

This repo contains documentation on Triton backends and also source,
scripts and utilities for creating Triton backends. You do not need to
use anything provided in this repo to create a Triton backend but you
will likely find its contents useful.

## Frequently Asked Questions

Full documentation is included below but these shortcuts can help you
get started in the right direction.

### Where can I ask general questions about Triton and Triton backends?

Be sure to read all the information below as well as the [general
Triton
documentation](https://github.com/triton-inference-server/server#triton-inference-server)
available in the main
[server](https://github.com/triton-inference-server/server) repo. If
you don't find your answer there you can ask questions on the main
Triton [issues
page](https://github.com/triton-inference-server/server/issues).

### Where can I find all the backends that are available for Triton?

Anyone can develop a Triton backend, so it isn't possible for us to
know about all available backends. But the Triton project does provide
a set of supported backends that are tested and updated with each
Triton release.

**TensorRT**: The TensorRT backend is used to execute TensorRT
models. The
[tensorrt_backend](https://github.com/triton-inference-server/tensorrt_backend)
repo contains the source for the backend.

**ONNX Runtime**: The ONNX Runtime backend is used to execute ONNX
models. The
[onnxruntime_backend](https://github.com/triton-inference-server/onnxruntime_backend)
repo contains the documentation and source for the backend.

**TensorFlow**: The TensorFlow backend is used to execute TensorFlow
models in both GraphDef and SavedModel formats. The same backend is
used to execute both TensorFlow 1 and TensorFlow 2 models. The
[tensorflow_backend](https://github.com/triton-inference-server/tensorflow_backend)
repo contains the documentation and source for the backend.

**PyTorch**: The PyTorch backend is used to execute TorchScript
models. The
[pytorch_backend](https://github.com/triton-inference-server/pytorch_backend)
repo contains the documentation and source for the backend.

**OpenVINO**: The OpenVINO backend is used to execute
[OpenVINO](https://docs.openvinotoolkit.org/latest/index.html)
models. The
[openvino_backend](https://github.com/triton-inference-server/openvino_backend)
repo contains the documentation and source for the backend.

**Python**: The Python backend allows you to write your model logic in
Python. For example, you can use this backend to execute pre/post
processing code written in Python, or to execute a PyTorch Python
script directly (instead of first converting it to TorchScript and
then using the PyTorch backend). The
[python_backend](https://github.com/triton-inference-server/python_backend)
repo contains the documentation and source for the backend.

**DALI**: [DALI](https://github.com/NVIDIA/DALI) is a collection of
highly optimized building blocks and an execution engine that
accelerates the pre-processing of the input data for deep learning
applications. The DALI backend allows you to execute your DALI
pipeline within Triton. The
[dali_backend](https://github.com/triton-inference-server/dali_backend)
repo contains the documentation and source for the backend.

**FIL**: The FIL ([Forest Inference
Library](https://github.com/rapidsai/cuml/tree/branch-21.10/python/cuml/fil))
backend is used to execute a variety of tree-based ML models, including
XGBoost models, LightGBM models, Scikit-Learn random forest models, and cuML
random forest models. The
[fil_backend](https://github.com/triton-inference-server/fil_backend) repo
contains the documentation and source for the backend.

**vLLM**: The vLLM backend is designed to run
[supported models](https://vllm.readthedocs.io/en/latest/models/supported_models.html)
on a [vLLM engine](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py).
This backend depends on [python_backend](https://github.com/triton-inference-server/python_backend)
to load and serve models. The
[vllm_backend](https://github.com/triton-inference-server/vllm_backend) repo
contains the documentation and source for the backend.

**Important Note!** Not all the above backends are supported on every platform
supported by Triton. Look at the
[Backend-Platform Support Matrix](docs/backend_platform_support_matrix.md)
to learn about the same.

### How can I develop my own Triton backend?

First you probably want to ask on the main Triton [issues
page](https://github.com/triton-inference-server/server/issues) to
make sure you are not duplicating a backend that already exists. Then
follow the [tutorial](examples/README.md) to learn how to create your
first simple Triton backend and incrementally improve it to add more
features. You should also read the complete documentation on [Triton
backends](#backends).

### Can I add (or remove) a backend to an existing Triton installation?

Yes. See [Backend Shared Library](#backend-shared-library) for general
information about how the shared library implementing a backend is
managed by Triton, and [Triton with Unsupported and Custom
Backends](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/compose.md#triton-with-unsupported-and-custom-backends)
for documentation on how to add your backend to the released Triton
Docker image. For a standard install the globally available backends
are in /opt/tritonserver/backends.

### What about backends developed using the "legacy custom backend" API.

The legacy custom API is removed from Triton. If you have custom
backends that you developed using this older API you must port them to
the new [Triton Backend API](#triton-backend-api).

## Backends

A Triton *backend* is the implementation that executes a model. A
backend can be a wrapper around a deep-learning framework, like
PyTorch, TensorFlow, TensorRT, ONNX Runtime or OpenVINO. A backend can
also implement any functionality you want as long as it adheres to the
[backend API](#triton-backend-api). Triton uses this API to send
requests to the backend for execution and the backend uses the API to
communicate with Triton.

Every model must be associated with a backend. A model's backend is
specified in the model's configuration using the `backend` setting.
For using TensorRT backend, the value of this setting should be `tensorrt`.
Similarly, for using PyTorch, ONNX and TensorFlow backends, the `backend`
field should be set to `pytorch`, `onnxruntime` or `tensorflow` respectively.
For all other backends, `backend` must be set to the name of the backend.
Some backends may also check the `platform` setting for categorizing the model,
for example, in TensorFlow backend, `platform` should be set to
`tensorflow_savedmodel` or `tensorflow_graphdef` according to the model format.
Please refer to the specific backend repository on whether `platform` is used.

### Backend Shared Library

Each backend must be implemented as a shared library and the name of
the shared library must be *libtriton_\<backend-name\>.so*. For
example, if the name of the backend is "mybackend", a model indicates
that it uses the backend by setting the model configuration 'backend'
setting to "mybackend", and Triton looks for *libtriton_mybackend.so*
as the shared library that implements the backend. The
[tutorial](examples/README.md) shows examples of how to build your
backend logic into the appropriate shared library.

For a model, *M* that specifies backend *B*, Triton searches for the
backend shared library in the following places, in this order:

* \<model_repository\>/M/\<version_directory\>/libtriton_B.so

* \<model_repository\>/M/libtriton_B.so

* \<global_backend_directory\>/B/libtriton_B.so

Where \<global_backend_directory\> is by default
/opt/tritonserver/backends.  The --backend-directory flag can be used
to override the default.

Typically you will install your backend into the global backend
directory. For example, if using Triton Docker images you can follow
the instructions in [Triton with Unsupported and Custom
Backends](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/compose.md#triton-with-unsupported-and-custom-backends). Continuing
the example of a backend names "mybackend", you would install into the
Triton image as:

```
/opt/
  tritonserver/
    backends/
      mybackend/
        libtriton_mybackend.so
        ... # other files needed by mybackend
```

### Triton Backend API

A Triton backend must implement the C interface defined in
[tritonbackend.h](https://github.com/triton-inference-server/core/tree/main/include/triton/core/tritonbackend.h). The
following abstractions are used by the API.

#### TRITONBACKEND_Backend

A TRITONBACKEND_Backend object represents the backend itself. The
same backend object is shared across all models that use the
backend. The associated API, like TRITONBACKEND_BackendName, is used
to get information about the backend and to associate a user-defined
state with the backend.

A backend can optionally implement TRITONBACKEND_Initialize and
TRITONBACKEND_Finalize to get notification of when the backend object
is created and destroyed (for more information see [backend
lifecycles](#backend-lifecycles)).

#### TRITONBACKEND_Model

A TRITONBACKEND_Model object represents a model. Each model loaded by
Triton is associated with a TRITONBACKEND_Model. Each model can use
the TRITONBACKEND_ModelBackend API to get the backend object
representing the backend that is used by the model.

The same model object is shared across all instances of that
model. The associated API, like TRITONBACKEND_ModelName, is used to
get information about the model and to associate a user-defined state
with the model.

Most backends will implement TRITONBACKEND_ModelInitialize and
TRITONBACKEND_ModelFinalize to initialize the backend for a given
model and to manage the user-defined state associated with the model
(for more information see [backend lifecycles](#backend-lifecycles)).

The backend must take into account threading concerns when
implementing TRITONBACKEND_ModelInitialize and
TRITONBACKEND_ModelFinalize.  Triton will not perform multiple
simultaneous calls to these functions for a given model; however, if a
backend is used by multiple models Triton may simultaneously call the
functions with a different thread for each model. As a result, the
backend must be able to handle multiple simultaneous calls to the
functions. Best practice for backend implementations is to use only
function-local and model-specific user-defined state in these
functions, as is shown in the [tutorial](examples/README.md).

#### TRITONBACKEND_ModelInstance

A TRITONBACKEND_ModelInstance object represents a model
*instance*. Triton creates one or more instances of the model based on
the *instance_group* settings specified in the model
configuration. Each of these instances is associated with a
TRITONBACKEND_ModelInstance object.

The only function that the backend must implement is
TRITONBACKEND_ModelInstanceExecute. The
TRITONBACKEND_ModelInstanceExecute function is called by Triton to
perform inference/computation on a batch of inference requests. Most
backends will also implement TRITONBACKEND_ModelInstanceInitialize
and TRITONBACKEND_ModelInstanceFinalize to initialize the backend for
a given model instance and to manage the user-defined state associated
with the model (for more information see [backend
lifecycles](#backend-lifecycles)).

The backend must take into account threading concerns when
implementing TRITONBACKEND_ModelInstanceInitialize,
TRITONBACKEND_ModelInstanceFinalize and
TRITONBACKEND_ModelInstanceExecute.  Triton will not perform multiple
simultaneous calls to these functions for a given model instance;
however, if a backend is used by a model with multiple instances or by
multiple models Triton may simultaneously call the functions with a
different thread for each model instance. As a result, the backend
must be able to handle multiple simultaneous calls to the
functions. Best practice for backend implementations is to use only
function-local and model-specific user-defined state in these
functions, as is shown in the [tutorial](examples/README.md).

#### TRITONBACKEND_Request

A TRITONBACKEND_Request object represents an inference request made
to the model. The backend takes ownership of the request object(s) in
TRITONBACKEND_ModelInstanceExecute and must release each request by
calling TRITONBACKEND_RequestRelease. However, the ownership of request
object is returned back to Triton in case TRITONBACKEND_ModelInstanceExecute
returns an error. See [Inference Requests and Responses](#inference-requests-and-responses)
for more information about request lifecycle.

The Triton Backend API allows the backend to get information about the
request as well as the input and request output tensors of the
request. Each request input is represented by a TRITONBACKEND_Input
object.

#### TRITONBACKEND_Response

A TRITONBACKEND_Response object represents a response sent by the
backend for a specific request. The backend uses the response API to
set the name, shape, datatype and tensor values for each output tensor
included in the response. The response can indicate either a failed or
a successful request. See [Inference Requests and
Responses](#inference-requests-and-responses) for more information
about request-response lifecycle.

#### TRITONBACKEND_BackendAttribute

A `TRITONBACKEND_BackendAttribute` allows a backend to set certain attributes which
are queried by Triton to inform certain feature support, preferred configurations, and
other types of backend-specific behavior.

When initializing a backend, Triton will query the `TRITONBACKEND_GetBackendAttribute` function
if implemented by the backend. This function is optional to implement, but is generally used to call
the related `TRITONBACKEND_BackendAttribute` APIs for setting backend-specific attributes.

Some of the relevant BackendAttribute setter APIs are listed below:
- `TRITONBACKEND_BackendSetExecutionPolicy`
- `TRITONBACKEND_BackendAttributeAddPreferredInstanceGroup`
    - Defines a priority list of instance groups to prefer for this backend if a model config doesn't explicitly define any instance groups.
- `TRITONBACKEND_BackendAttributeSetParallelModelInstanceLoading`
    - Defines whether the backend can safely handle concurrent calls to `TRITONBACKEND_ModelInstanceInitialize` or not.
    - Loading model instances in parallel can improve server startup times for large instance counts.
    - By default, this attribute is set to false, meaning that parallel instance loading is disabled for all backends unless explicitly enabled.
    - The following official backends currently support loading model instances in parallel:
        - Python
        - ONNXRuntime

The full list of `TRITONBACKEND_BackendAttribute` related APIs are defined in
[tritonbackend.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonbackend.h).

### Backend Lifecycles

A backend must carefully manage the lifecycle of the backend itself,
the models and model instances that use the backend and the inference
requests that execute on the model instances using the backend.

#### Backend and Model

Backend, model and model instance initialization is triggered when
Triton loads a model.

* If the model requires a backend that is not already in use by an
  already loaded model, then:

  * Triton [loads the shared library](#backend-shared-library) that
    implements the backend required by the model.

  * Triton creates the TRITONBACKEND_Backend object that represents
    the backend.

  * Triton calls TRITONBACKEND_Initialize if it is implemented in the
    backend shared library. TRITONBACKEND_Initialize should not return
    until the backend is completely initialized. If
    TRITONBACKEND_Initialize returns an error, Triton will report that
    the model failed to load.

* Triton creates the TRITONBACKEND_Model object that represents the
  model. Triton calls TRITONBACKEND_ModelInitialize if it is
  implemented in the backend shared library.
  TRITONBACKEND_ModelInitialize should not return until the backend
  is completely initialized for the model. If
  TRITONBACKEND_ModelInitialize returns an error, Triton will show
  that the model failed to load.

* For each model instance specified for the model in the model
  configuration:

  * Triton creates the TRITONBACKEND_ModelInstance object that
    represents the model instance.

  * Triton calls TRITONBACKEND_ModelInstanceInitialize if it is
    implemented in the backend shared library.
    TRITONBACKEND_ModelInstanceInitialize should not return until the
    backend is completely initialized for the instance. If
    TRITONBACKEND_ModelInstanceInitialize returns an error, Triton
    will show that the model failed to load.

Backend, model and model instance finalization is triggered when
Triton unloads a model.

* For each model instance:

  * Triton calls TRITONBACKEND_ModelInstanceFinalize if it is
    implemented in the backend shared library.
    TRITONBACKEND_ModelInstanceFinalize should not return until the
    backend is completely finalized, including stopping any threads
    create for the model instance and freeing any user-defined state
    created for the model instance.

  * Triton destroys the TRITONBACKEND_ModelInstance object that
    represents the model instance.

* Triton calls TRITONBACKEND_ModelFinalize if it is implemented in the
  backend shared library. TRITONBACKEND_ModelFinalize should not
  return until the backend is completely finalized, including stopping
  any threads create for the model and freeing any user-defined state
  created for the model.

* Triton destroys the TRITONBACKEND_Model object that represents the
  model.

* Even if no other loaded model requires the backend, Triton does not
  finalize and unload the backend until the tritonserver process is
  exiting. When the tritonserver process exits:

  * Triton calls TRITONBACKEND_Finalize if it is implemented in the
    backend shared library. TRITONBACKEND_ModelFinalize should not
    return until the backend is completely finalized, including
    stopping any threads create for the backend and freeing any
    user-defined state created for the backend.

  * Triton destroys the TRITONBACKEND_Backend object that represents
    the backend.

#### Inference Requests and Responses

Triton calls TRITONBACKEND_ModelInstanceExecute to execute inference
requests on a model instance. Each call to
TRITONBACKEND_ModelInstanceExecute communicates a batch of requests
to execute and the instance of the model that should be used to
execute those requests. The backend should not allow the caller
thread to return from TRITONBACKEND_ModelInstanceExecute until that
instance is ready to handle another set of requests. Typically this
means that the TRITONBACKEND_ModelInstanceExecute function will
create responses and release the requests before returning. However,
in case TRITONBACKEND_ModelInstanceExecute returns an error, the ownership
of requests is transferred back to Triton which will then be responsible
for releasing them. Therefore, in the case where TRITONBACKEND_ModelInstanceExecute
returns an error, the backend must not retain references to the requests
or access them in any way. For more detailed description of request/response
lifetimes, study the documentation of TRITONBACKEND_ModelInstanceExecute in
[tritonbackend.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonbackend.h).

##### Single Response

Most backends will create a single response for each request. For that
kind of backend, executing a single inference request requires the
following steps:

* Create a response for the request using TRITONBACKEND_ResponseNew.

* For each request input tensor use TRITONBACKEND_InputProperties to
  get shape and datatype of the input as well as the buffer(s)
  containing the tensor contents.

* For each output tensor which the request expects to be returned, use
  TRITONBACKEND_ResponseOutput to create the output tensor of the
  required datatype and shape. Use TRITONBACKEND_OutputBuffer to get a
  pointer to the buffer where the tensor's contents should be written.

* Use the inputs to perform the inference computation that produces
  the requested output tensor contents into the appropriate output
  buffers.

* Optionally set parameters in the response.

* Send the response using TRITONBACKEND_ResponseSend.

* Release the request using TRITONBACKEND_RequestRelease.

For a batch of requests the backend should attempt to combine the
execution of the individual requests as much as possible to increase
performance.

##### Decoupled Responses

It is also possible for a backend to send multiple responses for a
request or not send any responses for a request. A backend may also
send responses out-of-order relative to the order that the request
batches are executed. Such backends are called *decoupled* backends.
The decoupled backends use one `ResponseFactory` object per request to keep
creating and sending any number of responses for the request. For this
kind of backend, executing a single inference request typically requires
the following steps:

* For each request input tensor use TRITONBACKEND_InputProperties to
  get shape and datatype of the input as well as the buffer(s)
  containing the tensor contents.

* Create a `ResponseFactory` object for the request using
  TRITONBACKEND_ResponseFactoryNew.

  1. Create a response from the `ResponseFactory` object using
  TRITONBACKEND_ResponseNewFromFactory. As long as you have
  `ResponseFactory` object you can continue creating responses.

  2. For each output tensor which the request expects to be returned, use
  TRITONBACKEND_ResponseOutput to create the output tensor of the
  required datatype and shape. Use TRITONBACKEND_OutputBuffer to get a
  pointer to the buffer where the tensor's contents should be written.

  3. Use the inputs to perform the inference computation that produces
  the requested output tensor contents into the appropriate output
  buffers.

  4. Optionally set parameters in the response.

  5. Send the response using TRITONBACKEND_ResponseSend. If this is the
     last request then use TRITONSERVER_ResponseCompleteFlag with
     TRITONBACKEND_ResponseSend. Otherwise continue with Step 1 for
     sending next request

* Release the request using TRITONBACKEND_RequestRelease.

###### Special Cases

The decoupled API is powerful and supports various special cases:

* If the backend should not send any more responses for the request,
  TRITONBACKEND_ResponseFactorySendFlags can be used to send
  TRITONSERVER_RESPONSE_COMPLETE_FINAL using the `ResponseFactory`.

* The model can also send responses out-of-order in which it received
  requests.

* The backend can copy out the contents of the input buffer(s) if
  request is to be released before the contents are completely
  consumed to generate responses. After copy, the request can be
  released anytime before exiting TRITONBACKEND_ModelInstanceExecute.
  The copies and `ResponseFactory` object can be passed to a separate
  thread in backend. This means main caller thread can exit from
  TRITONBACKEND_ModelInstanceExecute and the backend can still continue
  generating responses as long as it holds `ResponseFactory` object.


The [repeat example](examples/README.md) demonstrates full power of
what can be achieved from decoupled API.


Study documentation of these TRITONBACKEND_* functions in
[tritonbackend.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonbackend.h)
for more details on these APIs. Read
[Decoupled Backends and Models](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/decoupled_models.md)
for more details on how to host a decoupled model.

## Build the Backend Utilities

The source in this repo builds into a single "backend utilities"
library that is useful when building backends. You don't need to use
these utilities but they will be helpful for most backends.

Typically you don't need to build this repo directly but instead you
can include it in the build of your backend as is shown in the
CMakeLists.txt files of the [tutorial examples](examples/README.md).

To build and install in a local directory use the following commands.

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
$ make install
```

The following required Triton repositories will be pulled and used in
the build. By default the "main" branch/tag will be used for each repo
but the listed CMake argument can be used to override.

* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]
* triton-inference-server/core: -DTRITON_CORE_REPO_TAG=[tag]

See the [CMakeLists.txt](CMakeLists.txt) file for other build options.

## Python-based Backends

Triton also provides an option to create [Python-based backends](docs/python_based_backends.md).
These backends should implement the
[`TritonPythonModel` interface](https://github.com/triton-inference-server/python_backend#usage),
which could be re-used as a backend by multiple models.
While the only required function is `execute`,
you may find it helpful to enhance your implementation by adding `initialize`,
`finalize`, and any other helper functions. For examples, please refer to
the [vLLM backend](https://github.com/triton-inference-server/vllm_backend),
which provides a common python script to serve models supported by vLLM.
