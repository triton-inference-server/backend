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

# Backend-Platform Support Matrix

Even though Triton supports inference across various platforms such as
cloud, data center, edge and embedded devices on NVIDIA GPUs, x86 and
ARM CPU, or AWS Inferentia, it does so by relying on the backends.
Note that not all Triton backends support every platform. The purpose
of this document is to go over what all compute platforms are supported
by each of these Triton backends.
GPU in this document refers to Nvidia GPU. See
[GPU, Driver, and CUDA Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
to learn more about supported GPUs.

## Ubuntu 20.04

The table below describes target device(s) supported for inference by
each backend on different platforms.

| Backend      | x86       | ARM-SBSA      |
| ------------ | --------- | ------------- |
| TensorRT     |  :heavy_check_mark: GPU <br/> :x: CPU | :heavy_check_mark: GPU <br/> :x: CPU       |
| ONNX Runtime |  :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU  |   :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU      |
| TensorFlow   |  :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU  |   :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU      |
| PyTorch      |  :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU  |   :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU      |
| OpenVINO     |  :x: GPU <br/> :heavy_check_mark: CPU    |     :x: GPU <br/> :x: CPU       |
| Python[^1]   |  :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU  |  :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU  |
| DALI         |  :heavy_check_mark: GPU <br/> :x: CPU     |     :heavy_check_mark: GPU <br/> :x: CPU       |
| FIL          |  :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU  |  Unsupported  |



## Windows 10

Only TensorRT and ONNX Runtime backends are supported on Windows.

| Backend      | x86       | ARM-SBSA      |
| ------------ | --------- | ------------- |
| TensorRT     |  :heavy_check_mark: GPU <br/> :x: CPU | :heavy_check_mark: GPU <br/> :x: CPU       |
| ONNX Runtime |  :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU  |   :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU      |


## Jetson JetPack

Following backends are currently supported on Jetson Jetpack:

| Backend      |   Jetson  |
| ------------ | --------- |
| TensorRT     |  :heavy_check_mark: GPU <br/> :x: CPU    |
| ONNX Runtime |  :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU  |   :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU  |
| TensorFlow   |  :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU  |   :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU  |
| PyTorch      |  :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU  |   :heavy_check_mark: GPU <br/> :heavy_check_mark: CPU  |
| Python[^1]   |  :x: GPU <br/> :heavy_check_mark: CPU    |


Look at the [Triton Inference Server Support for Jetson and JetPack](https://github.com/triton-inference-server/server/blob/main/docs/jetson.md).


## AWS Inferentia

Currently, inference on AWS Inferentia is only supported via
[python backend](https://github.com/triton-inference-server/python_backend#running-with-inferentia)
where the deployed python script invokes AWS Neuron SDK.


[^1]: The supported devices for Python Backend are mentioned with
respect to Triton. The python script running in Python Backend can
be used to execute inference on any hardware if there are available
python APIs to do so. AWS inferentia is one such example. Triton
core is largely unaware of the fact that inference will run on
Inferentia.