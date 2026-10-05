<!--
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Security Policy

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

To report a potential security vulnerability in this repository or any other
NVIDIA product, use one of the following channels:

1. **NVIDIA Vulnerability Disclosure Program** (preferred):
   [https://www.nvidia.com/en-us/security/](https://www.nvidia.com/en-us/security/)
2. **Email**: [psirt@nvidia.com](mailto:psirt@nvidia.com). Please encrypt
   sensitive reports with NVIDIA's public PGP key
   ([PGP key page](https://www.nvidia.com/en-us/security/pgp-key)).
3. **GitHub Private Vulnerability Reporting**: use the **Security** tab of this
   repository, if enabled.

OEM partners should contact their NVIDIA Customer Program Manager.

Please include:

1. Product and version or branch that contains the vulnerability
2. Type of vulnerability (for example code execution, denial of service,
   buffer overflow)
3. Step-by-step instructions to reproduce the issue
4. Proof-of-concept or exploit code, if available
5. Potential impact, including how an attacker could exploit the issue

NVIDIA PSIRT acknowledges reports, assesses severity, coordinates a fix and
disclosure timeline with the reporter, and publishes security bulletins at
[https://www.nvidia.com/en-us/security/](https://www.nvidia.com/en-us/security/).

## Security Architecture and Context

**Project:** Triton Inference Server Backend. This repository provides the
backend API documentation, the C++ utility library built from `src/` and
`include/triton/backend/` (input collection, output responding, memory
management, model and instance helpers, common parsing and file utilities),
and example backends under `examples/`.

**Classification:** Library / SDK. It is compiled into, and runs inside the
process of, backend implementations that are loaded by Triton Inference
Server. It does not open network listeners, implement authentication, or
store data itself.

**Repository Exposure Classification:** Public (repository visibility is public
on GitHub).

**Service Exposure Classification:** Not determined (low confidence). The
exposure depends on how the embedding server and backend are deployed.

**Primary security responsibility:** memory safety and correct bounds handling
when copying tensor data between request, response, host, pinned and CUDA
device memory, and safe handling of model configuration values and file paths
passed in by the server.

**Key interfaces and boundaries:**

- The `TRITONBACKEND_*` C API between the server and a backend. Requests,
  tensor shapes and byte sizes originate from clients of the server.
- Model configuration (`config.pbtxt` / JSON), parsed in `src/backend_common.cc`
  (for example shape, batch input and batch output parsing).
- Local file helpers in `src/backend_common.cc` (`ReadTextFile`, `FileExists`,
  `IsDirectory`, directory listing) that operate on paths supplied by the
  caller.
- Host and device memory paths in `src/backend_memory.cc`,
  `src/backend_input_collector.cc` and `src/backend_output_responder.cc`,
  including `memcpy`, CUDA copy calls and CUDA kernels (`src/kernel.cu`).

## Threat Model

1. **Out-of-bounds read or write in tensor copy paths:** a request whose
   declared shape, batch size or byte size does not match the supplied buffer
   could cause the input collector or output responder
   (`backend_input_collector.cc`, `backend_output_responder.cc`,
   `backend_common.cc` copy helpers) to read or write past a buffer.
2. **Integer overflow in size calculations:** large dimensions or batch sizes
   multiplied into byte counts could wrap and produce undersized allocations
   followed by oversized copies.
3. **Malformed model configuration:** crafted shape, batch input or batch output
   entries could trigger parsing errors, unchecked indexing or excessive
   allocation in the `ParseShape` and `BatchInput` / `BatchOutput` parsing
   code.
4. **Unbounded file reads and path handling:** `ReadTextFile` sizes its buffer
   from the file length and performs no path restriction, so a caller that
   passes an attacker-influenced path could read unintended files or exhaust
   memory.
5. **Resource exhaustion through memory allocation:** repeated or very large
   pinned, host or CUDA allocations by the memory helpers could degrade or deny
   service to the hosting server.
6. **Unsafe example code reused in production:** the example backends and clients
   are illustrative and omit hardening; copying them unchanged could carry
   missing validation into a deployed backend.
7. **Build and supply chain:** CMake configuration pulls in the common and core
   repositories and, optionally, CUDA toolchain components; unpinned or
   unverified sources could alter the built library. Both repositories are
   fetched from `main` by default; pin them with `TRITON_COMMON_REPO_TAG` and
   `TRITON_CORE_REPO_TAG`.

## Critical Security Assumptions

- The Triton Inference Server core validates request metadata. Authentication,
  authorization and TLS are deployment responsibilities, not guaranteed checks:
  the documented example server command and the example clients use plain HTTP
  on `localhost:8000` with no credentials. This library assumes the deployer
  has enforced these controls before requests reach a backend, and that callers
  are already authenticated.
- Byte sizes, shapes and buffer pointers handed in through the
  `TRITONBACKEND_*` API are assumed consistent with each other; this library
  does not independently re-verify them in every helper.
- Model configuration files and the model repository are assumed to be
  trusted, administrator-controlled content.
- File paths passed to the file helpers are assumed to be trusted and are not
  sandboxed.
- The operating system, GPU driver and CUDA runtime are assumed to provide
  correct memory isolation between processes.
- Backends built on this library are assumed to be run in a deployment where
  the hosting server is not directly exposed to untrusted networks without
  appropriate network controls.
- Example code in `examples/` is assumed to be used for learning and testing
  only.
