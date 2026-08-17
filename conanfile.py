from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeDeps, cmake_layout
from conan.errors import ConanInvalidConfiguration


class TritonBackendConan(ConanFile):
    name = "triton-backend"
    version = "2.68.0"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "enable_gpu": [True, False],
    }
    default_options = {
        "enable_gpu": True,
    }

    def validate(self):
        if self.settings.os != "Linux":
            raise ConanInvalidConfiguration("triton-backend only supports Linux")

    def requirements(self):
        self.requires("rapidjson/cci.20230929")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["TRITON_ENABLE_GPU"]             = self.options.enable_gpu
        tc.variables["TRITON_SKIP_THIRD_PARTY_FETCH"] = True
        tc.generate()
        CMakeDeps(self).generate()
