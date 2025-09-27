#include <string>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "vision_util_extensions.h"


namespace py = pybind11;

PYBIND11_MODULE(vision_util_cpp_extensions, m) {

    m.doc() = "pybind11 module for vision_util_extensions.cpp"; // optional module docstring

    m.def("pack_sta_buffer",
            &pack_sta_buffer,
            pybind11::return_value_policy::take_ownership,
            "Pack the byte buffer for a single STA.");
}
