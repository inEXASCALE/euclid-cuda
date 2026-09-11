/*
 * Python bindings for uniform precision kernels
 */
#include "euclidean_kernels_uniform.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Uniform precision Euclidean distance (no fallback)";
    
    m.def("pairwise_euclidean_uniform_fp16", &pairwise_euclidean_uniform_fp16,
          "Pure FP16 Euclidean distance (no fallback)",
          py::arg("P"), py::arg("C"));
    
    m.def("pairwise_euclidean_uniform_bf16", &pairwise_euclidean_uniform_bf16,
          "Pure BF16 Euclidean distance (no fallback)",
          py::arg("P"), py::arg("C"));
    
    m.def("pairwise_euclidean_uniform_tf32", &pairwise_euclidean_uniform_tf32,
          "Pure TF32 Euclidean distance (no fallback)",
          py::arg("P"), py::arg("C"));

    // Direct formula (no GEMM, more stable for near points)
    m.def("pairwise_euclidean_uniform_fp16_direct", &pairwise_euclidean_uniform_fp16_direct,
          "Pure FP16 Euclidean distance using direct formula sum((p-c)^2)",
          py::arg("P"), py::arg("C"));
    
    m.def("pairwise_euclidean_uniform_bf16_direct", &pairwise_euclidean_uniform_bf16_direct,
          "Pure BF16 Euclidean distance using direct formula",
          py::arg("P"), py::arg("C"));
    
    m.def("pairwise_euclidean_uniform_tf32_direct", &pairwise_euclidean_uniform_tf32_direct,
          "Pure TF32/FP32 Euclidean distance using direct formula",
          py::arg("P"), py::arg("C"));


    m.def("pairwise_euclidean_uniform_fp32_direct", &pairwise_euclidean_uniform_fp32_direct,
          "Pure FP32 Euclidean distance using direct formula",
          py::arg("P"), py::arg("C"));
    
    m.def("pairwise_euclidean_uniform_fp64_direct", &pairwise_euclidean_uniform_fp64_direct,
          "Pure FP64 Euclidean distance using direct formula",
          py::arg("P"), py::arg("C"));
}