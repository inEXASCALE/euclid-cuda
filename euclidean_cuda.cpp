/*
 * PyTorch bindings for Euclidean distance CUDA kernels
 */
#include "euclidean_kernels.cuh"

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("pairwise_euclidean_single", &pairwise_euclidean_single, 
//           "Pairwise Euclidean distance (single precision)",
//           py::arg("P"), py::arg("C"));
//     
//     m.def("pairwise_euclidean_double", &pairwise_euclidean_double, 
//           "Pairwise Euclidean distance (double precision)",
//           py::arg("P"), py::arg("C"));
//     
//     m.def("pairwise_euclidean_fp32_fp64", &pairwise_euclidean_fp32_fp64, 
//           "Pairwise Euclidean distance (mixed precision with fallback)",
//           py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
// }


#include "euclidean_kernels.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fast pairwise Euclidean distance with mixed-precision";
    
    m.def("pairwise_euclidean_single", &pairwise_euclidean_single,
          "Pairwise Euclidean distance in single precision (FP32)",
          py::arg("P"), py::arg("C"));
    
    m.def("pairwise_euclidean_double", &pairwise_euclidean_double,
          "Pairwise Euclidean distance in double precision (FP64)",
          py::arg("P"), py::arg("C"));
}