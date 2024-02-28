#include <torch/extension.h>

// CUDA forward declarations
torch::Tensor linear(
        torch::Tensor A,
        torch::Tensor W,
        int M, int N, int K, int block_size
);

// C++ interface
torch::Tensor linear_cpu(
        torch::Tensor A,
        torch::Tensor W,
        int M, int N, int K, int block_size
) {
    auto y = linear(A, W, M, N, K, block_size);
    return y;
}

// pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear", &linear_cpu, "Linear function using custom CUDA kernel");
}
