#include <torch/extension.h>
#include <iostream>

template <typename scalar_t, int NUM_THREADS>
__global__ void linear_kernel(
//        float* A, float* W, float* y,
//        int M, int N, int K, int block_size
        torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> A,
        torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> W,
        torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> y,
        int M, int N, int K, int block_size
) {
    int b_x = blockIdx.x;       // row
    int b_y = blockIdx.y;       // col
    int tid = threadIdx.x;
    __shared__ float psum_arr[NUM_THREADS];
    // Init shared memory
    if (tid == 0) {
        for (int i = 0; i < NUM_THREADS; i++) {
            psum_arr[i] = 0;
        }
    }
    __syncthreads();

    int k_blk_idx = tid;
    int done = 0;
    while (1) {
        // Bound check
        if (done || k_blk_idx >= K/block_size) {
            break;
        }

        // Compute psum of a block
        for (int k = 0; k < block_size; k++) {
            if (k_blk_idx*block_size + k >= K) {
                done = 1;
                break;
            }
            psum_arr[tid] += A[b_x*K + k_blk_idx*block_size + k] * W[b_y*K + k_blk_idx*block_size + k];
        }

        k_blk_idx += NUM_THREADS;
    }
    __syncthreads();

    // Write result
    if (tid == 0) {
        float sum = 0;
        for (int i = 0; i < NUM_THREADS; i++) {
            sum += psum_arr[i];
        }
        y[b_x*N + b_y] = sum;
    }
}

torch::Tensor linear(
        torch::Tensor A,
        torch::Tensor W,
        int M, int N, int K, int block_size
) {
//void linear(
//        float* A, float* W, float* y,
//        int M, int N, int K, int block_size
//) {
    dim3 threads(1024, 1);
    dim3 grids(M, N);
    
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available. Exiting..." << std::endl;
        exit(0);
    }
    torch::Device device(torch::kCUDA);
    auto y = torch::zeros(M*N, torch::TensorOptions().device(device));

    AT_DISPATCH_FLOATING_TYPES(A.type(), "linear", ([&] {
                linear_kernel<scalar_t, 1024> <<<grids, threads>>> (
                        A.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
                        W.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
                        y.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
                        M, N, K, block_size
                );
    }));

    return y;
}
