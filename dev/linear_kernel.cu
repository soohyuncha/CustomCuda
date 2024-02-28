#include <stdio.h>
#include <stdlib.h>
#include <math.h>

template <int NUM_THREADS>
__global__ void linear_kernel(float* A, float* W, float* y,
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

    if (tid == 0) {
        float sum = 0;
        for (int i = 0; i < NUM_THREADS; i++) {
            sum += psum_arr[i];
        }
        y[b_x*N + b_y] = sum;
    }
}

void linear(float* A, float* W, float* y,
        int M, int N, int K, int block_size
) {
    dim3 threads(1024, 1);
    dim3 grids(M, N);
                
    float* A_gpu;
    float* W_gpu;
    float* y_gpu;

    cudaMalloc((void**)&A_gpu, sizeof(float)*M*K);
    cudaMalloc((void**)&W_gpu, sizeof(float)*N*K);
    cudaMalloc((void**)&y_gpu, sizeof(float)*M*N);

    cudaMemcpy(A_gpu, A, sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(W_gpu, W, sizeof(float)*N*K, cudaMemcpyHostToDevice);

    linear_kernel<1024> <<<grids, threads>>> (
            A_gpu, W_gpu, y_gpu,
            M, N, K, block_size
    );

    cudaMemcpy(y, y_gpu, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

    cudaFree(A_gpu);
    cudaFree(W_gpu);
    cudaFree(y_gpu);
}

void print_matrix(float* mat, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%.2f ", mat[i*col + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int M = 256;
    int K = 1024;
    int N = 256;
    int block_size = 32;

    float A[M*K];
    float W[N*K];
    float y[M*N];
    float y_cpu[M*N];       // for answer check
    
    // Init
    srand(10000);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i*K + j] = (rand() / static_cast<float>(RAND_MAX) - 0.5);
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            W[i*K + j] = (rand() / static_cast<float>(RAND_MAX) - 0.5);
        }
    }

    //print_matrix(A, M, K);
    //print_matrix(W, N, K);


    // Linear() in cpu
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float psum = 0;
            for (int k = 0; k < K; k++) {
                psum += A[i*K + k] * W[j*K + k];
            }
            y_cpu[i*N + j] = psum;
        }
    }
    //print_matrix(y_cpu, M, N);

    linear(A, W, y, M, N, K, block_size);

    //print_matrix(y, M, N);

    float err = 0.001;
    int correct = 1;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(y_cpu[i*N+j] - y[i*N+j]) > err) {
                correct = 0;
                printf("[%d, %d] %.4f %.4f\n", i, j, y_cpu[i*N+j], y[i*N+j]);
            }
        }
    }

    if (correct) {
        printf("Result same\n");
    }
    else {
        printf("Result wrong\n");
    }

}
