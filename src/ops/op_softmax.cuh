#pragma once
#include <cmath>
#include "utils/tensor.cuh"

// Softmax operation
template <typename T>
void op_softmax(const Tensor<T>& input, Tensor<T>& output) {
    assert(input.h == output.h && input.w == output.w);

    for (int i = 0; i < input.h; ++i) {
        T max_val = -std::numeric_limits<T>::infinity();
        for (int j = 0; j < input.w; ++j) {
            max_val = std::max(max_val, Index(input, i, j));
        }

        T sum_exp = 0.0;
        for (int j = 0; j < input.w; ++j) {
            sum_exp += std::exp(Index(input, i, j) - max_val);
        }

        for (int j = 0; j < input.w; ++j) {
            Index(output, i, j) = std::exp(Index(input, i, j) - max_val) / sum_exp;
        }
    }
}

// GPU Kernel for softmax
template <typename T>
__global__ void softmax_kernel(const T* input, T* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ T shared_mem[];

    // Compute max value in the row
    T max_val = -std::numeric_limits<T>::infinity();
    for (int j = tid; j < width; j += blockDim.x) {
        max_val = max(max_val, input[row * width + j]);
    }

    // Reduce max value
    shared_mem[tid] = max_val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_mem[tid] = max(shared_mem[tid], shared_mem[tid + offset]);
        }
        __syncthreads();
    }

    max_val = shared_mem[0];

    // Compute sum of exponentials
    T sum_exp = 0.0;
    for (int j = tid; j < width; j += blockDim.x) {
        sum_exp += exp(input[row * width + j] - max_val);
    }

    // Reduce sum of exponentials
    shared_mem[tid] = sum_exp;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_mem[tid] += shared_mem[tid + offset];
        }
        __syncthreads();
    }

    sum_exp = shared_mem[0];

    // Compute softmax
    for (int j = tid; j < width; j += blockDim.x) {
        output[row * width + j] = exp(input[row * width + j] - max_val) / sum_exp;
    }
}

// Softmax operation on GPU
template <typename T>
void op_softmax_device(const Tensor<T>& input, Tensor<T>& output) {
    assert(input.h == output.h && input.w == output.w);

    int num_rows = input.h;
    int num_cols = input.w;
    int block_size = 256;
    int grid_size = num_rows;
    int shared_mem_size = block_size * sizeof(T);

    softmax_kernel<<<grid_size, block_size, shared_mem_size>>>(input.rawp, output.rawp, num_cols);
    cudaDeviceSynchronize();
}

// Dispatch function for softmax
template <typename T>
void op_softmax(const Tensor<T>& input, Tensor<T>& output) {
    if (input.on_device) {
        op_softmax_device(input, output);
    } else {
        op_softmax_host(input, output);
    }
}

template <typename T>
void op_softmax_host(const Tensor<T>& input, Tensor<T>& output) {
    assert(input.h == output.h && input.w == output.w);

    for (int i = 0; i < input.h; ++i) {
        T max_val = -std::numeric_limits<T>::infinity();
        for (int j = 0; j < input.w; ++j) {
            max_val = std::max(max_val, Index(input, i, j));
        }

        T sum_exp = 0.0;
        for (int j = 0; j < input.w; ++j) {
            sum_exp += std::exp(Index(input, i, j) - max_val);
        }

        for (int j = 0; j < input.w; ++j) {
            Index(output, i, j) = std::exp(Index(input, i, j) - max_val) / sum_exp;
        }
    }
}

