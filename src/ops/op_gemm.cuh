#pragma once
#include "utils/tensor.cuh"
#include <cublas_v2.h>

template <typename T>
void op_gemm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C, bool transA = false, bool transB = false) {
    assert(A.on_device && B.on_device && C.on_device);
    assert(transA ? A.w == C.h : A.h == C.h);
    assert(transB ? B.h == C.w : B.w == C.w);
    assert(transA ? A.h == B.w : A.w == B.h);

    int m = C.h;
    int n = C.w;
    int k = transA ? A.h : A.w;

    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    T alpha = 1.0f, beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSgemm(handle, opA, opB, m, n, k, &alpha, A.rawp, A.stride_h, B.rawp, B.stride_h, &beta, C.rawp, C.stride_h);

    cublasDestroy(handle);
}
