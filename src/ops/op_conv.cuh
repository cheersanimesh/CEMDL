#pragma once

#include "utils/tensor.cuh"

//This operator compute C = A@B
template <typename T>
class MultiplyFunc2
{
public:
    __host__ __device__ T operator()(T x, T a)
    {
        //Lab-1: add your code here (delete return 0)
        return x * a;
    }
};

template <typename OpFunc, typename T>
__global__ void mat_mul_kernel(OpFunc f, const Tensor<T> A, const Tensor<T> B, Tensor<T> C)
{
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;

       if(!IndexOutofBound(C,i,j)){
            T val=(T)0.0;
            for(int k=0;k<A.w;k++)
            {
                val+= f(Index(A,i,k), Index(B,k,j));
            }
            Index(C,i,j) = val;
       }
}
template <typename T>
void op_conv(const Tensor<T>& input, const Tensor<T>& weights, Tensor<T>& conv_output)
{
    assert(A.h == C.h && B.w == C.w && A.w == B.h);
    assert(A.on_device && B.on_device && C.on_device);
    //Lab-1: please complete this
    //You need to define separate kernel function(s) and launch them here
    //delete assert(0) when you are finished
    int32_t N1 = A.h, M2 = B.w;
    
    dim3 threadsPerBlock(16,16);

    MultiplyFunc2<T> f;
    dim3 numBlocks( 
        (M2+ threadsPerBlock.x-1)/threadsPerBlock.x, 
        (N1+threadsPerBlock.y-1)/threadsPerBlock.y 
    );
    //mat_mul_kernel<<<numBlocks, threadsPerBlock>>>(f, A, B, C); 
    mat_mul_kernel<<<numBlocks, threadsPerBlock>>>(f, A, B, C); 
    //assert(0);
}
