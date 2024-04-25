#pragma once

#include "utils/assert.cuh"
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
    //    //int hh = C.h, ww = C.w; 
    //    //printf("%d , %d, %f, %f\n", i,j, C.h, C.w);

    //     __shared__ float sA[16][16];
    //     __shared__ float sB[16][16];
      
    //    T val = (T) 0.0;
      
    //   int thr = threadIdx.y, thc = threadIdx.x;

    //   int32_t width = A.w;
    //   for(int m=0;m < (width+16-1)/16; m++)
    //   {
    //     sA[thr][thc] = Index(A, i, m*16 + thc);
    //     sB[thr][thc] = Index(B, (m*16 + thr),j);

    //     __syncthreads();
        
    //     for(int k=0; k<16;k++)
    //         val += f( sA[thr][k], sB[k][thc] );
        
    //     __syncthreads();
    //   }
    //   Index(C, i, j)= val;

       if(!IndexOutofBound(C,i,j)){
            //printf("hello bey\n");
            T val=(T)0.0;
            for(int k=0;k<A.w;k++)
            {
                val+= f(Index(A,i,k), Index(B,k,j));
            }
            Index(C,i,j) = val;
       }
}
template <typename T>
void op_mm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C)
{
    assert(A.h == C.h && B.w == C.w && A.w == B.h);
    assert(A.on_device && B.on_device && C.on_device);
    //printf("kaisa hai\n");
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
