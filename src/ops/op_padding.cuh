#pragma once
#include "utils/tensor3D.cuh"
#include "utils/tensor.cuh"



template <typename T>

__global__ void padding_kernel(const Tensor3D<T> input, Tensor3D<T> output, int padding)
{
	int chan = blockIdx.x * blockDim.x + threadIdx.y;

	int num_channels = input.d;

	if(chan < num_channels){
		
		for(int i=0;i<input.h, i++)
			for(int j=0;j<input.w;j++)
				Index(output, i+padding, j+padding) = Index(input, i, j);
	}
	
}
// template <typename OpFunc, typename T>
// __global__ void mat_mul_kernel(OpFunc f, const Tensor<T> A, const Tensor<T> B, Tensor<T> C)
// {
//         int i = blockIdx.y * blockDim.y + threadIdx.y;
//         int j = blockIdx.x * blockDim.x + threadIdx.x;

//        if(!IndexOutofBound(C,i,j)){
//             T val=(T)0.0;
//             for(int k=0;k<A.w;k++)
//             {
//                 val+= f(Index(A,i,k), Index(B,k,j));
//             }
//             Index(C,i,j) = val;
//        }
// }
template <typename T>
void op_padding(const Tensor3D<T> &input, const Tensor3D<T>& output, int padding)
{
    assert(input.d == output.d);

    assert(input.on_device && output.on_device);
    //Lab-1: please complete this
    //You need to define separate kernel function(s) and launch them here
    //delete assert(0) when you are finished
    // int32_t N1 = A.h, M2 = B.w;
    
    // dim3 threadsPerBlock(16,16);

    // MultiplyFunc2<T> f;
    // dim3 numBlocks( 
    //     (M2+ threadsPerBlock.x-1)/threadsPerBlock.x, 
    //     (N1+threadsPerBlock.y-1)/threadsPerBlock.y 
    // );
    // //mat_mul_kernel<<<numBlocks, threadsPerBlock>>>(f, A, B, C); 
    // mat_mul_kernel<<<numBlocks, threadsPerBlock>>>(f, A, B, C); 
    //assert(0);

	int num_channels = input.d;
	
	dim3 threadsPerBlock(16);

	dim3 numBlocks((num_channels+threadsPerBlock.x-1)/threadsPerBlock.x);

	padding_kernel<<<numBlocks, threadsPerBlock>>>(input, output, padding);


}
