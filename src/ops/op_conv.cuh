#pragma once

#include "utils/tensor.cuh"

//This operator compute C = A@B
template <typename T>
class MultiplyFunc
{
    public:
        __host__ __device__ T operator()(T x, T a)
        {
            //Lab-1: add your code here (delete return 0)
            return x * a;
        }
};

template <typename OpFunc, typename T>
__global__ void conv_kernel(const Tensor3D<T> input, Tensor3D<T> weights, Tensor<T> output, int stride, int padding)
{
        int out_x = blockIdx.x * blockDim.x + threadIdx.x;
        int out_y = blockIdx.y * blockDim.y + threadIdx.y;
        int channel = blockIdx.z * blockDim.z + threadIdx.z;

        int kernel_radius_x = kernel_width / 2;
        int kernel_radius_y = kernel_height / 2;

        if (channel >= channels) return;

        float value = 0.0;

        int in_x_origin = (out_x * stride) - padding;
        int in_y_origin = (out_y * stride) - padding;

        // if (out_x >= (width + 2 * pad_x - kernel_width + stride_x) / stride_x ||
        //     out_y >= (height + 2 * pad_y - kernel_height + stride_y) / stride_y)
        //     return;

        if(!IndexOutofBound(out_x, out_y)){

            for (int i = 0; i < kernel_width; i++) {
                for (int j = 0; j < kernel_height; j++) {
                    int in_x = in_x_origin + i;
                    int in_y = in_y_origin + j;

                    if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                        int input_index = (in_y * width + in_x) * channels + channel;
                        int kernel_index = (j * kernel_width + i) * channels + channel;
                        value += input[input_index] * kernel[kernel_index];
                    }
                }
            }

            int output_index = ((out_y * ((width + 2 * pad_x - kernel_width + stride_x) / stride_x) + out_x) * channels) + channel;
            output[output_index] = value;
        }
}
template <typename T>
void op_conv(const Tensor3D<T>& input, const Tensor3D<T>& weights, Tensor<T>& output, int stride, int padding)
{
    int32_t val1 = (input.h - weights.h + 2 * padding+stride)/ stride;
    int32_t val2 = (input.w - weights.w + 2 * padding+stride)/ stride;

    assert(output.h == val1 && output.w == val2 && input.d == weights.d);
    assert(input.on_device && weights.on_device && output.on_device);


    //Lab-1: please complete this
    //You need to define separate kernel function(s) and launch them here
    //delete assert(0) when you are finished
    int32_t N1 = A.h, M2 = B.w;
    
    dim3 threadsPerBlock(16,16,1);

    MultiplyFunc<T> f;
    dim3 numBlocks( 
        (val2+ threadsPerBlock.x-1)/threadsPerBlock.x, 
        (val1+threadsPerBlock.y-1)/threadsPerBlock.y,
        input.d
    );

    conv_kernel<<numBlocks , threadsPerBlock>>>(input, weights, output, stride, padding)
    //mat_mul_kernel<<<numBlocks, threadsPerBlock>>>(f, A, B, C); 
    //mat_mul_kernel<<<numBlocks, threadsPerBlock>>>(f, A, B, C); 
    //assert(0);
}


__global__ void conv2d_multichannel_kernel(float *input, float *output, float *kernel,
                                           int width, int height, int channels,
                                           int kernel_width, int kernel_height,
                                           int pad_x, int pad_y, int stride_x, int stride_y) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z * blockDim.z + threadIdx.z;

    int kernel_radius_x = kernel_width / 2;
    int kernel_radius_y = kernel_height / 2;

    if (channel >= channels) return;

    float value = 0.0;
    // Map the output position to the input position, considering stride and padding
    int in_x_origin = (out_x * stride_x) - pad_x;
    int in_y_origin = (out_y * stride_y) - pad_y;

    if (out_x >= (width + 2 * pad_x - kernel_width + stride_x) / stride_x ||
        out_y >= (height + 2 * pad_y - kernel_height + stride_y) / stride_y)
        return;

    for (int i = 0; i < kernel_width; i++) {
        for (int j = 0; j < kernel_height; j++) {
            int in_x = in_x_origin + i;
            int in_y = in_y_origin + j;

            if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                int input_index = (in_y * width + in_x) * channels + channel;
                int kernel_index = (j * kernel_width + i) * channels + channel;
                value += input[input_index] * kernel[kernel_index];
            }
        }
    }

    int output_index = ((out_y * ((width + 2 * pad_x - kernel_width + stride_x) / stride_x) + out_x) * channels) + channel;
    output[output_index] = value;
}