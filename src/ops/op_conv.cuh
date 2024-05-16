#pragma once
#include "utils/tensor.cuh"
#include "ops/op_gemm.cuh"

template <typename T>
void op_conv(const Tensor<T>& input, const Tensor<T>& weight, const Tensor<T>& bias, Tensor<T>& output, int stride, int padding) {
    int batch_size = input.h;
    int input_channels = input.w;
    int input_height = input.d1;
    int input_width = input.d2;

    int output_channels = weight.h;
    int kernel_height = weight.d1;
    int kernel_width = weight.d2;

    int output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

    Tensor<T> input_padded = input.pad(padding);

    output = Tensor<T>{batch_size, output_channels, output_height, output_width, true};

    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < output_channels; ++oc) {
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    Tensor<T> input_slice = input_padded.slice(b, b + 1, 0, input_channels, oh * stride, oh * stride + kernel_height, ow * stride, ow * stride + kernel_width);
                    Tensor<T> weight_slice = weight.slice(oc, oc + 1, 0, input_channels, 0, kernel_height, 0, kernel_width);

                    op_gemm(input_slice, weight_slice, output.slice(b, b + 1, oc, oc + 1, oh, oh + 1, ow, ow + 1));
                }
            }
            op_add(output.slice(b, b + 1, oc, oc + 1, 0, output_height, 0, output_width), bias.slice(0, 1, oc, oc + 1, 0, 1, 0, 1), output.slice(b, b + 1, oc, oc + 1, 0, output_height, 0, output_width));
        }
    }
}

template <typename T>
void op_conv_back(const Tensor<T>& input, const Tensor<T>& output_grad, const Tensor<T>& weight, Tensor<T>& weight_grad, Tensor<T>& bias_grad, Tensor<T>& input_grad, int stride, int padding) {
    int batch_size = input.h;
    int input_channels = input.w;
    int input_height = input.d1;
    int input_width = input.d2;

    int output_channels = weight.h;
    int kernel_height = weight.d1;
    int kernel_width = weight.d2;

    int output_height = output_grad.d1;
    int output_width = output_grad.d2;

    Tensor<T> input_padded = input.pad(padding);

    weight_grad = Tensor<T>{output_channels, input_channels, kernel_height, kernel_width, true};
    bias_grad = Tensor<T>{1, output_channels, 1, 1, true};
    input_grad = Tensor<T>{batch_size, input_channels, input_height, input_width, true};

    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < output_channels; ++oc) {
            op_sum(output_grad.slice(b, b + 1, oc, oc + 1, 0, output_height, 0, output_width), bias_grad.slice(0, 1, oc, oc + 1, 0, 1, 0, 1));

            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    Tensor<T> input_slice = input_padded.slice(b, b + 1, 0, input_channels, oh * stride, oh * stride + kernel_height, ow * stride, ow * stride + kernel_width);
                    Tensor<T> output_grad_slice = output_grad.slice(b, b + 1, oc, oc + 1, oh, oh + 1, ow, ow + 1);

                    op_gemm(output_grad_slice, input_slice.transpose(), weight_grad.slice(oc, oc + 1, 0, input_channels, 0, kernel_height, 0, kernel_width), true, false);
                    op_gemm(output_grad_slice, weight.slice(oc, oc + 1, 0, input_channels, 0, kernel_height, 0, kernel_width), input_grad.slice(b, b + 1, 0, input_channels, oh * stride, oh * stride + kernel_height, ow * stride, ow * stride + kernel_width), false, true);
                }
            }
        }
    }
}
