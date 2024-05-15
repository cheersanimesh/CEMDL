#pragma once
#include "utils/tensor3D.cuh"

template <typename T>
Tensor3D<T> max_pool2d_forward(const Tensor3D<T>& input, int kernel_size) {
    int output_d = input.d;
    int output_h = input.h / kernel_size;
    int output_w = input.w / kernel_size;

    Tensor3D<T> output(output_d, output_h, output_w, input.on_device);

    if (!input.on_device) {
        for (int i = 0; i < output_d; i++) {
            for (int j = 0; j < output_h; j++) {
                for (int k = 0; k < output_w; k++) {
                    T max_val = input.rawp[i * input.stride_d + j * kernel_size * input.stride_h + k * kernel_size * input.stride_w];
                    for (int m = 0; m < kernel_size; m++) {
                        for (int n = 0; n < kernel_size; n++) {
                            T val = input.rawp[i * input.stride_d + (j * kernel_size + m) * input.stride_h + (k * kernel_size + n) * input.stride_w];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                    output.rawp[i * output.stride_d + j * output.stride_h + k * output.stride_w] = max_val;
                }
            }
        }
    }

    return output;
}

template <typename T>
Tensor3D<T> max_pool2d_backward(const Tensor3D<T>& grad_output, const Tensor3D<T>& input, int kernel_size) {
    Tensor3D<T> grad_input(input.d, input.h, input.w, input.on_device);

    if (!input.on_device) {
        for (int i = 0; i < grad_input.d; i++) {
            for (int j = 0; j < grad_input.h; j++) {
                for (int k = 0; k < grad_input.w; k++) {
                    grad_input.rawp[i * grad_input.stride_d + j * grad_input.stride_h + k * grad_input.stride_w] = 0;
                }
            }
        }

        int output_h = grad_output.h;
        int output_w = grad_output.w;

        for (int i = 0; i < grad_input.d; i++) {
            for (int j = 0; j < output_h; j++) {
                for (int k = 0; k < output_w; k++) {
                    T grad_val = grad_output.rawp[i * grad_output.stride_d + j * grad_output.stride_h + k * grad_output.stride_w];
                    T max_val = input.rawp[i * input.stride_d + j * kernel_size * input.stride_h + k * kernel_size * input.stride_w];
                    for (int m = 0; m < kernel_size; m++) {
                        for (int n = 0; n < kernel_size; n++) {
                            T val = input.rawp[i * input.stride_d + (j * kernel_size + m) * input.stride_h + (k * kernel_size + n) * input.stride_w];
                            if (val == max_val) {
                                grad_input.rawp[i * input.stride_d + (j * kernel_size + m) * input.stride_h + (k * kernel_size + n) * input.stride_w] = grad_val;
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}
