#pragma once

#include "utils/tensor3D.cuh"

template <typename T>
Tensor3D<T> max_pool2d_forward(const Tensor3D<T>& input, int kernel_size) {
    int output_h = input.h / kernel_size;
    int output_w = input.w / kernel_size;
    Tensor3D<T> output(output_h, output_w, input.d, input.on_device);

    for (int i = 0; i < input.d; ++i) {
        for (int j = 0; j < output_h; ++j) {
            for (int k = 0; k < output_w; ++k) {
                T max_val = Index3D(input, j * kernel_size, k * kernel_size, i);
                for (int m = 0; m < kernel_size; ++m) {
                    for (int n = 0; n < kernel_size; ++n) {
                        T val = Index3D(input, j * kernel_size + m, k * kernel_size + n, i);
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                Index3D(output, j, k, i) = max_val;
            }
        }
    }
    return output;
}

template <typename T>
Tensor3D<T> max_pool2d_backward(const Tensor3D<T>& grad_output, const Tensor3D<T>& input, int kernel_size) {
    Tensor3D<T> grad_input(input.h, input.w, input.d, input.on_device);

    for (int i = 0; i < input.d; ++i) {
        for (int j = 0; j < grad_input.h; ++j) {
            for (int k = 0; k < grad_input.w; ++k) {
                Index3D(grad_input, j, k, i) = 0;
            }
        }

        for (int j = 0; j < grad_output.h; ++j) {
            for (int k = 0; k < grad_output.w; ++k) {
                T grad_val = Index3D(grad_output, j, k, i);
                T max_val = Index3D(input, j * kernel_size, k * kernel_size, i);
                int max_row = j * kernel_size;
                int max_col = k * kernel_size;
                for (int m = 0; m < kernel_size; ++m) {
                    for (int n = 0; n < kernel_size; ++n) {
                        T val = Index3D(input, j * kernel_size + m, k * kernel_size + n, i);
                        if (val > max_val) {
                            max_val = val;
                            max_row = j * kernel_size + m;
                            max_col = k * kernel_size + n;
                        }
                    }
                }
                Index3D(grad_input, max_row, max_col, i) = grad_val;
            }
        }
    }
    return grad_input;
}

