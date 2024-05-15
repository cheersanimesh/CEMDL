#pragma once

#include "utils/tensor3D.cuh"
#include "utils/tensor.cuh"

// Converts a multi-dimensional tensor into a 2D tensor,
// where the first dimension is often the batch size, and the second dimension is the flattened data from the other dimensions.
template <typename T>
class FlattenLayer {
public:
    void forward(const Tensor3D<T>& x, Tensor<T>& y) {
        int batch_size = x.d;
        int input_size = x.h * x.w;
        y = Tensor<T>(batch_size, input_size, x.on_device);

        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < x.h; i++) {
                for (int j = 0; j < x.w; j++) {
                    y.rawp[b * input_size + i * x.w + j] = Index3D(x, i, j, b);
                }
            }
        }
    }

    void backward(const Tensor<T>& dy, Tensor3D<T>& dx) {
        int batch_size = dx.d;
        int input_size = dx.h * dx.w;
        dx = Tensor3D<T>(dx.h, dx.w, dx.d, dy.on_device);

        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < dx.h; i++) {
                for (int j = 0; j < dx.w; j++) {
                    Index3D(dx, i, j, b) = dy.rawp[b * input_size + i * dx.w + j];
                }
            }
        }
    }
};

