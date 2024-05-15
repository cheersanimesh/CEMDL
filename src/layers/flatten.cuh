#pragma once

#include "utils/tensor3D.cuh"
#include "utils/tensor.cuh"

// Template class for a FlattenLayer.
// This layer converts a multi-dimensional tensor into a 2D tensor,
// where the first dimension is often the batch size, and the second dimension is the flattened data from the other dimensions.
template <typename T>
class FlattenLayer {
public:
    // Forward pass of the flatten layer.
    // It reshapes the input tensor `x` from a 3D tensor (batch_size, height, width) into a 2D tensor (batch_size, height * width).
    void forward(const Tensor3D<T>& x, Tensor<T>& y) {
        int batch_size = x.d;  // Batch size is assumed to be the depth of the tensor
        int input_size = x.h * x.w;  // Total number of elements per batch element after flattening
        y = Tensor<T>(batch_size, input_size, x.on_device);  // Initialize output tensor to hold the flattened data

        // Iterate over each element in the batch
        for (int b = 0; b < batch_size; b++) {
            // Flatten each element
            for (int i = 0; i < x.h; i++) {
                for (int j = 0; j < x.w; j++) {
                    y.rawp[b * input_size + i * x.w + j] = Index3D(x, i, j, b);
                }
            }
        }
    }

    // Backward pass of the flatten layer.
    // It takes the gradient with respect to the output of the layer `dy` and computes the gradient with respect to the input `dx`.
    void backward(const Tensor<T>& dy, Tensor3D<T>& dx) {
        int batch_size = dx.d;  // Batch size is taken from the depth of the input tensor
        int input_size = dx.h * dx.w;  // Total number of elements per batch element in the input
        dx = Tensor3D<T>(dx.h, dx.w, dx.d, dy.on_device);  // Initialize the input gradient tensor to match the input dimensions

        // Iterate over each element in the batch
        for (int b = 0; b < batch_size; b++) {
            // Reconstruct the gradient for each original element
            for (int i = 0; i < dx.h; i++) {
                for (int j = 0; j < dx.w; j++) {
                    Index3D(dx, i, j, b) = dy.rawp[b * input_size + i * dx.w + j];
                }
            }
        }
    }
};

