#pragma once

#include "utils/tensor3D.cuh"

// Template class for a FlattenLayer.
// This layer converts a multi-dimensional tensor into a 2D tensor,
// where the first dimension is often the batch size, and the second dimension is the flattened data from the other dimensions.
template <typename T>
class FlattenLayer {
public:
    // Forward pass of the flatten layer.
    // It reshapes the input tensor `x` from a 3D tensor (batch_size, width, channels) into a 2D tensor (batch_size, width*channels).
    void forward(const Tensor3D<T>& x, Tensor<T>& y) {
        int batch_size = x.h;  // Batch size is assumed to be the height of the tensor
        int input_size = x.w * x.h;  // Total number of elements per batch element after flattening
        y.resize(batch_size, input_size);  // Resize output tensor to hold the flattened data

        // Iterate over each element in the batch
        for (int b = 0; b < batch_size; b++) {
            // Flatten each element
            for (int i = 0; i < input_size; i++) {
                int w = i % x.w;  // Compute the width index in the original tensor
                int c = i / x.w;  // Compute the channel index in the original tensor
                y.at(b, i) = x.at(b, w, c);  // Assign the flattened data to the output tensor
            }
        }
    }

    // Backward pass of the flatten layer.
    // It takes the gradient with respect to the output of the layer `dy` and computes the gradient with respect to the input `dx`.
    void backward(const Tensor<T>& x, const Tensor<T>& dy, Tensor<T>& dx) {
        int batch_size = x.h;  // Batch size is taken from the input tensor's height
        int input_size = x.w * x.c;  // Total number of elements per batch element in the input
        dx.resize(x.h, x.w, x.c);  // Resize the input gradient tensor to match the input dimensions

        // Iterate over each element in the batch
        for (int b = 0; b < batch_size; b++) {
            // Reconstruct the gradient for each original element
            for (int i = 0; i < input_size; i++) {
                int w = i % x.w;  // Compute the width index in the original tensor
                int c = i / x.w;  // Compute the channel index in the original tensor
                dx.at(b, w, c) = dy.at(b, i);  // Assign the gradient from the flattened data back to the corresponding position
            }
        }
    }
};

