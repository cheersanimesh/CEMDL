#pragma once
#include "utils/tensor3D.cuh"
#include "utils/conv.cuh"
#include "utils/pooling.cuh"
#include "utils/flatten.cuh"
#include "ops/op_softmax.cuh" // Ensure this includes the softmax operation
#include <vector>
#include <string>
#include <cmath>

template <typename T>
class ResNetBlock {
public:
    ConvLayer<T> conv1;
    ConvLayer<T> conv2;
    bool downsample;
    Tensor3D<T> identity;

    ResNetBlock(int in_channels, int out_channels, int stride, bool gpu) {
        downsample = (stride != 1) || (in_channels != out_channels);
        conv1 = ConvLayer<T>("conv1", in_channels, out_channels, 3, 3, stride, 1, gpu);
        conv2 = ConvLayer<T>("conv2", out_channels, out_channels, 3, 3, 1, 1, gpu);

        if (downsample) {
            identity = Tensor3D<T>(1, 1, out_channels, gpu);
        }
    }

    void forward(const Tensor3D<T>& x, Tensor3D<T>& out) {
        Tensor3D<T> residual = x;
        conv1.forward(x, out);
        conv2.forward(out, out);

        if (downsample) {
            identity = Tensor3D<T>(x.h / 2, x.w / 2, out.d, x.on_device);
            for (int i = 0; i < out.d; ++i) {
                for (int j = 0; j < identity.h; ++j) {
                    for (int k = 0; k < identity.w; ++k) {
                        Index3D(identity, j, k, i) = Index3D(residual, j * 2, k * 2, i);
                    }
                }
            }
        } else {
            identity = residual;
        }

        for (int i = 0; i < out.d; ++i) {
            for (int j = 0; j < out.h; ++j) {
                for (int k = 0; k < out.w; ++k) {
                    Index3D(out, j, k, i) += Index3D(identity, j, k, i);
                }
            }
        }
    }

    void backward(const Tensor3D<T>& grad_output, const Tensor3D<T>& x, Tensor3D<T>& grad_input) {
        Tensor3D<T> grad_residual = grad_output;
        Tensor3D<T> grad_conv1, grad_conv2;
        conv2.backward(x, grad_output, grad_conv2);
        conv1.backward(x, grad_conv2, grad_conv1);

        if (downsample) {
            Tensor3D<T> grad_identity = Tensor3D<T>(x.h / 2, x.w / 2, grad_residual.d, grad_residual.on_device);
            for (int i = 0; i < grad_residual.d; ++i) {
                for (int j = 0; j < grad_identity.h; ++j) {
                    for (int k = 0; k < grad_identity.w; ++k) {
                        Index3D(grad_identity, j, k, i) = Index3D(grad_residual, j * 2, k * 2, i);
                    }
                }
            }
            grad_input = grad_identity;
        } else {
            grad_input = grad_residual;
        }
    }
};

template <typename T>
class ResNet {
public:
    ConvLayer<T> initial_conv;
    std::vector<ResNetBlock<T>> layers;
    FlattenLayer<T> flatten;
    Tensor<T> fc_weights;
    Tensor<T> fc_bias;
    int num_classes;

    ResNet(int num_classes_, bool gpu) : num_classes(num_classes_) {
        initial_conv = ConvLayer<T>("initial_conv", 1, 64, 7, 7, 2, 3, gpu);
        layers.push_back(ResNetBlock<T>(64, 64, 1, gpu));
        layers.push_back(ResNetBlock<T>(64, 64, 1, gpu));
        layers.push_back(ResNetBlock<T>(64, 128, 2, gpu));
        layers.push_back(ResNetBlock<T>(128, 128, 1, gpu));
        layers.push_back(ResNetBlock<T>(128, 256, 2, gpu));
        layers.push_back(ResNetBlock<T>(256, 256, 1, gpu));
        layers.push_back(ResNetBlock<T>(256, 512, 2, gpu));
        layers.push_back(ResNetBlock<T>(512, 512, 1, gpu));
        flatten = FlattenLayer<T>();
        fc_weights = Tensor<T>(512, num_classes_, gpu);
        fc_bias = Tensor<T>(1, num_classes_, gpu);
        op_uniform_init(fc_weights);
        op_const_init(fc_bias, 0.0);
    }

    void forward(const Tensor3D<T>& x, Tensor<T>& out) {
        Tensor3D<T> current = x;
        initial_conv.forward(current, current);
        for (auto& layer : layers) {
            layer.forward(current, current);
        }
        flatten.forward(current, out);
        Tensor<T> logits{out.h, fc_weights.w, out.on_device};
        op_mm(out, fc_weights, logits);
        op_add(logits, fc_bias, out);
        op_softmax(out, out); // Apply softmax to convert logits to probabilities
    }

    void backward(const Tensor<T>& grad_output, const Tensor3D<T>& x, Tensor3D<T>& grad_input) {
        Tensor<T> grad_flatten;
        Tensor<T> grad_fc_weights{fc_weights.h, fc_weights.w, fc_weights.on_device};
        Tensor<T> grad_fc_bias{fc_bias.h, fc_bias.w, fc_bias.on_device};

        // Gradient with respect to fully connected layer weights and bias
        op_mm(grad_output.transpose(), flatten.y.transpose(), grad_fc_weights);
        op_sum(grad_output, grad_fc_bias);

        // Gradient with respect to the output of the flatten layer
        op_mm(grad_output, fc_weights.transpose(), grad_flatten);

        Tensor3D<T> current_grad = grad_flatten.toDevice();
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            it->backward(current_grad, x, current_grad);
        }
        grad_input = current_grad;
    }

    std::vector<Tensor<T>> parameters() {
        std::vector<Tensor<T>> params;
        params.push_back(fc_weights);
        params.push_back(fc_bias);
        return params;
    }
};

