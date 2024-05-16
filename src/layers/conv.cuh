#pragma once
#include "modules/param.cuh"
#include "ops/op_conv.cuh"
#include "ops/op_reduction.cuh"

template<typename T>
class ConvLayer {
private:
    int in_channels, out_channels, kernel_size;
    int stride, padding;

    Parameter<T> w;
    Parameter<T> b;

public:
    ConvLayer(int in_channels_, int out_channels_, int kernel_size_, int stride_=1, int padding_=0, bool gpu=true)
        : in_channels(in_channels_), out_channels(out_channels_), kernel_size(kernel_size_),
          stride(stride_), padding(padding_), w({out_channels, in_channels, kernel_size, kernel_size, gpu}), 
          b({1, out_channels, 1, 1, gpu}) {}

    ConvLayer() {}

    ConvLayer(ConvLayer&& other)
        : in_channels(other.in_channels), out_channels(other.out_channels), kernel_size(other.kernel_size),
          stride(other.stride), padding(other.padding), w(std::move(other.w)), b(std::move(other.b)) {}

    std::vector<Parameter<T>*> parameters() {
        std::vector<Parameter<T>*> v;
        v.push_back(&w);
        v.push_back(&b);
        return v;
    }

    void init_uniform() {
        float max = 1.0f / std::sqrt(in_channels * kernel_size * kernel_size);
        op_uniform_init(w.t, -max, max);
        op_uniform_init(b.t, -max, max);
    }

    void forward(const Tensor<T>& x, Tensor<T>& y) {
        int batch_size = x.h;
        int input_height = x.w / in_channels; // Use appropriate calculation
        int input_width = x.w / in_channels;  // Use appropriate calculation
        int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
        int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

        Tensor<T> x_padded = x.pad(padding);
        y = Tensor<T>{batch_size, out_channels * output_height * output_width, true};

        for (int b = 0; b < batch_size; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int oh = 0; oh < output_height; ++oh) {
                    for (int ow = 0; ow < output_width; ++ow) {
                        Tensor<T> x_slice = x_padded.slice(b, b + 1, 0, in_channels * input_height * input_width, oh * stride * input_width + ow * stride, oh * stride * input_width + ow * stride + kernel_size);
                        Tensor<T> w_slice = w.t.slice(oc, oc + 1, 0, in_channels * kernel_size * kernel_size);

                        op_gemm(x_slice, w_slice, y.slice(b, b + 1, oc, oc + 1));
                    }
                }
                op_add(y.slice(b, b + 1, oc, oc + 1), this->b.t.slice(0, 1, oc, oc + 1), y.slice(b, b + 1, oc, oc + 1));
            }
        }
    }

    void backward(const Tensor<T>& x, const Tensor<T>& dy, Tensor<T>& dx, Tensor<T>& dw, Tensor<T>& db) {
        int batch_size = x.h;
        int input_height = x.w / in_channels; // Use appropriate calculation
        int input_width = x.w / in_channels;  // Use appropriate calculation
        int output_height = dy.w / out_channels; // Use appropriate calculation
        int output_width = dy.w / out_channels;  // Use appropriate calculation

        Tensor<T> x_padded = x.pad(padding);
        dx = Tensor<T>{batch_size, in_channels * input_height * input_width, true};
        dw = Tensor<T>{out_channels, in_channels * kernel_size * kernel_size, true};
        db = Tensor<T>{1, out_channels, true};

        for (int b = 0; b < batch_size; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                op_sum(dy.slice(b, b + 1, oc, oc + 1), db.slice(0, 1, oc, oc + 1));

                for (int oh = 0; oh < output_height; ++oh) {
                    for (int ow = 0; ow < output_width; ++ow) {
                        Tensor<T> x_slice = x_padded.slice(b, b + 1, 0, in_channels * input_height * input_width, oh * stride * input_width + ow * stride, oh * stride * input_width + ow * stride + kernel_size);
                        Tensor<T> dy_slice = dy.slice(b, b + 1, oc, oc + 1);

                        op_gemm(dy_slice, x_slice.transpose(), dw.slice(oc, oc + 1, 0, in_channels * kernel_size * kernel_size), true, false);
                        op_gemm(dy_slice, w.t.slice(oc, oc + 1, 0, in_channels * kernel_size * kernel_size), dx.slice(b, b + 1, 0, in_channels * input_height * input_width), false, true);
                    }
                }
            }
        }
    }
};

