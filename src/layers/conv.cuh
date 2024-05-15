#pragma once
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"
#include "ops/op_mm.cuh"
#include "modules/convParams.cuh"
#include "ops/op_padding.cuh"

template<typename T>
class ConvLayer {

public:
   string layerid;
   int in_height;
   int in_width;
   int in_depth;
   int out_height;
   int out_width;
   int w_height;
   int w_width;
   int num_filters;
   int stride;
   int padding;
   ConvParameter<T> params;

   ConvLayer(string layer_id_, int in_height_, int in_width_, int in_depth_, int out_height_, int out_width_, int w_height_, int w_width_, int num_filters_, bool gpu, int stride_ = 1, int padding_ = 0)
       : layerid(layer_id_), in_height(in_height_), in_width(in_width_), in_depth(in_depth_), out_height(out_height_), out_width(out_width_), w_height(w_height_), w_width(w_width_), num_filters(num_filters_), stride(stride_), padding(padding_) {
       params = ConvParameter<T>{w_height, w_width, in_depth, num_filters, gpu};
   }

   ConvLayer() {}

   ConvLayer(ConvLayer&& other)
       : in_height(other.in_height), in_width(other.in_width), in_depth(other.in_depth), out_height(other.out_height), out_width(other.out_width), w_height(other.w_height), num_filters(other.num_filters) {}

   void init_uniform() {
       float max = 1.0f / std::sqrt(in_height + in_width + in_depth);

       for (int i = 0; i < num_filters; i++) {
           for (int j = 0; j < in_depth; j++) {
               op_uniform_init(params.weights[i].values[j], -max, max);
               op_uniform_init(params.d_weights[i].values[j], -max, max);
           }
       }
   }

   void forward(const Tensor3D<float>& in, Tensor3D<float>& out) {
       assert(in.d == params.weights[0].d && num_filters == out.d);

       Tensor3D<float> in_device = in.toDevice();
       Tensor3D<float> padded_tensor = Tensor3D<float>{in.h + 2 * padding, in.w + 2 * padding, in.d, true};
       op_padding(in, padded_tensor, padding);
       for (int i = 0; i < num_filters; i++) {
           Tensor3D<float> w_t_device = params.weights[i].toDevice();
           op_conv(padded_tensor, w_t_device, out.values[i], stride, padding);
           op_relu(out.values[i], out.values[i]);
       }
   }

   void backward(const Tensor3D<float>& x, const Tensor3D<float>& dy, Tensor3D<float>& dx) {
       Tensor3D<float> x_padded = Tensor3D<float>{x.h + 2 * padding, x.w + 2 * padding, x.d, true};
       op_padding(x, x_padded, padding);

       for (int i = 0; i < num_filters; i++) {
           Tensor3D<float> dy_i = dy.values[i];
           Tensor3D<float> dx_i = dx.values[i];
           Tensor3D<float> w_t = params.weights[i];

           op_conv_back_data(dy_i, w_t, x_padded, dx_i, stride, padding);
           op_conv_back_weights(x_padded, dy_i, w_t, params.d_weights[i], stride, padding);
       }
   }
};
