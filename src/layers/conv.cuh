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
	
    ConvLayer(string layer_id_, int in_height_, int in_width_, int in_depth_, int out_height_, int out_width_, int w_height_, int w_width_, int num_filters_, bool gpu,int stride_=1, int padding_=0): layer_id(layer_id_),in_height(in_dim_), in_width(in_width_), in_depth(in_depth_), out_height(out_height_), w_weight(w_height_), num_filters(num_filters_), stride(stride_), padding(padding_){
		  params = ConvParameter<T>{w_height, w_width, in_depth, num_filters, gpu};
    }

    ConvLayer() {}
    
    ConvLayer(ConvLayer&& other) : in_height(other.in_dim), in_width(other.in_width), in_depth(other.in_depth), out_height(other.out_height),out_width(other.out_width), w_weight(other.w_height), num_filters(other.num_filters) {}
                
    // std::vector<Parameter<T>*> parameters() {
    //     std::vector<Parameter<T> *> v;
    //     v.push_back(&w);
    //     v.push_back(&b);
    //     return v;
    // }
    
    void init_uniform() {

        // Do Kaiming uniform
        float max = 1.0f / std::sqrt(in_height+in_width+in_depth);

        for(int i=0;i<num_filters;i++){
          //changed implementation for 3d
            for(int j=0;j<in_depth;j++){

              op_uniform_init(params.weights[i].values[j], -max, max);
              op_uniform_init(params.d_weights[i].values[j], -max, max);

            }
        }
    }

    //This function calculates the output of a conv layer 
    //and stores the result in tensor "y"
    void forward(const Tensor3D<float> &in, Tensor3D<float> &out) {
       
       assert(in.d== weights[0].d && num_filters == out.d);

       Tensor3D<float> in_device = in.toDevice();
       Tensor3D<float> padded_tensor = Tensor3D<float>{in.h+2*padding, in.w + 2*padding, in.d, true};
       op_padding(in, padded_tensor, padding);
       for(int i=0;i<num_filters;i++)
       {
          Tensor3D<float> w_t_device =  params.weights[i].toDevice();
          op_conv( padded_tensor, w_t_device, out.values[i], stride, padding);
          op_relu(out.values[i], out.values[i]);
       }
              
    }

    //This function performs the backward operation of a linear layer
    //Suppose y = Linear(x). Then function argument "dy" is the gradients of "y", 
    //and function argument "x" is the saved x.
    //This function compute the weight gradients (dw, db) and saves them in w.dt and b.dt respectively
    //It also computes the graidents of "x" and saves it in dx.
    void backward(const Tensor<float> &x, const Tensor<float> &dy, Tensor<float> &dx) {
      //Lab-2: Please add your code here
      
      // Tensor<T> output_y{x.h, w.t.w, true};
      // Tensor x_temp = x.toDevice(); 
      // //printf("p\n");
      // forward(x, output_y);
      // Tensor<T> derivRelu{ x.h, out_dim, true};

      // op_relu_back(output_y, dy, derivRelu);
      // //printf("h\n");
      // op_mm(x.transpose(),derivRelu, w.dt);
      // op_sum(derivRelu, b.dt);
      // //printf("k\n");
      // op_mm(derivRelu, w.t.transpose(), dx);
    }

};
