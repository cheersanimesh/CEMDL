#pragma once
#include "modules/param.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"
#include "ops/op_mm.cuh"

template<typename T>
class LinearLayer {
    private:
        int in_dim;
        int out_dim;

        Parameter<T> w;
        Parameter<T> b;

    public:
    LinearLayer(int in_dim_, int out_dim_, bool gpu):in_dim(in_dim_), out_dim(out_dim_) {
        w = Parameter<T>{in_dim, out_dim, gpu};
        b = Parameter<T>{1, out_dim, gpu};
    }

    LinearLayer() {}
    
    LinearLayer(LinearLayer&& other) : in_dim(other.in_dim), out_dim(other.out_dim), w(other.w), b(other.b) {}
                
    std::vector<Parameter<T>*> parameters() {
        std::vector<Parameter<T> *> v;
        v.push_back(&w);
        v.push_back(&b);
        return v;
    }
    
    void init_uniform() {
        // Do Kaiming uniform
        float max = 1.0f / std::sqrt(in_dim);
        op_uniform_init(w.t, -max, max);
        op_uniform_init(b.t, -max, max);
        //std::cout << "init b=" << b.t.str() << std::endl;
    }

    //This function calculates the output of a lienar layer 
    //and stores the result in tensor "y"
    void forward(const Tensor<float> &x, Tensor<float> &y) {
      //Lab-2: please add your code here
      //Tensor<float> z;

      //std::cout<<x.h<<" "<<x.w<<" "<<w.t.h<<" "<<w.t.w<<" "<<y.h<<" "<<y.w<<std::endl;

       Tensor<float> x_device = x.toDevice();
       Tensor<float> w_t_device =  w.t.toDevice();
       Tensor<float> b_t_device = b.t.toDevice();
       op_mm(x_device, w_t_device, y);
       op_add(y, b_t_device, y);
       op_relu(y, y);
    }

    //This function performs the backward operation of a linear layer
    //Suppose y = Linear(x). Then function argument "dy" is the gradients of "y", 
    //and function argument "x" is the saved x.
    //This function compute the weight gradients (dw, db) and saves them in w.dt and b.dt respectively
    //It also computes the graidents of "x" and saves it in dx.
    void backward(const Tensor<float> &x, const Tensor<float> &dy, Tensor<float> &dx) {
      Tensor<T> output_y{x.h, w.t.w, true};
      Tensor x_temp = x.toDevice(); 
      forward(x, output_y);
      Tensor<T> derivRelu{ x.h, out_dim, true};

      op_relu_back(output_y, dy, derivRelu);
      op_mm(x.transpose(),derivRelu, w.dt);
      op_sum(derivRelu, b.dt);
      op_mm(derivRelu, w.t.transpose(), dx);
    }

};
