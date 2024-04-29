#pragma once
#include "modules/linear.cuh"
#include "modules/tensor3D.cuh"
#include "modules/tensor.cuh"
template <typename T>
class ConvNet
{
private:
    std::vector<LinearLayer<T>> layers;
    std::vector<int> layer_dims;
    std::vector<Tensor<T>> activ;
    std::vector<Tensor<T>> d_activ;

    int batch_size;
    int in_dim;

public:
    MLP(int batch_size_, int in_dim_, std::vector<int> layer_dims_, bool gpu)
        : batch_size(batch_size_), in_dim(in_dim_), layer_dims(layer_dims_)
    {
        for (int i = 0; i < layer_dims.size(); i++)
        {
            if (i == 0)
            {
                layers.emplace_back(in_dim, layer_dims[i], gpu);
            }
            else
            {
                layers.emplace_back(layer_dims[i - 1], layer_dims[i], gpu);
            }
        }
        // make all the activation tensors
        activ.reserve(layer_dims.size() - 1);
        d_activ.reserve(layer_dims.size() - 1);
        for (int i = 0; i < layer_dims.size() - 1; i++)
        {
            activ.emplace_back(Tensor<T>(batch_size, layer_dims[i], gpu));
            // technically, i do not need to save d_activ for backprop, but since iterative
            // training does repeated backprops, reserving space for these tensor once is a good idea
            d_activ.emplace_back(Tensor<T>(batch_size, layer_dims[i], gpu));
        }
    }

    std::vector<Parameter<T> *> parameters()
    {
        std::vector<Parameter<T> *> params;
        for (int i = 0; i < layer_dims.size(); i++)
        {
            auto y = layers[i].parameters();
            params.insert(params.end(), y.begin(), y.end());
        }
        return params;
    }

    void init() {
        for (int i = 0; i < layers.size(); i++) {
            layers[i].init_uniform();
        }
    }

    //This function peforms the forward operation of a MLP model
    //Specifically, it should call the forward oepration of each linear layer 
    //Except for the last layer, it should invoke Relu activation after each layer.
    void forward(const Tensor<T> &in, Tensor<T> &out)
    {
        //Lab-2: add your code here
        int num_layers = layers.size();
        Tensor<T> intermediate_input = in;
        for(int i=0;i<num_layers;i++)
        {
                //std::cout<<intermediate_input.h<<" "<<intermediate_input.w<<" "<<layers[i].w.h<<" "<<layers[i].w.w<<std::endl;
                Tensor<T> intermediate_activation{batch_size, layer_dims[i], true};
                layers[i].forward(intermediate_input, intermediate_activation);
                if(i!=num_layers-1)
                    activ[i] = intermediate_activation;

                intermediate_input= intermediate_activation;
                out = intermediate_activation;
        }
        //std::cout<<out.h<<" "<<out.w<<std::endl;
    }

    //This function perofmrs the backward operation of a MLP model.
    //Tensor "in" is the gradients for the outputs of the last linear layer (aka d_logits from op_cross_entropy_loss)
    //Invoke the backward function of each linear layer and Relu from the last one to the first one.
    void backward(const Tensor<T> &in, const Tensor<T> &d_out, Tensor<T> &d_in)
    {
       //Lab-2: add your code here
       int n_layers = layers.size();
       d_activ[n_layers-1] = d_out;

       for(int i=n_layers-1;i>=0;i--)
       {
            if(i!=0){
                layers[i].backward(activ[i-1], d_activ[i], d_activ[i-1]);
            }else{
                layers[i].backward(in, d_activ[i], d_in);
            }
       }
    }
};
