#pragma once
#include "utils/tensor.cuh"
#include "op_reduction.cuh"

//This function calculates the cross_entropy loss from the "logits" tensor for a batch of training innput
//and the batch's corresponding "target" label tensor and returns the average loss of the batch.
//It also returns the gradient of the logits tensor.
template<typename T>
__global__ void op_cross_entropy_kernel(Tensor<T> logits, Tensor<T> max_indices, Tensor<char> targets, Tensor<T> d_logits, Tensor<T> losses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(!IndexOutofBound(logits,idx,0)){
        int tar_idx = (int)Index(targets, idx,0);
        int batch_size = logits.h;
        //T numr = exp( Index(logits, idx, tar_idx)- xmax);
        T denomr = (T) 0.0;
        // T xmax = Index( 
        //     logits,
        //     idx,
        //     Index(max_indices, idx,0)
        // );
        T xmax = Index(max_indices, 0,0);
        T numr = (T) exp(Index(logits, idx, tar_idx) - xmax);
        for(int i=0;i<logits.w;i++)
        {
            denomr += exp( Index(logits, idx,i)- xmax);
            Index(d_logits, idx, i) = (T) exp(Index(logits, idx, i) - xmax);
        }
        T prob = numr /denomr;
        
        // for(int i=0;i<logits.w;i++)
        // {
        //     if(i==tar_idx)
        //         Index(d_logits, idx,i)=prob-1;
        //     else
        //         Index(d_logits, idx,i)=prob;
        // }
        for(int i=0;i<logits.w;i++)
        {
            T d_prob = Index(d_logits, idx, i)/denomr; 
            if(i==tar_idx)
                Index(d_logits, idx,i)=(prob-1)/batch_size;
            else
                Index(d_logits, idx,i)= d_prob / batch_size;
        }
        T loss = -1 * logf(prob);
        __syncthreads();
        Index(losses, idx,0)= loss;
    }
}

template <typename T>
T op_cross_entropy_loss(const Tensor<T> &logits, const Tensor<char> targets,
                               Tensor<T> &d_logits)
{
    assert(logits.h == targets.h && logits.h == d_logits.h);
    assert(logits.w == d_logits.w);
    assert(targets.w == 1);
    //std::cout<<logits.on_device<<" "<<targets.on_device<<" "<<d_logits.on_device<<std::endl;

    //assert(logits.on_device && targets.on_device && d_logits.on_device); 
    //Lab-2: please add your code here. 
    //You need to define separate GPU kernel function(s) and launch them here
    //In order to calculate d_logits, you should derive what its values should be 
    //symbolically.

    int batch_size = logits.h, num_classes = logits.w;

    Tensor<int> max_indices{batch_size,1, true};

    Tensor<char> target_idx_cpu = targets.toHost();

    // op_argmax(logits, max_indices);
    // Tensor<int> max_index{1,1,true};
    // op_argmax(max_indices, max_index);

    Tensor<T> max_val{1,1,false};
    Tensor<T> logits_host = logits.toHost();
    Index(max_val, 0,0)= Index(logits_host,0,0);
    for(int i=0;i<logits_host.h;i++)
    {
        for(int j=0;j<logits_host.w;j++)
            Index(max_val,0,0) = max( Index(max_val,0,0) , Index(logits_host,i,j) );
    }
    Tensor<T> max_val_device = max_val.toDevice(); 

    dim3 threadsPerBlock(16);
    dim3 numBlocks ((batch_size+ threadsPerBlock.x -1 ) / threadsPerBlock.x);

    Tensor<T> losses{batch_size, 1, true};
    op_cross_entropy_kernel<<< numBlocks, threadsPerBlock>>>(logits, max_val_device, targets, d_logits, losses);

    //op_cross_entropy_kernel<<< numBlocks, threadsPerBlock>>>(logits, max_indices, targets, d_logits, losses);
    const Tensor<T> losses_2 = losses.toHost();
        
    T sum_loss= 0.0;

    for(int i=0;i<losses_2.h;i++){
        sum_loss += Index(losses_2,i,0);
        //std::cout<<"Hello-->"<<Index(losses_2,i,0)<<std::endl;
    }
    return sum_loss/batch_size;
}
