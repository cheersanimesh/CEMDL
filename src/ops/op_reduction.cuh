#pragma once

#include "utils/tensor.cuh"

template <typename T>
class MaxAccumFunc
{
public:
    //This function compares input x with the current accumulated maximum value stored in accum
    //If x is bigger than accum, stores x in accum and stores x's index (ind_x) to ind_accum
    __host__ __device__ void operator()(const T &x, const int &ind_x, T &accum, int &ind_accum)
    {
      //Lab-1: add your code here
        //printf("index = %d\n", ind_x);
        if(x >= accum)
        {
            accum = x;
            ind_accum = ind_x;
        }
    }
};

template <typename T>
class SumAccumFunc
{
public:
    //This function adds input x to the current accumulated sum value stored in accum
    //The accumu's value is updated (to add x).  The ind_x and ind_accum arguments are not used. 
    __host__ __device__ void operator()(const T &x, const int &ind_x, T &accum, int &ind_accum)
    {
      //Lab-1: add your code here
      accum+= x;
    }
};

//This kernel function performs column-wise reduction of the "in" tensor and stores the result in "out" tensor.
//If "get_index" is true, then the index accumulator values are stored in "out_index" and "out" is not touched.
template <typename OpFunc, typename T>
__global__ void op_reduction_kernel_colwise(OpFunc f, Tensor<T> in, Tensor<T> out, Tensor<int> out_index, bool get_index)
{
    //Lab-1: add your code here
    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
    //if(get_index)
        //printf("call 66\n");
    if(
        (!get_index && !IndexOutofBound(out,row_index,0)) 
        || 
        (get_index && !IndexOutofBound(out_index,row_index,0))
    )

    {
        
        T res =(T) 0.0;
        int res_ind=0; 
        for(int i=0;i<in.w;i++){
                f(
                    Index(in, row_index,i), 
                    i,
                    res,
                    res_ind
                );
        }
        if(!get_index){
            Index(out, row_index, 0) = res;
        }else{
            //printf("result --> %d\n", res_ind);
            Index(out_index, row_index, 0) = res_ind;
        }
    }
    //if(get_index)
            //printf("call 88\n");
}

//This kernel function performs row-wise reduction of the "in" tensor and stores the result in "out" tensor.
//If "get_index" is true, then the index accumulator values are stored in "out_index" and "out" is not touched.
template <typename OpFunc, typename T>
__global__ void op_reduction_kernel_rowwise(OpFunc f, Tensor<T> in, Tensor<T> out, Tensor<int> out_index, bool get_index)
{
    //Lab-1: add your code here

    int col_index = blockDim.x * blockIdx.x + threadIdx.x;
    ////printf("Col Index --> %d \n", col_index);
    if((!get_index && !IndexOutofBound(out,0,col_index)) 
        || 
        (get_index && !IndexOutofBound(out_index,0,col_index))
    )
    {
        //if(get_index)
            //printf("call 7\n");
        T res =(T) 0.0;
        int res_ind=0; 
        for(int i=0;i<in.h;i++){
                f(
                    Index(in, i, col_index), 
                    i,
                    res,
                    res_ind
                );
        }
        //printf("%d get_index %f res_ind %d \n",get_index, res,res_ind);
        if(!get_index){
            Index(out, 0, col_index) = res;
        }else{
            //printf("result --> %d\n", res_ind);
            Index(out_index, 0, col_index) = res_ind;
        }
    }
    //if(get_index)
        //printf("call 8\n");
}    

template <typename OpFunc, typename T>
void op_reduction_gpu(OpFunc f, const Tensor<T> &in, Tensor<T> &out, Tensor<int> &out_index, bool get_index = false)
{
    int out_h = out.h;
    if (!get_index) {
        assert((out.h == 1 && in.w == out.w) || (out.w == 1 && in.h == out.h));
    } else {
        //printf("call 3\n");
        out_h = out_index.h;
        assert((out_index.h == 1 && in.w == out_index.w) || (out_index.w == 1 && in.h == out_index.h));
        //printf("call 4\n");
    }
    if (in.h > out_h)
    {
      //Lab-1: add your code here to launch op_reduction_kernel_rowwise
      //delete assert(0) when you are finished
      int32_t M = in.w;
      
      dim3 threadsPerBlock(16);
      dim3 numBlocks( (M + threadsPerBlock.x -1) /threadsPerBlock.x);
      
      //if(get_index)
        //printf("call 4\n");
      op_reduction_kernel_rowwise<<<numBlocks, threadsPerBlock>>>(f, in, out, out_index, get_index);
      //assert(0);
    }
    else
    {
      //Lab-1: add your code here to launch op_reduction_kernel_colwise
      //delete assert(0) when you are finished

      int32_t N = in.h;
      
      dim3 threadsPerBlock(16);
      dim3 numBlocks( (N + threadsPerBlock.x -1) /threadsPerBlock.x);
      //if(get_index)
        //printf("call 44\n");
      op_reduction_kernel_colwise<<<numBlocks, threadsPerBlock>>>(f, in, out, out_index, get_index);
      //assert(0);
    }
}


template <typename T>
void op_sum(const Tensor<T> &in, Tensor<T> &out)
{
    Tensor<int> out_index;
    SumAccumFunc<T> f;
    if (in.on_device && out.on_device) {
        op_reduction_gpu(f, in, out, out_index, false);
    } else
        assert(0);
}

template <typename T>
void op_argmax(const Tensor<T> &in, Tensor<int> &out_index)
{
    Tensor<T> out;
    MaxAccumFunc<T> f;
    //printf("call 1\n");
    if (in.on_device && out_index.on_device) {
        op_reduction_gpu(f, in, out, out_index, true);
    } else
        assert(0);
}
