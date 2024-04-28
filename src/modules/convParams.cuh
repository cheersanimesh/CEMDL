#pragma once

#include "utils/tensor.cuh"
#include "ops/op_elemwise.cuh"
#include "utils/tensor3D.cuh"

template <typename T>
class ConvParameter{

	public:

	vector<Tensor3D<T>> weights;
	vector<Tensor3D<T>> d_weights;
	int num_filters;

	
	ConvParameter(int height, int width, int depth, int num_filters_, bool gpu) : num_filters(num_filters_){
		for(int i=0;i<num_filters;i++){
			weights.push_back(Tensor3D<T>{height, width, depth, gpu});
			d_weights.push_back(Tensor3D<T>{height, width, depth, gpu});
		}
	}
}