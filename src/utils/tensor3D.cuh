#pragma once
#include <assert.h>
#include <random>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "utils/assert.cuh"
#include "utils/tensor.cuh"

#define ISCLOSE_RELTOL 1e-6 // this would not work for precision lower than float
#define ISCLOSE_ABSTOL 1e-6

//Index is a MACRO that returns the element of t tensor at row, col coordinate
//#define Index3D(t, row, col, chan) ((((t).rawp) [(t).offset + (chan) * (t).stride_d + (row) * (t).stride_h + (col) * (t).stride_w]))
#define Index3D(t, row, col, depth) (((t).values[(depth)].rawp)[(t).values[(depth)].offset + (row) * (t).values[(depth)].stride_h + (col) * (t).values[(depth)].stride_w])
//IndexOutofBound is a MACRO to test whether coordinate row,col is considered out of bounds for tensor "t"
//#define IndexOutofBound(t, row, col, chan) ((((row) >= (t).h) || ((col) >= (t).w) || ((chan) >= (t).d)))


template <typename T>
class Tensor3D
{
public:
  int32_t h; // height
  int32_t w; // width
  int32_t d; // depth
  int32_t stride_h;
  int32_t stride_w;
  int32_t stride_d;
  int32_t offset;
  T *rawp;

  std::vector<Tensor<T>> values;
  
  std::shared_ptr<T> ref; // refcounted pointer, for garbage collection use only
  bool on_device;

  Tensor3D() : h(0), w(0),d(0), stride_h(0), stride_w(0), offset(0), rawp(nullptr), on_device(false)
  {
    ref = std::shared_ptr<T>(rawp, cpuDeleter<T>());
  }

  Tensor3D(int32_t h_, int32_t w_, int32_t d_, bool on_device_ = false)
      : h(h_), w(w_),d(d_), stride_h(w_), stride_w(1), stride_d(h_*w_), offset(0), on_device(on_device_)
  {
    if (on_device_)
    {
      for(int i=0;i<d;i++){
        Tensor<T> temp{h, w, on_device};
        values.push_back(temp);
      }
    }
    else
    {
      for(int i=0;i<d;i++){
        Tensor<T> temp{h, w, false};
        values.push_back(temp);
      }
    }
  }

  void toHost(Tensor3D<T> &out) const
  {
    assert(!out.on_device);
    assert(h == out.h && w == out.w && d == out.d);
    if (!on_device) {
      out = *this;
      return;
    }
    out.offset = offset;
    out.stride_h = stride_h;
    out.stride_w = stride_w;
    out.stride_d = stride_d;

    for(int i=0;i<d;i++){
      cudaAssert(cudaMemcpy(out.values[i].rawp, values[i].rawp, h * w * sizeof(T), cudaMemcpyDeviceToHost));
    }
  }

  Tensor3D<T> toHost() const
  {
    Tensor3D<T> t{h, w, d};
    toHost(t);
    return t;
  }

  void toDevice(Tensor3D<T> &out) const
  {
    assert(out.on_device);
    assert(h == out.h && w == out.w && d==out.d);
    if (on_device) {
      out = *this;
      return;
    }
    out.offset = offset;
    out.stride_h = stride_h;
    out.stride_w = stride_w;
    out.stride_d = stride_d;

    for(int i=0;i<d;i++){
      cudaAssert(cudaMemcpy(out.values[i].rawp, values[i].rawp, h * w * d * sizeof(T), cudaMemcpyHostToDevice));
    }
  }

  Tensor3D<T> toDevice() const
  {
    Tensor3D<T> t{h, w, d, true};
    toDevice(t);
    return t;
  }

  // Tensor3D<T> transpose() const
  // {
  //   Tensor3D<T> t{};
  //   t.w = h;
  //   t.stride_w = stride_h;
  //   t.h = w;
  //   t.stride_h = stride_w;
  //   t.offset = offset;
  //   t.ref = ref;
  //   t.rawp = rawp;
  //   t.on_device = on_device;
  //   return t;
  // }

  // Tensor<T> slice(int start_h, int end_h, int start_w, int end_w) const
  // {
  //   Tensor<T> t{};
  //   assert(start_h < end_h && end_h <= h);
  //   assert(start_w < end_w && end_w <= w);
  //   t.w = end_w - start_w;
  //   t.h = end_h - start_h;
  //   t.stride_h = stride_h;
  //   t.stride_w = stride_w;
  //   t.ref = ref;
  //   t.rawp = rawp;
  //   t.offset = offset + start_h * stride_h + start_w * stride_w;
  //   t.on_device = on_device;
  //   return t;
  // }

  std::string str() const
  {
    Tensor3D<T> t{};
    if (on_device)
    {
      t = toHost();
    }
    else
    {
      t = *this;
    }
    std::stringstream ss;
    for (int i = 0; i < h; i++)
    {
      for (int j = 0; j < w; j++)
      {
        for(int z = 0; z < d; z++)
        {
          if (std::is_same_v<T, char> || std::is_same_v<T, unsigned char>)
          {
            ss << (int)Index3D(t, i, j, z) << " ";
          }
          else
          {
          // std::cout << "haha " << Index(t, i, j) << std::endl;
            ss << Index3D(t, i, j, z) << " ";
          }
          ss << "";
        }
        ss << "\n";
      }
      ss << "\n";
    }
    return ss.str();
  }

  // T mean() const 
  // {
  //   assert(!on_device);
  //   T sum = 0;
  //   for (int i = 0; i < h; i++) {
  //     for (int j = 0; j < w; j++) {
  //       sum += Index(*this, i, j);
  //     }
  //   }
  //   return sum/(h*w);
  // }

  // T range() const 
  // {
  //   assert(!on_device);
  //   T min, max;
  //   min = max = Index(*this, 0, 0);
  //   for (int i = 0; i < h; i++) {
  //     for (int j = 0; j < w; j++) {
  //       T e = Index(*this, i, j);
  //       if (e > max) {
  //         max = e;
  //       } else if (e < min) {
  //         min = e;
  //       }
  //     }
  //   }
  //   return max-min;
  // }
};
