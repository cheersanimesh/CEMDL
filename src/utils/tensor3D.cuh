#pragma once
#include <cmath> // for std::abs
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

// Define isClose function template
template <typename T>
bool isClose(T a, T b, T rel_tol = ISCLOSE_RELTOL, T abs_tol = ISCLOSE_ABSTOL) {
    return std::abs(a - b) <= (abs_tol + rel_tol * std::abs(b));
}

// Index is a MACRO that returns the element of t tensor at row, col, depth coordinate
#define Index3D(t, row, col, depth) (((t).values[(depth)].rawp)[(t).values[(depth)].offset + (row) * (t).values[(depth)].stride_h + (col) * (t).values[(depth)].stride_w])
// IndexOutofBound is a MACRO to test whether coordinate row, col, depth is considered out of bounds for tensor "t"
#define IndexOutofBound3D(t, row, col, depth) ((((row) >= (t).h) || ((col) >= (t).w) || ((depth) >= (t).d)))

template <typename T>
class Tensor3D {
public:
    int32_t h; // height
    int32_t w; // width
    int32_t d; // depth
    int32_t stride_h;
    int32_t stride_w;
    int32_t stride_d;
    int32_t offset;
    std::vector<Tensor<T>> values;
    std::shared_ptr<T> ref; // refcounted pointer, for garbage collection use only
    bool on_device;

    std::vector<Tensor<T>> gradients;

    Tensor3D() : h(0), w(0), d(0), stride_h(0), stride_w(0), stride_d(0), offset(0), on_device(false) {}

    Tensor3D(int32_t h_, int32_t w_, int32_t d_, bool on_device_ = false)
        : h(h_), w(w_), d(d_), stride_h(w_), stride_w(1), stride_d(h_ * w_), offset(0), on_device(on_device_) {
        for (int i = 0; i < d; i++) {
            Tensor<T> temp{h, w, on_device};
            values.push_back(temp);
        }
    }

    void toHost(Tensor3D<T>& out) const {
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

        for (int i = 0; i < d; i++) {
            cudaAssert(cudaMemcpy(out.values[i].rawp, values[i].rawp, h * w * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    Tensor3D<T> toHost() const {
        Tensor3D<T> t{h, w, d};
        toHost(t);
        return t;
    }

    void toDevice(Tensor3D<T>& out) const {
        assert(out.on_device);
        assert(h == out.h && w == out.w && d == out.d);
        if (on_device) {
            out = *this;
            return;
        }
        out.offset = offset;
        out.stride_h = stride_h;
        out.stride_w = stride_w;
        out.stride_d = stride_d;

        for (int i = 0; i < d; i++) {
            cudaAssert(cudaMemcpy(out.values[i].rawp, values[i].rawp, h * w * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    Tensor3D<T> toDevice() const {
        Tensor3D<T> t{h, w, d, true};
        toDevice(t);
        return t;
    }

    Tensor3D<T> transpose() const {
        Tensor3D<T> t{w, h, d, on_device};
        for (int i = 0; i < d; i++) {
            t.values[i] = values[i].transpose();
        }
        return t;
    }

    Tensor3D<T> slice(int start_h, int end_h, int start_w, int end_w, int start_d, int end_d) const {
        Tensor3D<T> t{};
        assert(start_h < end_h && end_h <= h);
        assert(start_w < end_w && end_w <= w);
        assert(start_d < end_d && end_d <= d);
        t.h = end_h - start_h;
        t.w = end_w - start_w;
        t.d = end_d - start_d;
        t.stride_h = stride_h;
        t.stride_w = stride_w;
        t.stride_d = stride_d;
        t.on_device = on_device;

        for (int i = start_d; i < end_d; i++) {
            t.values.push_back(values[i].slice(start_h, end_h, start_w, end_w));
        }
        return t;
    }

    std::string str() const {
        Tensor3D<T> t = (on_device) ? toHost() : *this;
        std::stringstream ss;
        for (int z = 0; z < d; z++) {
            ss << "Depth " << z << ":\n";
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    if (std::is_same_v<T, char> || std::is_same_v<T, unsigned char>) {
                        ss << (int)Index3D(t, i, j, z) << " ";
                    } else {
                        ss << Index3D(t, i, j, z) << " ";
                    }
                }
                ss << "\n";
            }
            ss << "\n";
        }
        return ss.str();
    }

    T mean() const {
        assert(!on_device);
        T sum = 0;
        for (int i = 0; i < d; i++) {
            sum += values[i].mean();
        }
        return sum / (h * w * d);
    }

    T range() const {
        assert(!on_device);
        T min, max;
        min = max = Index3D(*this, 0, 0, 0);
        for (int z = 0; z < d; z++) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    T e = Index3D(*this, i, j, z);
                    if (e > max) {
                        max = e;
                    } else if (e < min) {
                        min = e;
                    }
                }
            }
        }
        return max - min;
    }
};

