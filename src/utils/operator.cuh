#pragma once

#include <functional>

template<typename T>
class Operator {
public:
    std::function<void()> backward_op;

    Operator(std::function<void()> func) : backward_op(func) {};

    void backward() {
        backward_op();
    }
};

