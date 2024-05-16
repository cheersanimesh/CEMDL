#pragma once

template<typename T>
class Operator {
public:
    std::function<void()> backward_op;

    Op(std::function<void()> func) : backward_op(func) {};

    void backward() {
        backward_op();
    }
};