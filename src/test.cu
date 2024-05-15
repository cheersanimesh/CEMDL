#include <getopt.h>
#include <iostream>
#include "layers/flattenLayer.cuh"
#include "layers/pooling.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"
#include "ops/op_cross_entropy.cuh"
#include "utils/tensor.cuh"
#include "utils/tensor3D.cuh"

unsigned long long randgen_seed = 0;

void test_matmul(int m, int n, int k, bool on_gpu)
{
    Tensor<float> A{m, k, on_gpu};
    op_uniform_init(A);

    Tensor<float> B{k, n, on_gpu};
    op_uniform_init(B);

    Tensor<float> C{m, n, on_gpu};
    op_mm(A, B, C);

    Tensor<float> C2{n, m, on_gpu};
    op_mm(B.transpose(), A.transpose(), C2);
    assert(op_allclose(C2.transpose(), C)); // test transpose

    std::cout << "matmul passed..." << std::endl;
}

void test_elemwise(int m, int n, bool on_gpu)
{
    Tensor<float> X{m, n, on_gpu};
    op_const_init(X, 2.0);

    Tensor<float> Y{m, n, on_gpu};
    op_const_init(Y, 3.0);

    Tensor<float> Z{m, n, on_gpu};
    op_add(X, Y, Z);

    Tensor<float> Zref{m, n, false};
    op_const_init(Zref, 5.0);
    assert(op_allclose(Z, Zref));

    Tensor<float> Y2{1, n, on_gpu};
    op_const_init(Y2, 3.0);
    op_add(X, Y2, Z); //test broadcasting
    assert(op_allclose(Z, Zref));

    op_add<float>(X, 3.0, Z);
    assert(op_allclose(Z, Zref));

    std::cout << "op_add passed..." << std::endl;

    op_multiply(X, Y, Z);

    op_const_init(Zref, 6.0);
    assert(op_allclose(Z, Zref));

    op_multiply(X, Y2, Z);
    assert(op_allclose(Z, Zref));

    op_multiply<float>(X, 3.0, Z);
    assert(op_allclose(Z, Zref));

    std::cout << "op_multiply passed..." << std::endl;

    float lr = 0.02;
    Tensor<float> A{m, n, on_gpu};
    op_uniform_init(A);
    Tensor<float> A_host = A.toHost();

    Tensor<float> dA{m, n, on_gpu};
    op_uniform_init(dA);
    Tensor<float> dA_host = dA.toHost();

    Tensor<float> Aref{m, n, false};
    for (int i = 0; i < Aref.h; i++)
    {
        for (int j = 0; j < Aref.w; j++)
        {
          Index(Aref, i, j) = Index(A_host, i, j) - lr * Index(dA_host, i, j);
        }
    }
    op_sgd(A, dA, A, lr);
    assert(op_allclose(A, Aref));

    std::cout << "op_sgd passed..." << std::endl;

}

bool is_close_enough(float a, float b) {
    return std::abs(a - b) <= 0.0001;
}

void assert_all_close_enough(Tensor<float> t, std::vector<float> v)
{
    for (int i = 0; i < t.h; i++) {
        for (int j = 0; j < t.w; j++) {
            assert(is_close_enough(Index(t, i, j), v[i * t.w + j]));
        }
    }
}

void test_op_cross_entropy_loss(bool on_gpu)
{
    Tensor<float> logits_host{2, 3};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            Index(logits_host, i, j) = i * 3 + j;
        }
    }
    Tensor<char> targets{2, 1};
    for (int i = 0; i < 2; i++) {
        Index(targets, i, 0) = i;
    }
    Tensor<float> logits = logits_host;
    if (on_gpu) {
        logits = logits.toDevice();
        targets = targets.toDevice();
    }
    Tensor<float> d_logits{2, 3, on_gpu};
    float loss = op_cross_entropy_loss(logits, targets, d_logits);
    assert(is_close_enough(loss, 1.9076));

    Tensor<float> d_logits_host = d_logits;
    if (on_gpu) {
        d_logits_host = d_logits.toHost();
    }
    std::vector<float> d_logits_ref{-0.4550, 0.1224, 0.3326, 0.0450, -0.3776, 0.3326};

    assert_all_close_enough(d_logits_host, d_logits_ref);

    //test if the save version is implemented
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            Index(logits_host, i, j) += 100000;
        }
    }
    if (on_gpu) {
        logits = logits_host.toDevice();
    }
    loss = op_cross_entropy_loss(logits, targets, d_logits);
    if (on_gpu) {
        d_logits_host = d_logits.toHost();
    }
    assert_all_close_enough(d_logits_host, d_logits_ref);

    std::cout << "op_cross_entropy_loss passed..." << std::endl;

}

void test_reduction(int m, int n, bool on_gpu)
{
    Tensor<int> X_host{m, n};
    op_const_init(X_host, 0);

    int reduce_sum = m > n ? n : m;
    for (int i = 0; i < X_host.h; i++) 
    {
        if (i >= X_host.w) {
            break;
        }
        Index(X_host, i, i) = 1;
    }

    Tensor<int> X;
    if (on_gpu) {
        X = X_host.toDevice();
    } else {
        X = X_host;
    } 
    Tensor<int> Y{1, n, on_gpu};
    op_sum(X, Y);

    Tensor<int> Yref{1, n};
    op_const_init(Yref, 0);
    for (int j = 0; j < reduce_sum; j++) {
        Index(Yref, 0, j) = 1;
    }
    Tensor<int> Y_host = Y.toHost();
    assert(op_allclose(Y, Yref));

    Tensor<int> Y1{1, 1, on_gpu};
    op_sum(Y, Y1);
    Tensor<int> Y1_host = Y1.toHost();
    assert(Index(Y1_host, 0, 0) == reduce_sum);

    op_const_init(X, 1);
    op_sum(X, Y);
    op_const_init(Yref, X.h);
    assert(op_allclose(Y, Yref));
    
    Tensor<int> YY{m, 1, on_gpu};
    op_sum(X, YY);
    Tensor<int> YYref{m, 1};
    op_const_init(YYref, n);
    op_sum(YY, Y1);
    Y1_host = Y1.toHost();
    assert(Index(Y1_host, 0, 0) == m * n);

    std::cout << "op_sum passed..." << std::endl;

    //try to create an A matrix whose last column has the biggest value
    Tensor<float> A{m, n, on_gpu};
    op_uniform_init<float>(A, 0.0, 1.0);
    auto AA = A.slice(0, A.h, A.w - 1, A.w);
    op_add<float>(AA, 10.0, AA);

    Tensor<int> ind{m, 1, on_gpu};
    op_argmax(A, ind);
    Tensor<int> indref{m, 1};
    op_const_init(indref, n - 1);
    assert(op_allclose(ind, indref));
}

void test_views()
{
    Tensor<float> A{5, 5};
    for (int i = 0; i < A.h; i++) {
        for (int j = 0; j < A.w; j++) {
            Index(A, i, j) = i * A.w + j;
        }
    }
    auto B = A.slice(1, 3, 1, 3);
    assert(Index(B, 0, 0) == 6);
    assert(Index(B, 0, 1) == 7);
    assert(Index(B, 1, 0) == 11);
    assert(Index(B, 1, 1) == 12);
    auto C = B.transpose();
    assert(Index(C, 0, 0) == 6);
    assert(Index(C, 0, 1) == 11);
    assert(Index(C, 1, 0) == 7);
    assert(Index(C, 1, 1) == 12);
    std::cout << "slice passed..." << std::endl;
}

void test_flatten(int batch_size, int height, int width, bool on_gpu) {
    // Create a 3D tensor to represent a batch of 2D images
    Tensor3D<float> x(height, width, batch_size, on_gpu);

    // Calculate the expected width of the flattened tensor
    int expected_w = height * width;
    Tensor<float> expected(batch_size, expected_w);

    // Manually flattening the tensor using 3D indexing
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int index = i * width + j;
                Index(expected, b, index) = Index3D(x, i, j, b);
            }
        }
    }

    // Assuming FlattenLayer is correctly defined and included
    FlattenLayer<float> flatten;
    Tensor<float> y;
    flatten.forward(x, y); // Flatten the tensor using the FlattenLayer

    // Check if the output tensor y is equal to the expected tensor
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < expected_w; i++) {
            assert(Index(y, b, i) == Index(expected, b, i)); // Compare using the Index macro
        }
    }

    std::cout << "flatten test passed..." << std::endl;
}

void test_pooling() {
    Tensor3D<float> input(4, 4, 1, false);
    int idx = 0;
    for (int i = 0; i < input.d; i++) {
        for (int j = 0; j < input.h; j++) {
            for (int k = 0; k < input.w; k++) {
                input.values[i].rawp[idx++] = i * 16 + j * 4 + k;
            }
        }
    }

    int kernel_size = 2;
    Tensor3D<float> output = max_pool2d_forward(input, kernel_size);
    Tensor3D<float> grad_input = max_pool2d_backward(output, input, kernel_size);

    bool maxPoolCorrect = true;
    idx = 0;
    for (int i = 0; i < output.d; i++) {
        for (int j = 0; j < output.h; j++) {
            for (int k = 0; k < output.w; k++) {
                if (is_close_enough(output.values[i].rawp[idx], static_cast<float>(i * 0 + j * 6 + k * 2 + 3))) {
                    maxPoolCorrect = false;
                    break;
                }
                idx++;
            }
        }
    }

    bool gradInputCorrect = true;
    idx = 0;
    for (int i = 0; i < grad_input.d; i++) {
        for (int j = 0; j < grad_input.h; j++) {
            for (int k = 0; k < grad_input.w; k++) {
                if ((j == 1 && k == 1) || (j == 1 && k == 2) || (j == 2 && k == 1) || (j == 2 && k == 2)) {
                    if (is_close_enough(grad_input.values[i].rawp[idx++], 1.0f)) {
                        gradInputCorrect = false;
                        break;
                    }
                }
            }
        }
    }

    std::cout << "max pooling forward test: " << (maxPoolCorrect ? "passed..." : "failed!") << std::endl;
    std::cout << "max pooling backward test: " << (gradInputCorrect ? "passed..." : "failed!") << std::endl;
}

int main(int argc, char *argv[])
{
    bool test_gpu = true;
    int test_m = 335, test_n = 587, test_k= 699;

    for (;;)
    {
        switch (getopt(argc, argv, "s:ch:l:b:e:"))
        {
        case 's':
            randgen_seed = atoll(optarg);
            continue;
        case 'c': //cpu testing only
            test_gpu = false;
            continue;
        case 'm':
            test_m = atoi(optarg);
            continue;
        case 'n':
            test_n = atoi(optarg);
            continue;
        case 'k':
            test_k = atoi(optarg);
            continue;
        case -1:
            break;
        }
        break;
    }

    test_views();
    test_elemwise(test_m, test_n, test_gpu);
    test_matmul(test_m, test_n, test_k, test_gpu);
    test_reduction(test_m, test_n, test_gpu);
    test_op_cross_entropy_loss(test_gpu);
    test_flatten(test_m, test_n, test_k, test_gpu);
    test_pooling();
    std::cout << "All tests completed successfully!" << std::endl;
    return 0;
}

