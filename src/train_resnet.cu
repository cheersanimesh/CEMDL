#include <getopt.h>
#include <iostream>
#include <vector>

#include "modules/resnet.cuh"
#include "modules/sgd.cuh"
#include "utils/dataset_mnist.hh"
#include "ops/op_elemwise.cuh"
#include "ops/op_cross_entropy.cuh"

unsigned long long randgen_seed = 1;
static bool on_gpu = true;

int correct(const Tensor<float> &logits, const Tensor<char> &targets) 
{
    assert(targets.w == 1);
    Tensor<int> predictions{targets.h, targets.w, on_gpu};
    op_argmax(logits, predictions);
    Tensor<int> correct_preds{targets.h, targets.w, on_gpu};
    op_equal(predictions, targets, correct_preds);
    Tensor<int> sum_correct{1,1, on_gpu};
    op_sum(correct_preds, sum_correct);
    if (on_gpu) {
        auto tmp = sum_correct.toHost();
        return Index(tmp, 0, 0);
    }
    return Index(sum_correct, 0, 0);
}

void do_one_epoch(ResNet<float>& resnet, SGD<float>& sgd, Tensor<float>& images, const Tensor<char>& targets, 
    int batch_size, bool is_training, int epoch_num) 
{
    Tensor<float> logits{batch_size, 10, on_gpu};
    Tensor<float> d_logits{batch_size, 10, on_gpu};
    Tensor3D<float> d_input_images{images.h, images.w, batch_size, on_gpu};
    int num_batches = 0, total_correct = 0;
    float total_loss = 0.0;
    for (int b = 0; b < images.h / batch_size; b++)
    {
        if ((b + 1) * batch_size > images.h)
        {
            break;
        }
        num_batches++;
        Tensor<float> b_images = images.slice(b * batch_size, (b + 1) * batch_size, 0, images.w);
        Tensor<char> b_targets = targets.slice(b * batch_size, (b + 1) * batch_size, 0, targets.w);

        resnet.forward(b_images, logits);
        float loss = op_cross_entropy_loss(logits, b_targets, d_logits);
        total_loss += loss;
        total_correct += correct(logits, b_targets);
        if (is_training) {
            resnet.backward(d_logits, b_images, d_input_images);
            sgd.step();
        }
    }

    std::cout << (is_training ? "TRAINING" : "TEST") << " epoch=" << epoch_num << " loss=" << total_loss / num_batches
              << " accuracy=" << total_correct / (float)(num_batches * batch_size)
              << " num_batches=" << num_batches << std::endl;
}

void train_and_test(int epochs, int batch_size)
{
    MNIST mnist_train{"../data/MNIST/raw", MNIST::Mode::kTrain};
    MNIST mnist_test{"../data/MNIST/raw", MNIST::Mode::kTest};
    std::cout << "# of training datapoints=" << mnist_train.images.h << " # of test datapoints= " 
              << mnist_test.images.h << " feature size=" << mnist_train.images.w << std::endl;
    std::cout << "training datapoints mean=" << mnist_train.images.mean() << std::endl;

    auto train_images = mnist_train.images;
    auto train_targets = mnist_train.targets;
    auto test_images = mnist_test.images;
    auto test_targets = mnist_test.targets;

    if (on_gpu) {
        train_images = train_images.toDevice();
        train_targets = train_targets.toDevice();
        test_images = test_images.toDevice();
        test_targets = test_targets.toDevice();
    }
    
    ResNet<float> resnet(10, on_gpu); // ResNet with 10 output classes
    SGD<float> sgd{resnet.parameters(), 0.01};

    for (int i = 0; i < epochs; i++)
    {
        do_one_epoch(resnet, sgd, train_images, train_targets, batch_size, true, i);
    }
    do_one_epoch(resnet, sgd, test_images, test_targets, batch_size, false, 0);
}

int main(int argc, char *argv[])
{
    int batch_size = 32;
    int num_epochs = 10;

    for (;;)
    {
        switch (getopt(argc, argv, "s:g:b:e:"))
        {
        case 's':
            randgen_seed = atoll(optarg);
            continue;
        case 'g':
            on_gpu = atoi(optarg) ? true : false;
            continue;
        case 'b':
            batch_size = atoi(optarg);
            continue;
        case 'e':
            num_epochs = atoi(optarg);
            continue;
        case -1:
            break;
        }
        break;
    }
    train_and_test(num_epochs, batch_size);
}

