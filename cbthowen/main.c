#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "model.h"
#include "tensor.h"
#include "data_loader.h"
#include "model_manager.h"

#include "batch.h"

void train() {
    model_t model;

    size_t input_size = 784;
    size_t bits_per_input = 2;
    size_t num_inputs = input_size * bits_per_input;

    size_t num_classes = 10;

    size_t filter_inputs = 28;
    size_t filter_entries = 1024;
    size_t filter_hashes = 2;

    model_init(&model, num_inputs, num_classes, filter_inputs, filter_entries, filter_hashes, bits_per_input, 1);

    printf("Loading dataset\n");
    load_mnist();

    printf("Binarizing dataset\n");
    binarize_mnist(bits_per_input);

    print_binarized_mnist_image(7555, 2);

    printf("Training\n");

    for(size_t sample_it = 0; sample_it < MNIST_NUM_TRAIN; ++sample_it) {
        model_train(&model, MATRIX_AXIS1(binarized_train, sample_it), train_labels[sample_it]);
        if(sample_it % 10000 == 0)
            printf("    %zu\n", sample_it);
    }

    printf("Testing\n");
    model.bleach = 10; // optimal bleaching threshold

    size_t correct = 0;
    for(size_t sample_it = 0; sample_it < MNIST_NUM_TEST; ++sample_it) {
        size_t class = model_predict(&model, MATRIX_AXIS1(binarized_test, sample_it));
        correct += (class == test_labels[sample_it]);
    }

    double accuracy = ((double) correct) / ((double) MNIST_NUM_TEST);
    printf("Accuracy %zu/%d (%f%%)\n", correct, MNIST_NUM_TEST, 100 * accuracy);

    write_model("model.dat", &model);
}

void load_and_test() {
    model_t model;

    printf("Loading model\n");

    read_model("model.dat", &model);

    printf("Loading dataset\n");
    load_mnist();

    printf("Binarizing dataset with %zu bits per input\n", model.bits_per_input);
    binarize_mnist(model.bits_per_input);

    print_binarized_mnist_image(7555, 2);

    printf("Testing with bleach %d\n", model.bleach);

    size_t correct = 0;
    for(size_t sample_it = 0; sample_it < MNIST_NUM_TEST; ++sample_it) {
        size_t class = model_predict2(&model, MATRIX_AXIS1(binarized_test, sample_it));
        correct += (class == test_labels[sample_it]);
    }

    double accuracy = ((double) correct) / ((double) MNIST_NUM_TEST);
    printf("Accuracy %zu/%d (%f%%)\n", correct, MNIST_NUM_TEST, 100 * accuracy);

}

void compare() {
    model_t model;

    printf("Loading model\n");

    read_model("model.dat", &model);

    printf("Loading dataset\n");
    load_mnist();

    printf("Binarizing dataset with %zu bits per input\n", model.bits_per_input);
    binarize_mnist(model.bits_per_input);

    print_binarized_mnist_image(7555, 2);

    printf("Testing with bleach %d\n", model.bleach);

    size_t agree = 0;
    for(size_t sample_it = 0; sample_it < MNIST_NUM_TEST; ++sample_it) {
        size_t class1 = model_predict(&model, MATRIX_AXIS1(binarized_test, sample_it));
        size_t class2 = model_predict2(&model, MATRIX_AXIS1(binarized_test, sample_it));
        agree += (class1 == class2);
    }
    printf("Agreeing: %lf%%\n", 100 * ((double) agree) / MNIST_NUM_TEST);
}

void test_batching() {
    model_t model;

    printf("Loading model\n");

    read_model("model.dat", &model);

    printf("Loading dataset\n");
    load_mnist();

    printf("Binarizing dataset with %zu bits per input\n", model.bits_per_input);
    binarize_mnist(model.bits_per_input);

    print_binarized_mnist_image(7555, 2);

    printf("Testing with bleach %d\n", model.bleach);

    size_t batch_size = 30;
    size_t agree = 0;

    size_t* results = calloc(batch_size, sizeof(*results));
    size_t* results_batch = calloc(batch_size, sizeof(*results_batch));

    batch_prediction(results_batch, &model, &binarized_test, batch_size);

    for(size_t sample_it = 0; sample_it < batch_size; ++sample_it) {
        results[sample_it] = model_predict2(&model, MATRIX_AXIS1(binarized_test, sample_it));
        agree += (results[sample_it] == results_batch[sample_it]);
    }
    printf("Agreeing: %lf%%\n", 100 * ((double) agree) / batch_size);
}

void test_reordering_dataset() {
    model_t model;

    printf("Loading model\n");

    read_model("model.dat", &model);

    printf("Loading dataset\n");
    load_mnist();

    printf("Binarizing dataset with %zu bits per input\n", model.bits_per_input);
    binarize_mnist(model.bits_per_input);

    print_binarized_mnist_image(7555, 2);

    printf("Reordering dataset\n");
    reorder_binarized_mnist(model.input_order, model.bits_per_input);

    printf("Testing with bleach %d\n", model.bleach);

    size_t agree = 0;

    unsigned char* reordered_sample = (unsigned char *) calloc(MNIST_IM_SIZE * model.bits_per_input, sizeof(*binarized_test.data));

    for(size_t sample_it = 0; sample_it < MNIST_NUM_TEST; ++sample_it) {
        reorder_array(reordered_sample, MATRIX_AXIS1(binarized_test, sample_it), model.input_order, model.num_inputs_total);
        
        if(memcmp(reordered_sample, MATRIX_AXIS1(reordered_binarized_test, sample_it), MNIST_IM_SIZE * model.bits_per_input * sizeof(*reordered_binarized_test.data)) == 0) {
            agree += 1;
        }
    }
    printf("Reorder agreeing: %lf%%\n", 100 * ((double) agree) / MNIST_NUM_TEST);
}

int main(int argc, char *argv[]) {                              

    /* Error Checking */
    if(argc < 2) {
        printf("Error: usage: %s 0..4.\n\t \
        0 is for training from scratch\n\t \
        1 is for loading model.dat and testing\n\t \
        2 is for comparing predict1 and predict2\n\t \
        3 is for testing batching\n\t \
        4 if for testing reordering dataset\n",
                argv[0]);
        exit(1);
    }

    if(argv[1][0] == '0')
        train();
    else if(argv[1][0] == '1')
        load_and_test();
    else if(argv[1][0] == '2')
        compare();
    else if(argv[1][0] == '3')
        test_batching();
    else 
        test_reordering_dataset();

    return 0;
}
