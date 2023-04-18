#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "assert.h"

#include "math.h"
#include "distributions.h"

#include "tensor.h"

// set appropriate path for data
#define MNIST_TRAIN_IMAGE "./data/train-images-idx3-ubyte"
#define MNIST_TRAIN_LABEL "./data/train-labels-idx1-ubyte"
#define MNIST_TEST_IMAGE "./data/t10k-images-idx3-ubyte"
#define MNIST_TEST_LABEL "./data/t10k-labels-idx1-ubyte"

#define MNIST_IM_SIZE 784 // 28*28
#define MNIST_NUM_TRAIN 60000
#define MNIST_NUM_TEST 10000
#define MNIST_LEN_INFO_IMAGE 4
#define MNIST_LEN_INFO_LABEL 2

extern bmatrix_t train_images; // Of shape (#Samples, 784) -> flattened
extern bmatrix_t test_images; // Of shape (#Samples, 784) -> flattened
extern unsigned char* train_labels; // Of shape (#Samples,)
extern unsigned char* test_labels; // Of shape (#Samples,)

extern bmatrix_t binarized_train; // Of shape (#Samples, 784 * bits_per_pixel)
extern bmatrix_t binarized_test; // Of shape (#Samples, 784 * bits_per_pixel)

void load_mnist();

void binarize_matrix(bmatrix_t* result, bmatrix_t* dataset, size_t sample_size, size_t num_samples, size_t num_bits);
void binarize_mnist(size_t num_bits);

void print_binarized_mnist_image(size_t index, size_t num_bits);
void print_mnist_image(size_t index);
void print_mnist_image_raw(size_t index);

void fill_input_random(unsigned char* input, size_t input_length);
