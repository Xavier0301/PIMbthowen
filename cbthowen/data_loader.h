#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "assert.h"

#include "math.h"
#include "distributions.h"

#include "tensor.h"
#include "model.h"

// set appropriate path for data
#define MNIST_TRAIN_IMAGE "./data/train-images-idx3-ubyte"
#define MNIST_TRAIN_LABEL "./data/train-labels-idx1-ubyte"
#define MNIST_TEST_IMAGE "./data/t10k-images-idx3-ubyte"
#define MNIST_TEST_LABEL "./data/t10k-labels-idx1-ubyte"

#define INFIMNIST_PATTERNS "./data/mnist8m-patterns-idx3-ubyte"
#define INFIMNIST_LABELS "./data/mnist8m-labels-idx1-ubyte"

#define MNIST_IM_SIZE 784 // 28*28
#define MNIST_SIDE_LEN 28
#define INFIMNIST_NUM_SAMPLES 8000000
#define MNIST_NUM_TRAIN 60000
#define MNIST_NUM_TEST 10000
#define MNIST_LEN_INFO_IMAGE 4
#define MNIST_LEN_INFO_LABEL 2

void load_mnist_file(bmatrix_t* patterns, unsigned char* labels, char* image_path, char* label_path, size_t num_samples);
void load_mnist_train(bmatrix_t* patterns, unsigned char* labels, size_t num_samples);
void load_mnist_test(bmatrix_t* patterns, unsigned char* labels, size_t num_samples);
void load_infimnist(bmatrix_t* patterns, unsigned char* labels, size_t num_samples);

void binarize_matrix(bmatrix_t* result, bmatrix_t* dataset, size_t sample_size, size_t num_samples, size_t num_bits);

void reorder_dataset(bmatrix_t* result, bmatrix_t* dataset, size_t* order, size_t num_samples, size_t num_elements);

void print_binarized_image_raw(bmatrix_t* m, unsigned char* labels, size_t index, size_t num_bits);
void print_binarized_image(bmatrix_t* m, unsigned char* labels, size_t index, size_t num_bits);
void print_image_raw(bmatrix_t* m, unsigned char* labels, size_t index);
void print_image(bmatrix_t* m, unsigned char* labels, size_t index);

void fill_input_random(unsigned char* input, size_t input_length);

