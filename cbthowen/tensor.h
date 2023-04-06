#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

typedef uint64_t entry_t;

typedef struct {
    size_t axis1;
    size_t axis2;
    size_t axis3;
} tensor_index_t;

typedef struct {
    size_t stride1;
    size_t stride2;
    entry_t* data;
} tensor3d_t;

#define TENSOR3D_AXIS1(t, i) ((t).data + i * (t).stride1)
#define TENSOR3D_AXIS2(t, i, j) (TENSOR3D_AXIS1(t, i) + j * (t).stride2)
#define TENSOR3D(t, i, j, k) (TENSOR3D_AXIS2(t, i, j) + k)
#define TENSOR3D_(t, index) (TENSOR3D(t, index.axis1, index.axis2, index.axis2))

void tensor_init(tensor3d_t* t, size_t shape1, size_t shape2, size_t shape3);

typedef struct {
    size_t axis1;
    size_t axis2;
} matrix_index_t;

typedef struct {
    size_t stride;
    uint64_t* data;
} matrix_t;

#define MATRIX_AXIS1(t, i) ((t).data + i * (t).stride)
#define MATRIX(t, i, j) (MATRIX_AXIS1(t, i) + j)
#define MATRIX_(t, index) (MATRIX(t, index.axis1, index.axis2))

void matrix_init(matrix_t* m, size_t rows, size_t cols);

typedef struct {
    size_t stride;
    unsigned char* data;
} bmatrix_t;

void bmatrix_init(bmatrix_t* m, size_t rows, size_t cols);

void bmatrix_mean(double* mean, bmatrix_t* dataset, size_t sample_size, size_t num_samples);
void bmatrix_variance(double* variance, bmatrix_t* dataset, size_t sample_size, size_t num_samples, double* mean);

#endif
