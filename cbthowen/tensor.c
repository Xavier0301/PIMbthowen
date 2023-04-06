#include "tensor.h"

void tensor_init(tensor3d_t* t, size_t shape1, size_t shape2, size_t shape3) {
    t->stride1 = shape2 * shape3;
    t->stride2 = shape3;

    t->data = (uint64_t*) calloc(shape1 * shape2 * shape3, sizeof(uint64_t));
}

void matrix_init(matrix_t* m, size_t rows, size_t cols) {
    m->stride = cols;

    m->data = (uint64_t*) calloc(rows * cols, sizeof(uint64_t));
}

void bmatrix_init(bmatrix_t* m, size_t rows, size_t cols) {
    m->stride = cols;

    m->data = (unsigned char*) calloc(rows * cols, sizeof(unsigned char));
}

void bmatrix_mean(double* mean, bmatrix_t* dataset, size_t sample_size, size_t num_samples) {
    for(size_t offset_it = 0; offset_it < sample_size; ++offset_it) 
        mean[offset_it] = 0;

    for(size_t sample_it = 0; sample_it < num_samples; ++sample_it) {
        for(size_t offset_it = 0; offset_it < sample_size; ++offset_it) {
            mean[offset_it] += *MATRIX(*dataset, sample_it, offset_it);
        }
    }

    for(size_t offset_it = 0; offset_it < sample_size; ++offset_it) 
        mean[offset_it] /= num_samples;
}

void bmatrix_variance(double* variance, bmatrix_t* dataset, size_t sample_size, size_t num_samples, double* mean) {
    for(size_t offset_it = 0; offset_it < sample_size; ++offset_it) 
        variance[offset_it] = 0;

    for(size_t sample_it = 0; sample_it < num_samples; ++sample_it) {
        for(size_t offset_it = 0; offset_it < sample_size; ++offset_it) {
            variance[offset_it] += pow(*MATRIX(*dataset, sample_it, offset_it) - mean[offset_it], 2);
        }
    }

    for(size_t offset_it = 0; offset_it < sample_size; ++offset_it) 
        variance[offset_it] /= (num_samples - 1);
}
