#include "batch.h"

void batch_hashing(tensor3d_t* resulting_hashes, model_t* model, bmatrix_t* input_batch, size_t batch_size) {
    matrix_t tmp_hashes = { .stride = model->filter_hashes, .data=NULL };
    // init_matrix(&tmp_hashes, model->num_filters, model->filter_hashes);
    for(size_t it = 0; it < batch_size; ++it) {
        tmp_hashes.data = TENSOR3D_AXIS1(*resulting_hashes, it);
        perform_hashing(tmp_hashes, model, MATRIX_AXIS1(*input_batch, it));
    }
}

void batch_prediction(size_t* results, model_t* model, bmatrix_t* input_batch, size_t batch_size) {
    for(size_t it = 0; it < batch_size; ++it) {
        results[it] = model_predict2(model, MATRIX_AXIS1(*input_batch, it));
    }
}
