#include "model.h"

element_t* reorder_buffer;

void set_reorder_buffer(element_t* input, size_t* order, size_t len) {
    for(size_t it = 0; it < len; ++it)
        reorder_buffer[it] = input[order[it]];
}

void randomize_input_order(size_t* input_order, size_t len) {
    for(size_t it = 0; it < len; ++it) {
        input_order[it] = it;
    }
    shuffle_array(input_order, len);
}

void generate_h3_values(matrix_t* values, size_t num_hashes, size_t num_inputs, size_t num_entries) {
    for(size_t i = 0; i < num_hashes; ++i) {
        for(size_t j = 0; j < num_inputs; ++j) {
            *MATRIX(*values, i, j) = unif_rand(num_entries - 1);
        }
    }
}

void model_init(model_t* model, size_t num_inputs, size_t num_classes, size_t filter_inputs, size_t filter_entries, size_t filter_hashes, size_t bits_per_input, unsigned char bleach) {
    model->pad_zeros = (((num_inputs / filter_inputs) * filter_inputs) - num_inputs) % filter_inputs;
    model->num_inputs_total = num_inputs + model->pad_zeros;
    model->bits_per_input = bits_per_input;
    model->num_classes = num_classes;

    model->num_filters = num_inputs / filter_inputs;
    model->filter_inputs = filter_inputs;
    model->filter_entries = filter_entries;
    model->filter_hashes = filter_hashes;

    model->bleach = bleach;

    model_init_buffers(model);
}

void model_init_buffers(model_t* model){
    model->input_order = calloc(model->num_inputs_total, sizeof(uint64_t));
    randomize_input_order(model->input_order, model->num_inputs_total);

    tensor_init(&model->data, model->num_classes, model->num_filters, model->filter_entries);
    
    matrix_init(&model->hash_parameters, model->filter_hashes, model->filter_inputs);
    generate_h3_values(&model->hash_parameters, model->filter_hashes, model->filter_inputs, model->filter_entries);

    reorder_buffer = calloc(model->num_inputs_total, sizeof(uint64_t));
}

// assumes input is already zero padded
uint64_t model_predict(model_t* model, element_t* input) {
    set_reorder_buffer(input, model->input_order, model->num_inputs_total);

    size_t response_index = 0;
    uint64_t max_response = 0;
    for(size_t it = 0; it < model->num_classes; ++it) {
        uint64_t resp = discriminator_predict(model, it, reorder_buffer);
        if(resp >= max_response) {
            max_response = resp;
            response_index = it;
        }
    }

    return response_index;
}

void model_train(model_t* model, element_t* input, uint64_t target) {    
    set_reorder_buffer(input, model->input_order, model->num_inputs_total);

    discriminator_train(model, target, reorder_buffer);
}

uint64_t discriminator_predict(model_t* model, size_t discriminator_index, element_t* input) {
    uint64_t response = 0;
    element_t* chunk = input;

    for(size_t it = 0; it < model->num_filters; ++it) {
        response += filter_check_membership(model, discriminator_index, it, chunk);
        chunk += model->filter_inputs;
    }

    return response;
}

void discriminator_train(model_t* model, size_t discriminator_index, element_t* input) {
    element_t* chunk = input;

    for(size_t it = 0; it < model->num_filters; ++it) {
        filter_add_member(model, discriminator_index, it, chunk);
        chunk += model->filter_inputs;
    }
}

/**
 * @brief 
 * 
 * @param result Resulting hash
 * @param input Boolean vector of shape (#inputs)
 * @param parameters Vector of shape (#inputs)
 */
uint64_t h3_hash(element_t* input, uint64_t* parameters, size_t num_inputs, size_t num_hashes) {
    uint64_t result = parameters[0] * input[0];
    for(size_t j = 1; j < num_inputs; ++j) {
        result ^= parameters[j] * input[j];
    }

    return result;
}

// Can be replaced by an AND reduction (ONLY WHEN BLEACH=1)
uint64_t filter_check_membership(model_t* model, size_t discriminator_index, size_t filter_index, element_t* input) {
    uint64_t hash_result;
    entry_t entry;

    uint64_t minimum = -1;
    for(size_t it = 0; it < model->filter_hashes; ++it) {
        hash_result = h3_hash(input, MATRIX_AXIS1(model->hash_parameters, it), model->filter_inputs, model->filter_hashes);
        entry = *TENSOR3D(model->data, discriminator_index, filter_index, hash_result);
        if(entry <= minimum) minimum = entry;
    }

    return minimum >= model->bleach;
}

void filter_add_member(model_t* model, size_t discriminator_index, size_t filter_index, element_t* input) {
    uint64_t hash_result;
    entry_t entry;

    // Get minimum of all filter hash response
    uint64_t minimum = -1;
    for(size_t it = 0; it < model->filter_hashes; ++it) {
        hash_result = h3_hash(input, MATRIX_AXIS1(model->hash_parameters, it), model->filter_inputs, model->filter_hashes);
        entry = *TENSOR3D(model->data, discriminator_index, filter_index, hash_result);
        if(entry <= minimum) minimum = entry;
    }

    // Increment the value of all minimum entries
    for(size_t it = 0; it < model->filter_hashes; ++it) {
        hash_result = h3_hash(input, MATRIX_AXIS1(model->hash_parameters, it), model->filter_inputs, model->filter_hashes);
        entry = *TENSOR3D(model->data, discriminator_index, filter_index, hash_result);
        if(entry == minimum) 
            *TENSOR3D(model->data, discriminator_index, filter_index, hash_result) = minimum + 1;
    }
}
