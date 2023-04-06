#include "external.h"

model_t global_model;

void create_model(size_t input_size, size_t bits_per_input, size_t num_classes, size_t filter_inputs, size_t filter_entries, size_t filter_hashes) {
    size_t num_inputs = input_size * bits_per_input;

    model_init(&global_model, num_inputs, num_classes, filter_inputs, filter_entries, filter_hashes, bits_per_input, 1);
}

void set_hashes(uint64_t* hash_values, size_t size) {
    for(size_t it = 0; it < size; ++it)
        global_model.hash_parameters.data[it] = hash_values[it];
}

void set_ordering(uint64_t* ordering, size_t size) {
    for(size_t it = 0; it < size; ++it)
        global_model.input_order[it] = ordering[it];
}

void fill_model(uint64_t* data, size_t size) {
    for(size_t it = 0; it < size; ++it) 
        global_model.data.data[it] = data[it];
}

uint64_t predict(unsigned char* input) {
    uint64_t pred = model_predict(&global_model, input);
    return pred;
}

void test_write_read() {
    char* model_path = "/Users/xavier/Desktop/Cours/Ici/WNN/Cbthowen/model.dat";
    write_model(model_path, &global_model);
    read_model(model_path, &global_model);
}

void test_read_model() {
    char* model_path = "/Users/xavier/Desktop/Cours/Ici/WNN/Cbthowen/model.dat";
    read_model(model_path, &global_model);
}
