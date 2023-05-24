#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "model.h"
#include "data_manager.h"

extern model_t global_model;

void create_model(size_t input_size, size_t bits_per_input, size_t num_classes, size_t filter_inputs, size_t filter_entries, size_t filter_hashes);

void set_hashes(uint64_t* hash_values, size_t size);

void set_ordering(uint64_t* ordering, size_t size);

void fill_model(uint64_t* data, size_t size);

uint64_t predict(unsigned char* input);

void test_write_read();
