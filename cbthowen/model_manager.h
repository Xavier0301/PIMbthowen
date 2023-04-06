#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "model.h"

void read_model(const char* filename, model_t* model);
void read_matrix(FILE* f, matrix_t* matrix, size_t size);
void read_tensor(FILE* f, tensor3d_t* tensor, size_t size);

void write_model(const char* filename, model_t* model);
void write_matrix(FILE* f, matrix_t* matrix, size_t size);
void write_tensor(FILE* f, tensor3d_t* tensor, size_t size);
