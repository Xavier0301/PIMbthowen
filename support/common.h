#ifndef _COMMON_H_
#define _COMMON_H_

#define ROUND_UP_TO_MULTIPLE_OF_8(x)    ((((x) + 7)/8)*8)

typedef struct {
    uint32_t num_classes;

    uint32_t num_filters;
    uint32_t filter_inputs;
    uint32_t filter_entries;
    uint32_t filter_hashes;
    
    uint32_t bleach;
} dpu_model_params_t;

typedef struct {
    uint32_t model_size_bytes;
    uint32_t input_size_bytes;

    enum kernels {
	    kernel1 = 0,
	    nr_kernels = 1,
	} kernel;

    dpu_model_params_t model_params;
} dpu_params_t;
 
typedef struct {
    uint64_t count; // Cycle count
} dpu_results_t;

typedef struct {
    uint64_t prediction;
} dpu_prediction_t;

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define divceil(n, m) (((n)-1) / (m) + 1)
#define roundup(n, m) ((n / m) * m + m)
#define aligned_count(count, size_bytes) (((count * size_bytes) % 8) != 0 ? roundup(count, 8) : count)
#endif
