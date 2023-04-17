/*
* AXPY with multiple tasklets
*
*/
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>

#include "../support/common.h"
#include "../support/cyclecount.h"

#include "../cbthowen/model.h"

// Input and output argumentsd
__host dpu_results_t DPU_RESULTS[NR_TASKLETS];

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

extern int main_kernel1(void);
int (*kernels[nr_kernels])(void) = {main_kernel1};
int main(void) { 
    // Kernel
    return kernels[DPU_INPUT_ARGUMENTS.kernel](); 
}

static uint16_t filter_reduction(uint16_t* filter, uint16_t* hashes, unsigned char nr_hashes) {
    uint16_t min = 0xffff;
    for(size_t it = 0; it < nr_hashes; ++it) {
        uint16_t entry = filter[hashes[it]];
        if(entry <= min) min = entry;
    }

    return min;
}

// Returns the popcount of the outputs of a given number of filters
static uint16_t filter_reductions(uint16_t* filters, uint16_t* hashes, unsigned char nr_filters, uint16_t filter_entries, unsigned char nr_hashes) {
    uint16_t popcount;

    uint16_t* filter;
    for(size_t it = 0; it < nr_filters; ++it) {
        filter = filters + it * filter_entries;
        popcount += filter_reduction(filter, hashes, nr_hashes);
    }

    return popcount;
}

// main_kernel1
int main_kernel1() {
    unsigned int tasklet_id = me();
#if PRINT
    printf("tasklet_id = %u\n", tasklet_id);
#endif
    if (tasklet_id == 0) { 
        mem_reset(); // Reset the heap
    } else {
        return 0;
    }

    // Barrier
    // barrier_wait(&my_barrier);

    // Load parameters
    uint32_t params_m = (uint32_t) DPU_MRAM_HEAP_POINTER;
    dpu_params_t* params_w = (dpu_params_t*) mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(sizeof(dpu_params_t)));
    mram_read((__mram_ptr void const*) params_m, params_w, ROUND_UP_TO_MULTIPLE_OF_8(sizeof(dpu_params_t)));

    // Load model + input
    uint32_t model_size_dpu_bytes_transfer = DPU_INPUT_ARGUMENTS.model_size_bytes; // Transfer input size per DPU in bytes
    uint32_t input_size_dpu_bytes_transfer = DPU_INPUT_ARGUMENTS.input_size_bytes; // Input size per DPU in bytes

    uint32_t mram_base_addr_model = (uint32_t) (DPU_MRAM_HEAP_POINTER + sizeof(dpu_params_t));
    uint32_t mram_base_addr_inputs = (uint32_t) (DPU_MRAM_HEAP_POINTER + model_size_dpu_bytes + sizeof(dpu_params_t));
    uint32_t mram_base_addr_outputs = (uint32_t) (DPU_MRAM_HEAP_POINTER + model_size_dpu_bytes + input_size_dpu_bytes + sizeof(dpu_params_t));

    // Initialize a local cache in WRAM to store the MRAM block
    uint16_t* filter_buffer = (uint16_t*) mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(sizeof(uint16_t)) * params_w->model_params.filter_entries);
    uint16_t* hashes_buffer = (uint16_t*) mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(sizeof(uint16_t)) * params_w->model_params.filter_inputs);

    uint16_t* popcounts = (uint16_t*) mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(sizeof(uint16_t)) * params_w->model_params.num_classes);

#if PRINT
    printf("%u. Starting work\n", tasklet_id);
#endif

    for(unsigned int filter_it = 0; filter_it < params_w->model_params.num_filters; ++filter_it) {
        mram_read(mram_base_addr_inputs + filter_it * params_w->model_params.filter_hashes * sizeof(uint16_t), hashes_buffer, params_w->model_params.filter_hashes * sizeof(uint16_t));
        for(unsigned int discriminator_it = 0; discriminator_it < params_w->model_params.num_classes; ++discriminator_it) {
            mram_read(mram_base_addr_model + discriminator_it * params_w->model_params.filter_entries * sizeof(uint16_t), filter_buffer, params_w->model_params.filter_entries * sizeof(uint16_t));

            popcounts[discriminator_it] += filter_reduction(filter_buffer, hashes_buffer, params_w->model_params.filter_hashes);
        }
    }

    for(unsigned int discriminator_it = 0; discriminator_it < params_w->model_params.num_classes; ++discriminator_it) {
        mram_write(popcounts, mram_base_addr_outputs, params_w->model_params.num_classes * sizeof(uint16_t));
    }
	
    return 0;
}
