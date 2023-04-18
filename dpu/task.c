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

// Input and output argumentsd
__host dpu_params_t DPU_INPUT_ARGUMENTS;
__host dpu_results_t DPU_RESULTS[NR_TASKLETS];
__host dpu_prediction_t DPU_PREDICTION;

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

extern int main_kernel1(void);
int (*kernels[nr_kernels])(void) = {main_kernel1};
int main(void) { 
    // Kernel
    return kernels[DPU_INPUT_ARGUMENTS.kernel](); 
}

static uint32_t filter_reduction(uint32_t* filter, uint32_t* hashes, uint32_t nr_hashes) {
    uint32_t min = -1;
    for(size_t it = 0; it < nr_hashes; ++it) {
        uint32_t entry = filter[hashes[it]];
        if(entry <= min) min = entry;
    }

    return min;
}

// main_kernel1
int main_kernel1() {
    unsigned int tasklet_id = me();
#if PRINT
    printf("tasklet_id = %u\n", tasklet_id);
#endif
    if (tasklet_id == 0) { 
        mem_reset(); // Reset the heap
#ifdef CYCLES
        perfcounter_config(COUNT_CYCLES, true); // Initialize once the cycle counter
#elif INSTRUCTIONS
        perfcounter_config(COUNT_INSTRUCTIONS, true); // Initialize once the instruction counter
#endif
    }

    // Barrier
    barrier_wait(&my_barrier);
#if defined(CYCLES) || defined(INSTRUCTIONS)
    perfcounter_count count;
    result->count = 0;
    counter_start(&count); // START TIMER
#endif

    // Load model + input
    uint32_t model_size_dpu_bytes = DPU_INPUT_ARGUMENTS.model_size_bytes; // Transfer input size per DPU in bytes
    uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.input_size_bytes; // Input size per DPU in bytes

    dpu_model_params_t model_params = DPU_INPUT_ARGUMENTS.model_params;

    uint32_t mram_base_addr_model = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    uint32_t mram_base_addr_inputs = (uint32_t) (mram_base_addr_model + model_size_dpu_bytes);

    // Initialize a local cache in WRAM to store the MRAM block
    uint32_t* filter_buffer = (uint32_t*) mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(sizeof(uint32_t) * model_params.filter_entries));
    uint32_t* hashes_buffer = (uint32_t*) mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(sizeof(uint32_t) * model_params.filter_inputs));

    uint32_t* popcounts = (uint32_t*) mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(sizeof(uint32_t) * model_params.num_classes));

#if PRINT
    printf("%u. Starting work\n", tasklet_id);
#endif

    for(unsigned int filter_it = 0; filter_it < model_params.num_filters; ++filter_it) {
        mram_read(mram_base_addr_inputs + filter_it * model_params.filter_hashes * sizeof(uint32_t), hashes_buffer, model_params.filter_hashes * sizeof(uint32_t));
        for(unsigned int discriminator_it = 0; discriminator_it < model_params.num_classes; ++discriminator_it) {
            mram_read(mram_base_addr_model + discriminator_it * model_params.filter_entries * sizeof(uint32_t), filter_buffer, model_params.filter_entries * sizeof(uint32_t));

            popcounts[discriminator_it] += filter_reduction(filter_buffer, hashes_buffer, model_params.filter_hashes);
        }
    }

    uint32_t max_pcount = 0;
    uint64_t argmax_pcount = 0;
    for(unsigned int discriminator_it = 0; discriminator_it < model_params.num_classes; ++discriminator_it) {
        if(popcounts[discriminator_it] >= max_pcount) {
            max_pcount = popcounts[discriminator_it];
            argmax_pcount = discriminator_it;
        }
    }

    DPU_PREDICTION.prediction = argmax_pcount;

#if PRINT
    printf("%u. Work done\n", tasklet_id);
#endif

#if defined(CYCLES) || defined(INSTRUCTIONS)
    result->count += counter_stop(&count); // STOP TIMER
#endif
	
    return 0;
}
