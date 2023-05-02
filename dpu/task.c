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

#define MODEL_BLOCKS_PER_FILTER (2)
#define MODEL_BLOCK_SIZE(p) (ROUND_UP_TO_MULTIPLE_OF_8((p).filter_entries / MODEL_BLOCKS_PER_FILTER))
#define MODEL_BLOCK_SIZE_B(p) (MODEL_BLOCK_SIZE(p) * sizeof(uint32_t))
#define MODEL_ENTRIES_PER_BLOCK(p) ((p).filter_entries / MODEL_BLOCKS_PER_FILTER)

#define MODEL_ENTRY_IDX(p, block_it, entry_it) ((block_it) * MODEL_ENTRIES_PER_BLOCK(p) + (entry_it))

#define MODEL_DISCR_ADDR(p, base, discr) ((base) + (discr) * (p).num_filters * (p).filter_entries * sizeof(uint32_t))
#define MODEL_FILTER_ADDR(p, base, discr, filter) (MODEL_DISCR_ADDR(p, base, discr) + (filter) * (p).filter_entries * sizeof(uint32_t))
#define MODEL_BLOCK_ADDR(p, base, discr, filter, block) (MODEL_FILTER_ADDR(p, base, discr, filter) + (block) * MODEL_BLOCK_SIZE_B(p))

// All hashes from a single sample are stored in WRAM
#define HASHES_BLOCK_SIZE(p) (ROUND_UP_TO_MULTIPLE_OF_8((p).filter_hashes * (p).num_filters * sizeof(uint32_t)))
#define HASHES_FILTER_PTR(p, base_ptr, filter) ((base_ptr) + (filter) * (p).filter_hashes)
#define HASHES_FILTER_ADDR(p, base, filter) ((base) + (filter) * (p).filter_hashes * sizeof(uint32_t))

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

extern int main_kernel1(void);
extern int print_kernel(void);
int (*kernels[2])(void) = {main_kernel1, print_kernel};
int main(void) { 
    // Kernel
    return main_kernel1();
    // return kernels[DPU_INPUT_ARGUMENTS.kernel](); 
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
    uint32_t nr_inputs = DPU_INPUT_ARGUMENTS.nr_inputs; // Number of inputs per DPU

    dpu_model_params_t model_params = DPU_INPUT_ARGUMENTS.model_params;

#if PRINT
    printf("model size: %u bytes\n", model_size_dpu_bytes);
    printf("input size: %u bytes\n", input_size_dpu_bytes);
    printf("model params: %u %u %u %u %u %u\n", model_params.num_classes, model_params.num_filters, model_params.filter_inputs, model_params.filter_entries, model_params.filter_hashes, model_params.bleach);
#endif

    uint32_t mram_base_addr_model = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    uint32_t mram_base_addr_inputs = (uint32_t) (mram_base_addr_model + model_size_dpu_bytes);

    // Initialize a local cache in WRAM to store the MRAM block
    uint32_t* filter_buffer = (uint32_t*) mem_alloc(MODEL_BLOCK_SIZE_B(model_params) * MODEL_BLOCKS_PER_FILTER);
    uint32_t* hashes_buffer = (uint32_t*) mem_alloc(HASHES_BLOCK_SIZE(model_params));

    uint32_t* popcounts = (uint32_t*) mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(sizeof(uint32_t) * model_params.num_classes));

#if PRINT
    printf("%u. Starting work\n", tasklet_id);
#endif

    mram_read(mram_base_addr_inputs, hashes_buffer, HASHES_BLOCK_SIZE(model_params));

    for(unsigned int filter_it = 0; filter_it < model_params.num_filters; ++filter_it) {
        uint32_t* filter_hashes = HASHES_FILTER_PTR(model_params, hashes_buffer, filter_it);
        for(unsigned int discriminator_it = 0; discriminator_it < model_params.num_classes; ++discriminator_it) {
            for(unsigned int block_it = 0; block_it < MODEL_BLOCKS_PER_FILTER; ++block_it) {
                mram_read(MODEL_BLOCK_ADDR(model_params, mram_base_addr_model, discriminator_it, filter_it, block_it), filter_buffer + MODEL_BLOCK_SIZE(model_params) * block_it, MODEL_BLOCK_SIZE_B(model_params));
            }

            popcounts[discriminator_it] += (filter_reduction(filter_buffer, filter_hashes, model_params.filter_hashes) >= model_params.bleach);
        }
    }

    uint32_t max_pcount = 0;
    uint64_t argmax_pcount = 0;
    for(unsigned int discriminator_it = 0; discriminator_it < model_params.num_classes; ++discriminator_it) {
#if PRINT
        printf("%u. Popcount %u: %u\n", tasklet_id, discriminator_it, popcounts[discriminator_it]);
#endif
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















// Simply prints the content of the model data and hashes data
int print_kernel() {
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

#if PRINT
    printf("model size: %u bytes\n", model_size_dpu_bytes);
    printf("input size: %u bytes\n", input_size_dpu_bytes);
    printf("model params: %u %u %u %u %u %u\n", model_params.num_classes, model_params.num_filters, model_params.filter_inputs, model_params.filter_entries, model_params.filter_hashes, model_params.bleach);
#endif

    uint32_t mram_base_addr_model = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    uint32_t mram_base_addr_inputs = (uint32_t) (mram_base_addr_model + model_size_dpu_bytes);

    printf("Addresses. Model %u. Hashes %u.\n", mram_base_addr_model, mram_base_addr_inputs);

    // Initialize a local cache in WRAM to store the MRAM block
    uint32_t* filter_buffer = (uint32_t*) mem_alloc(MODEL_BLOCK_SIZE_B(model_params) * MODEL_BLOCKS_PER_FILTER);

    printf("** MODEL DATA **\n");
    
    for(unsigned int discriminator_it = 0; discriminator_it < 1; ++discriminator_it) {
        printf("Discriminator %zu.\n\n", discriminator_it);
        for(unsigned int filter_it = 0; filter_it < 1; ++filter_it) {
            for(unsigned int block_it = 0; block_it < MODEL_BLOCKS_PER_FILTER; ++block_it) {
                printf("Reading at addr %u, block size %u\n", MODEL_BLOCK_ADDR(model_params, mram_base_addr_model, discriminator_it, filter_it, block_it), MODEL_BLOCK_SIZE_B(model_params));
                mram_read(MODEL_BLOCK_ADDR(model_params, mram_base_addr_model, discriminator_it, filter_it, block_it), filter_buffer + MODEL_BLOCK_SIZE(model_params) * block_it, MODEL_BLOCK_SIZE_B(model_params));
            }
            printf("Filter %zu.\n", filter_it);
            for(unsigned int entry_it = 0; entry_it < model_params.filter_entries; ++entry_it) {
                printf("%u ", filter_buffer[entry_it]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

    // uint32_t* hashes_buffer = (uint32_t*) mem_alloc(HASHES_BLOCK_SIZE(model_params));
    // printf("** INPUT DATA **\n");
    // for(unsigned int filter_it = 0; filter_it < model_params.num_filters; ++filter_it) {
    //     printf("Filter %u.\n", filter_it);
    //     mram_read(HASHES_FILTER_ADDR(model_params, mram_base_addr_inputs, filter_it), hashes_buffer, HASHES_BLOCK_SIZE(model_params));
    //     for(unsigned int hash_it = 0; hash_it < model_params.filter_hashes; ++hash_it) {
    //         printf("%u ", hashes_buffer[hash_it]);
    //     }
    //     printf("\n\n");
    // }

    DPU_PREDICTION.prediction = 0;

#if defined(CYCLES) || defined(INSTRUCTIONS)
    result->count += counter_stop(&count); // STOP TIMER
#endif
	
    return 0;
}
