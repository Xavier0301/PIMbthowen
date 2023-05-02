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

#define MODEL_ENTRY_SIZE_B (sizeof(uint32_t))
#define MODEL_FILTER_SIZE_B(p) ((p).filter_entries * MODEL_ENTRY_SIZE_B)
#define MODEL_DISCR_SIZE_B(p) ((p).num_filters * MODEL_FILTER_SIZE_B(p))

#define MODEL_DISCR_ADDR(p, base, discr) ((base) + (discr) * MODEL_DISCR_SIZE_B(p))
#define MODEL_FILTER_ADDR(p, base, discr, filter) (MODEL_DISCR_ADDR(p, base, discr) + (filter) * MODEL_FILTER_SIZE_B(p))
#define MODEL_ENTRY_ADDR(p, base, discr, filter, entry) (MODEL_FILTER_ADDR(p, base, discr, filter) + (entry) * MODEL_ENTRY_SIZE_B)

// All hashes from a single sample are stored in WRAM
#define HASHES_BLOCK_SIZE(p) ((p).filter_hashes * (p).num_filters)
#define HASHES_BLOCK_SIZE_B(p) (HASHES_BLOCK_SIZE(p) * sizeof(uint32_t))
#define HASHES_SAMPLE_ADDR(p, base, sample) ((base) + (sample) * HASHES_BLOCK_SIZE_B(p))

#define HASHES_FILTER_PTR(p, base_ptr, filter) ((base_ptr) + (filter) * (p).filter_hashes)
#define HASHES_ENTRY_PTR(p, base_ptr, filter, hash_idx) (HASHES_FILTER_PTR(p, base_ptr, filter) + (hash_idx))

#define PREDICTION_ADDR(p, base, sample) ((base) + (sample) * sizeof(uint64_t))

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
    dpu_results_t *result = &DPU_RESULTS[tasklet_id];
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
    uint32_t mram_base_addr_predictions = (uint32_t) (mram_base_addr_inputs + input_size_dpu_bytes);

    // Each tasklet only needs to store one filter element in mram
    uint32_t* filter_buffer = (uint32_t*) mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(sizeof(*filter_buffer)));
    // Each tasklet needs to store the hashes for a single sample for now
    uint32_t* hashes_buffer = (uint32_t*) mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(HASHES_BLOCK_SIZE_B(model_params)));

    uint32_t* popcounts = (uint32_t*) mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(sizeof(uint32_t) * model_params.num_classes));

#if PRINT
    printf("%u. Starting work\n", tasklet_id);
#endif

    for(unsigned int sample_it = tasklet_id; sample_it < nr_inputs; sample_it += NR_TASKLETS) {

        mram_read(HASHES_SAMPLE_ADDR(model_params, mram_base_addr_inputs, sample_it), hashes_buffer, ROUND_UP_TO_MULTIPLE_OF_8(HASHES_BLOCK_SIZE_B(model_params)));

        for(unsigned int discriminator_it = 0; discriminator_it < model_params.num_classes; ++discriminator_it) 
            popcounts[discriminator_it] = 0;

        for(unsigned int filter_it = 0; filter_it < model_params.num_filters; ++filter_it) {
            uint32_t* hashes_filter_buffer = HASHES_FILTER_PTR(model_params, hashes_buffer, filter_it);
            for(unsigned int discriminator_it = 0; discriminator_it < model_params.num_classes; ++discriminator_it) {
                // for(unsigned int block_it = 0; block_it < MODEL_BLOCKS_PER_FILTER; ++block_it) {
                //     mram_read(MODEL_BLOCK_ADDR(model_params, mram_base_addr_model, discriminator_it, filter_it, block_it), filter_buffer + MODEL_BLOCK_SIZE(model_params) * block_it, MODEL_BLOCK_SIZE_B(model_params));
                // }

                // (filter_reduction(filter_buffer, filter_hashes, model_params.filter_hashes)

                uint32_t min = -1;
                for(size_t hash_it = 0; hash_it < model_params.filter_hashes; ++hash_it) {
                    uint32_t hash = hashes_filter_buffer[hash_it];

                    uint32_t model_entry_addr = MODEL_ENTRY_ADDR(model_params, mram_base_addr_model, discriminator_it, filter_it, hash);
                    uint32_t aligned_addr = ROUND_DOWN_TO_MULTIPLE_OF_8(model_entry_addr);
                    uint32_t offset = (model_entry_addr - aligned_addr) / sizeof(*filter_buffer);
                    
                    mram_read(aligned_addr, filter_buffer, ROUND_UP_TO_MULTIPLE_OF_8(sizeof(*filter_buffer)));
    // #if PRINT
    //                 printf("%u. Hash %u: %u\n", tasklet_id, hash_it, hash);
    //                 printf("%u. Model entry address: %u (%u)\n", tasklet_id, aligned_addr, offset);
    //                 printf("%u. Model entry: %u (%u)\n", tasklet_id, filter_buffer[offset], filter_buffer[1-offset]);
    // #endif
                    uint32_t entry = filter_buffer[offset];
                    if(entry <= min) min = entry;
                }

                popcounts[discriminator_it] += (min >= model_params.bleach);
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
        mram_write(&argmax_pcount, PREDICTION_ADDR(model_params, mram_base_addr_predictions, sample_it), sizeof(argmax_pcount));
    }

    // DPU_PREDICTION.prediction = argmax_pcount;

#if PRINT
    printf("%u. Work done\n", tasklet_id);
#endif

#if defined(CYCLES) || defined(INSTRUCTIONS)
    result->count += counter_stop(&count); // STOP TIMER
#endif
	
    return 0;
}













#define OLD_MODEL_BLOCKS_PER_FILTER (2)
#define OLD_MODEL_BLOCK_SIZE(p) (ROUND_UP_TO_MULTIPLE_OF_8((p).filter_entries / OLD_MODEL_BLOCKS_PER_FILTER))
#define OLD_MODEL_BLOCK_SIZE_B(p) (OLD_MODEL_BLOCK_SIZE(p) * sizeof(uint32_t))
#define OLD_MODEL_ENTRIES_PER_BLOCK(p) ((p).filter_entries / OLD_MODEL_BLOCKS_PER_FILTER)

#define OLD_MODEL_ENTRY_IDX(p, block_it, entry_it) ((block_it) * OLD_MODEL_ENTRIES_PER_BLOCK(p) + (entry_it))

#define OLD_MODEL_DISCR_ADDR(p, base, discr) ((base) + (discr) * (p).num_filters * (p).filter_entries * sizeof(uint32_t))
#define OLD_MODEL_FILTER_ADDR(p, base, discr, filter) (OLD_MODEL_DISCR_ADDR(p, base, discr) + (filter) * (p).filter_entries * sizeof(uint32_t))
#define OLD_MODEL_BLOCK_ADDR(p, base, discr, filter, block) (OLD_MODEL_FILTER_ADDR(p, base, discr, filter) + (block) * OLD_MODEL_BLOCK_SIZE_B(p))

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
    dpu_results_t *result = &DPU_RESULTS[tasklet_id];
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
    uint32_t* filter_buffer = (uint32_t*) mem_alloc(OLD_MODEL_BLOCK_SIZE_B(model_params) * OLD_MODEL_BLOCKS_PER_FILTER);

    printf("** MODEL DATA **\n");
    
    for(unsigned int discriminator_it = 0; discriminator_it < 1; ++discriminator_it) {
        printf("Discriminator %zu.\n\n", discriminator_it);
        for(unsigned int filter_it = 0; filter_it < 1; ++filter_it) {
            for(unsigned int block_it = 0; block_it < OLD_MODEL_BLOCKS_PER_FILTER; ++block_it) {
                printf("Reading at addr %u, block size %u\n", OLD_MODEL_BLOCK_ADDR(model_params, mram_base_addr_model, discriminator_it, filter_it, block_it), MODEL_BLOCK_SIZE_B(model_params));
                mram_read(OLD_MODEL_BLOCK_ADDR(model_params, mram_base_addr_model, discriminator_it, filter_it, block_it), filter_buffer + MODEL_BLOCK_SIZE(model_params) * block_it, MODEL_BLOCK_SIZE_B(model_params));
            }
            printf("Filter %zu.\n", filter_it);
            for(unsigned int entry_it = 0; entry_it < model_params.filter_entries; ++entry_it) {
                printf("%u ", filter_buffer[entry_it]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

    // uint32_t* hashes_buffer = (uint32_t*) mem_alloc(HASHES_BLOCK_SIZE_B(model_params));
    // printf("** INPUT DATA **\n");
    // for(unsigned int filter_it = 0; filter_it < model_params.num_filters; ++filter_it) {
    //     printf("Filter %u.\n", filter_it);
    //     mram_read(HASHES_FILTER_ADDR(model_params, mram_base_addr_inputs, filter_it), hashes_buffer, HASHES_BLOCK_SIZE_B(model_params));
    //     for(unsigned int hash_it = 0; hash_it < model_params.filter_hashes; ++hash_it) {
    //         printf("%u ", hashes_buffer[hash_it]);
    //     }
    //     printf("\n\n");
    // }

#if defined(CYCLES) || defined(INSTRUCTIONS)
    result->count += counter_stop(&count); // STOP TIMER
#endif
	
    return 0;
}
