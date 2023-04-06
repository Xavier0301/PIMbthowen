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

// Input and output arguments
__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
__host dpu_results_t DPU_RESULTS[NR_TASKLETS];

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

extern int main_kernel1(void);
int (*kernels[nr_kernels])(void) = {main_kernel1};
int main(void) { 
    // Kernel
    return kernels[DPU_INPUT_ARGUMENTS.kernel](); 
}

// AXPY: Computes AXPY for a cached block 
static void axpy(T *bufferY, T *bufferX, T alpha, unsigned int l_size) {

    //@@ INSERT AXPY CODE
    for(size_t it = 0; it < l_size; ++it) {
        bufferY[it] = bufferY[it] + alpha * bufferX[it];
    }

}

// main_kernel1
int main_kernel1() {
    unsigned int tasklet_id = me();
#if PRINT
    printf("tasklet_id = %u\n", tasklet_id);
#endif
    if (tasklet_id == 0){ 
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

    uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.size; // Input size per DPU in bytes
    uint32_t input_size_dpu_bytes_transfer = DPU_INPUT_ARGUMENTS.transfer_size; // Transfer input size per DPU in bytes
    T alpha = DPU_INPUT_ARGUMENTS.alpha; // alpha (a in axpy)

    // Address of the current processing block in MRAM
    uint32_t base_tasklet = tasklet_id << BLOCK_SIZE_LOG2;
    uint32_t mram_base_addr_X = (uint32_t)DPU_MRAM_HEAP_POINTER;
    uint32_t mram_base_addr_Y = (uint32_t)(DPU_MRAM_HEAP_POINTER + input_size_dpu_bytes_transfer);

    // Initialize a local cache in WRAM to store the MRAM block
    //@@ INSERT WRAM ALLOCATION HERE

    // We simply use the incremental allocator here we don't need anything fancy
    T* blockX = (T*) mem_alloc(BLOCK_SIZE);
    T* blockY = (T*) mem_alloc(BLOCK_SIZE);

    // printf("%u. Starting work\n", tasklet_id);
    
    int i = 0;
    for(unsigned int byte_index = base_tasklet; byte_index < input_size_dpu_bytes; byte_index += BLOCK_SIZE * NR_TASKLETS){

        // printf("%u. byte_index: %d\n", tasklet_id, byte_index);
        // Bound checking
        // The last block might not be a full BLOCK_SIZE
        uint32_t l_size_bytes = (byte_index + BLOCK_SIZE >= input_size_dpu_bytes) ? (input_size_dpu_bytes - byte_index) : BLOCK_SIZE;

        // Load cache with current MRAM block
        mram_read(mram_base_addr_X + byte_index, blockX, l_size_bytes);
        mram_read(mram_base_addr_Y + byte_index, blockY, l_size_bytes);

        // Compute AXPY block-wise
        axpy(blockY, blockX, alpha, l_size_bytes >> DIV);

        // Write cache to current MRAM block
        mram_write(blockY, mram_base_addr_Y + byte_index, l_size_bytes);

        ++i;
    }
    // printf("loop_count %d. tasklet_id %d.\n", i, tasklet_id);

#if defined(CYCLES) || defined(INSTRUCTIONS)
    result->count += counter_stop(&count); // STOP TIMER
#endif
	
    return 0;
}
