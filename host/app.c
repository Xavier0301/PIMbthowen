/**
* app.c
* Host Application Source File
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>

#include "../support/common.h"
#include "../support/timer.h"
#include "../support/params.h"

#include "../cbthowen/model.h"
#include "../cbthowen/data_manager.h"
#include "../cbthowen/data_loader.h"
#include "../cbthowen/batch.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

// Define the WNN Model path as MODEL_PATH here
#ifndef MODEL_PATH
#define MODEL_PATH "./model.dat"
#endif

#ifndef NR_DPUS
#define NR_DPUS 1
#endif

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif

#define NUM_SAMPLES(num_dpus, num_samples, dpu_idx) (num_samples / num_dpus + (num_samples % num_dpus > dpu_idx ? 1 : 0))

// Pointer declarations
static tensor3d_t hashes; // (#SAMPLES, #FILTERS, #FILTER_HASHES)
static uint64_t* predictions; // (#SAMPLES)
static uint64_t* predictions_host; // (#SAMPLES)
static model_t model; // WNN model

void log_input_args(dpu_params_t input_arguments, size_t it) {
    printf("(%zu: %d) ", it, input_arguments.nr_inputs);
}

void transfer_data_to_dpus(struct dpu_set_t dpu_set, 
    unsigned int nr_dpus, 
    dpu_params_t* input_params, 
    unsigned int dpu_model_transfer_size_bytes,
    unsigned int dpu_input_transfer_size_bytes) {

    unsigned int each_dpu = 0;
    struct dpu_set_t dpu;

    printf("Broadcast model \n");

    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 0, model.data.data, dpu_model_transfer_size_bytes, DPU_XFER_DEFAULT));

    printf("Parallel hashes push \n");

    unsigned int sample_it = 0;
    DPU_FOREACH(dpu_set, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, TENSOR3D_AXIS1(hashes, sample_it)));
        sample_it += input_params[each_dpu].nr_inputs;
    }
    DPU_ASSERT(
        dpu_push_xfer(dpu_set, 
            DPU_XFER_TO_DPU, 
            DPU_MRAM_HEAP_POINTER_NAME, 
            dpu_model_transfer_size_bytes,
            dpu_input_transfer_size_bytes, 
            DPU_XFER_DEFAULT)
    );
}

void retrieve_data_from_dpus(struct dpu_set_t dpu_set, 
    unsigned int nr_dpus, 
    dpu_params_t* input_params, 
    unsigned int dpu_model_transfer_size_bytes,
    unsigned int dpu_input_transfer_size_bytes,
    unsigned int dpu_output_transfer_size_bytes) {

    unsigned int each_dpu = 0;
    struct dpu_set_t dpu;

    printf("Prediction pull \n");

    unsigned int pred_it = 0;
    DPU_FOREACH(dpu_set, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &predictions[pred_it]));
        pred_it += input_params[each_dpu].nr_inputs;
    }

    DPU_ASSERT(
        dpu_push_xfer(dpu_set, 
            DPU_XFER_FROM_DPU, 
            DPU_MRAM_HEAP_POINTER_NAME, 
            dpu_model_transfer_size_bytes + dpu_input_transfer_size_bytes, 
            dpu_output_transfer_size_bytes, 
            DPU_XFER_DEFAULT)
    );
}

// Main of the Host Application
int main(int argc, char **argv) {

    // Input parameters
    struct Params p = input_params(argc, argv);

    // Timer declaration
    Timer timer;
#if defined(CYCLES) || defined(INSTRUCTIONS)
    double cc = 0;
    double cc_min = 0;
#endif

printf("MACROS definitions:\n");
#if PRINT == 1
    printf("PRINT ");
#endif
#if defined (CYCLES)
    printf("CYCLES ");
#endif
#if defined (INSTRUCTIONS)
    printf("INSTRUCTIONS ");
#endif
printf("\n");
	
    // Allocate DPUs
    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus)); // Number of DPUs in the DPU set
    printf("Allocated %d DPU(s)\t", nr_of_dpus);
    printf("NR_TASKLETS\t%d\n", NR_TASKLETS);

    // Load binary
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));

    // Load model
    printf("Loading model\n");
         
    read_model(MODEL_PATH, &model);

    printf("Model has bleach %d\n", model.bleach);

    // Loading binarized dataset
    printf("Loading dataset\n");
    const unsigned int num_samples = p.num_samples;
    bmatrix_t binarized_infimnist;
    bmatrix_init(&binarized_infimnist, num_samples, MNIST_IM_SIZE * model.bits_per_input);
    size_t num_samples_total, sample_size;
    read_dataset_partial("../data/binarized8m.dat", &binarized_infimnist, num_samples, &num_samples_total, &sample_size);

#if PRINT
    print_binarized_image_raw(&binarized_infimnist, infimnist_labels, 0, 2);
#endif

    printf("Reordering dataset\n");
    bmatrix_t reordered_binarized_infinimnist;
    bmatrix_init(&reordered_binarized_infinimnist, num_samples, MNIST_IM_SIZE * model.bits_per_input);
    reorder_dataset(&reordered_binarized_infinimnist, &binarized_infimnist, model.input_order, num_samples, MNIST_IM_SIZE * model.bits_per_input);

    // Calculate model size (transfer size is identical to model size)
    const unsigned int model_entries = model.filter_entries * model.num_filters * model.num_classes;
    const unsigned int model_entry_bytes = sizeof(entry_t);
    const unsigned int model_entries_aligned = aligned_count(model_entries, model_entry_bytes);

    // Input size calculations
    const unsigned int dpu_num_samples_max = divceil(num_samples, nr_of_dpus);

    const unsigned int hashes_per_sample = model.num_filters * model.filter_hashes;
    const unsigned int dpu_num_hashes_max = hashes_per_sample * dpu_num_samples_max;
    const unsigned bytes_per_hash = sizeof(entry_t);
    const unsigned int bytes_per_sample = hashes_per_sample * bytes_per_hash;
    const unsigned int dpu_num_hashes_max_aligned = aligned_count(dpu_num_hashes_max, bytes_per_hash);

    // Output size calculations
    const unsigned int dpu_num_preds_max = dpu_num_samples_max;
    const unsigned int bytes_per_prediction = sizeof(uint64_t);
    const unsigned int dpu_num_preds_max_aligned = aligned_count(dpu_num_preds_max, bytes_per_prediction);

    // Input/output allocation in host main memory
    printf("Input/output allocation in host main memory\n");
    tensor_init(&hashes, num_samples, model.num_filters, model.filter_hashes);
    predictions = (uint64_t *) calloc(num_samples, sizeof(*predictions));
    predictions_host = (uint64_t *) calloc(num_samples, sizeof(*predictions_host));

    unsigned int i = 0;

    // Transfer sizes
    const unsigned int model_bytes = model_entries_aligned * model_entry_bytes;
    const unsigned int dpu_input_transfer_size_bytes =  dpu_num_hashes_max_aligned * bytes_per_hash;
    const unsigned int dpu_output_transfer_size_bytes = dpu_num_preds_max_aligned * bytes_per_prediction;

    unsigned int each_dpu = 0;

    printf("Batch hashing\n");
    batch_hashing(&hashes, &model, &reordered_binarized_infinimnist, num_samples);

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        if(rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup);
        reorder_dataset(&reordered_binarized_infinimnist, &binarized_infimnist, model.input_order, num_samples, MNIST_IM_SIZE * model.bits_per_input);
        if(rep >= p.n_warmup)
            stop(&timer, 0);

        if(rep >= p.n_warmup)
            start(&timer, 1, rep - p.n_warmup);
        batch_hashing(&hashes, &model, &reordered_binarized_infinimnist, num_samples);
        if(rep >= p.n_warmup)
            stop(&timer, 1);
#if defined(CHECK_RES)
        batch_prediction(predictions_host, &model, &binarized_infimnist, num_samples);
#endif

        printf("Load DPU arguments\n");
        // Input arguments
        unsigned int kernel = 0;
        dpu_model_params_t model_params = (dpu_model_params_t) {
            .num_classes = model.num_classes,
            .num_filters = model.num_filters,
            .filter_inputs = model.filter_inputs,
            .filter_entries = model.filter_entries,
            .filter_hashes = model.filter_hashes,
            .bleach = model.bleach
        };
        dpu_params_t input_arguments[NR_DPUS];
        for(i = 0; i < nr_of_dpus; i++) {
            const unsigned int dpu_num_samples = NUM_SAMPLES(nr_of_dpus, num_samples, i);
            input_arguments[i] = (dpu_params_t) {
                .model_size_bytes = model_bytes,

                .input_size_bytes = dpu_num_samples * bytes_per_sample,
                .input_transfer_size_bytes = dpu_input_transfer_size_bytes,

                .output_size_bytes = dpu_num_samples * bytes_per_prediction,
                .output_transfer_size_bytes = dpu_output_transfer_size_bytes,

                .nr_inputs = dpu_num_samples,

                .kernel = kernel,
                .model_params = model_params
            };
            // log_input_args(input_arguments[i], i);
        }
        // printf("\n");


        if(rep >= p.n_warmup)
            start(&timer, 2, rep - p.n_warmup); // Start timer (CPU-DPU transfers)
        i = 0;
		// Copy input arguments
        // Parallel transfers
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[0]), DPU_XFER_DEFAULT));

        transfer_data_to_dpus(dpu_set, nr_of_dpus, input_arguments, model_bytes, dpu_input_transfer_size_bytes);

        if(rep >= p.n_warmup)
            stop(&timer, 2); // Stop timer (CPU-DPU transfers)
		
        printf("Run program on DPU(s) \n");
        // Run DPU kernel
        if(rep >= p.n_warmup) {
            start(&timer, 3, rep - p.n_warmup); // Start timer (DPU kernel)
        }
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        if(rep >= p.n_warmup) {
            stop(&timer, 3); // Stop timer (DPU kernel)
        }

#if PRINT
        {
            unsigned int each_dpu = 0;
            printf("Display DPU Logs\n");
            DPU_FOREACH (dpu_set, dpu) {
                printf("DPU#%d:\n", each_dpu);
                DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
                each_dpu++;
            }
        }
#endif

        printf("Retrieve results\n");
        if(rep >= p.n_warmup)
            start(&timer, 4, rep - p.n_warmup); // Start timer (DPU-CPU transfers)
        i = 0;

        retrieve_data_from_dpus(dpu_set, nr_of_dpus, input_arguments, model_bytes, dpu_input_transfer_size_bytes, dpu_output_transfer_size_bytes);

        if(rep >= p.n_warmup)
            stop(&timer, 4); // Stop timer (DPU-CPU transfers)

#if defined(CYCLES) || defined(INSTRUCTIONS)
        dpu_results_t results[nr_of_dpus];
        // Parallel transfers
        dpu_results_t* results_retrieve[nr_of_dpus];
        DPU_FOREACH(dpu_set, dpu, i) {
            results_retrieve[i] = (dpu_results_t*)malloc(NR_TASKLETS * sizeof(dpu_results_t));
            DPU_ASSERT(dpu_prepare_xfer(dpu, results_retrieve[i]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "DPU_RESULTS", 0, NR_TASKLETS * sizeof(dpu_results_t), DPU_XFER_DEFAULT));
        DPU_FOREACH(dpu_set, dpu, i) {
            results[i].count = 0;
            // Retrieve tasklet count
            for (unsigned int each_tasklet = 0; each_tasklet < NR_TASKLETS; each_tasklet++) {
                // printf("instr. dpu %d. tasklet %d. %d\n", i, each_tasklet, results_retrieve[i][each_tasklet].count);
                if (results_retrieve[i][each_tasklet].count > results[i].count)
                    results[i].count = results_retrieve[i][each_tasklet].count;
            }
            free(results_retrieve[i]);
        }

        uint64_t max_count = 0;
        uint64_t min_count = 0xFFFFFFFFFFFFFFFF;
        // Print performance results
        if(rep >= p.n_warmup){
            i = 0;
            DPU_FOREACH(dpu_set, dpu) {
                if(results[i].count > max_count)
                    max_count = results[i].count;
                if(results[i].count < min_count)
                    min_count = results[i].count;
                i++;
            }
            cc += (double)max_count;
            cc_min += (double)min_count;
        }
        // Per tasklet
        cc /= (double) NR_TASKLETS;
        cc_min /= (double) NR_TASKLETS;
#endif

        if(rep >= p.n_warmup)
            start(&timer, 5, rep - p.n_warmup);
        batch_prediction(predictions_host, &model, &binarized_infimnist, num_samples);
        if(rep >= p.n_warmup)
            stop(&timer, 5);
    }
#ifdef CYCLES
    printf("results_and_timings(cycles), %d, %d, %d, %g", nr_of_dpus, NR_TASKLETS, num_samples, cc / p.n_reps);
#elif INSTRUCTIONS
    printf("results_and_timings(instructions), %d, %d, %d, %.0f", nr_of_dpus, NR_TASKLETS, num_samples, cc / p.n_reps);
    // printf("DPU instructions (min) = %f\n", cc_min / p.n_reps);
#endif
	
    // Print timing results
    // printf("CPU ");
    // print(&timer, 0, p.n_reps);
    // printf("CPU-DPU ");
    // print(&timer, 1, p.n_reps);
    // printf("DPU Kernel ");
    // print(&timer, 2, p.n_reps);
    // printf("DPU-CPU ");
    // print(&timer, 3, p.n_reps);

    printf(", ");
    print2(&timer, 0, p.n_reps);
    printf(", ");
    print2(&timer, 1, p.n_reps);
    printf(", ");
    print2(&timer, 2, p.n_reps);
    printf(", ");
    print2(&timer, 3, p.n_reps);
    printf(", ");
    print2(&timer, 4, p.n_reps);
    printf(", ");
    print2(&timer, 5, p.n_reps);
    printf(", ");
    print2(&timer, 6, p.n_reps);

    puts("");
#if defined(CHECK_RES)
    // Check output
    bool status = true;
    for (i = 0; i < num_samples; i++) {
        if(predictions_host[i] != predictions[i]) {
            status = false;
            printf("Sample %d> %u -- %u_ \n", i, predictions[i], predictions_host[i]);
        }
        // printf("Sample %d> %u -- %u_ \n", i, predictions[i], predictions_host[i]);
    }
    if (status) {
        printf("\n[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("\n[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }
#endif

    // Deallocation
    // free(X);
    // free(Y);
    // free(Y_host);
    DPU_ASSERT(dpu_free(dpu_set)); // Deallocate DPUs
	
    return 0;
}
