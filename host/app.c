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
#include "../cbthowen/model_manager.h"
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
    printf("(%zu. %d, %d, %d)\n", it, input_arguments.input_size_bytes, input_arguments.model_size_bytes, input_arguments.nr_inputs);
}

void transfer_data_to_dpus(struct dpu_set_t dpu_set, 
    unsigned int nr_dpus, 
    dpu_params_t* base_input_param, 
    dpu_params_t* last_input_param) {

    unsigned int each_dpu = 0;
    struct dpu_set_t dpu;

    struct dpu_set_t last_dpu;

    printf("Broadcast model \n");

    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 0, model.data.data, base_input_param->model_size_bytes, DPU_XFER_DEFAULT));

    printf("Parallel hashes push \n");

    DPU_FOREACH(dpu_set, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, TENSOR3D_AXIS1(hashes, each_dpu * base_input_param->nr_inputs)));
        if(each_dpu == nr_dpus - 1) last_dpu = dpu;
    }
    DPU_ASSERT(
        dpu_push_xfer(dpu_set, 
            DPU_XFER_TO_DPU, 
            DPU_MRAM_HEAP_POINTER_NAME, 
            base_input_param->model_size_bytes, 
            base_input_param->input_size_bytes, 
            DPU_XFER_DEFAULT)
    );

    // There are inputs yet to be sent in case of uneven repartition
    if(last_input_param->nr_inputs - base_input_param->nr_inputs > 0) {
        unsigned int num_sent_samples_total = base_input_param->nr_inputs * nr_dpus;
        unsigned int num_sent_samples_single = base_input_param->nr_inputs;
        unsigned int num_leftover_samples = last_input_param->nr_inputs - base_input_param->nr_inputs;

        DPU_ASSERT(
            dpu_copy_to(last_dpu, 
                DPU_MRAM_HEAP_POINTER_NAME, 
                base_input_param->model_size_bytes + base_input_param->input_size_bytes, 
                TENSOR3D_AXIS1(hashes, num_sent_samples_total),
                last_input_param->input_size_bytes - base_input_param->input_size_bytes)
        );
    }
}

void retrieve_data_from_dpus(struct dpu_set_t dpu_set, 
    unsigned int nr_dpus, 
    dpu_params_t* base_input_param, 
    dpu_params_t* last_input_param, 
    unsigned int bytes_per_prediction) {

    unsigned int each_dpu = 0;
    struct dpu_set_t dpu;

    struct dpu_set_t last_dpu;

    printf("Prediction pull \n");

    DPU_FOREACH(dpu_set, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, predictions + each_dpu * base_input_param->nr_inputs));
        if(each_dpu == nr_dpus - 1) last_dpu = dpu;
    }

    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, base_input_param->model_size_bytes + base_input_param->input_size_bytes, base_input_param->nr_inputs * bytes_per_prediction, DPU_XFER_DEFAULT));

    // There are predictions yet to be retrieved in case of uneven repartition
    if(last_input_param->nr_inputs - base_input_param->nr_inputs > 0) {
        unsigned int num_retrieved_pred_total = base_input_param->nr_inputs * nr_dpus;
        unsigned int num_retrieved_pred_single = base_input_param->nr_inputs;
        unsigned int num_leftover_pred = last_input_param->nr_inputs - base_input_param->nr_inputs;
        DPU_ASSERT(
            dpu_copy_from(last_dpu, 
                DPU_MRAM_HEAP_POINTER_NAME, 
                last_input_param->model_size_bytes + last_input_param->input_size_bytes + num_retrieved_pred_single * bytes_per_prediction, 
                &predictions[num_retrieved_pred_total],
                num_leftover_pred * bytes_per_prediction)
        );
    }
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

    // Loading+Binarizing dataset
    printf("Loading dataset\n");
    load_mnist();

    printf("Binarizing dataset with %zu bits per input\n", model.bits_per_input);
    binarize_mnist(model.bits_per_input);

    print_binarized_mnist_image(0, 2);

    printf("Reordering dataset\n");
    reorder_binarized_mnist(model.input_order, model.bits_per_input);

    // Calculate model size
    const unsigned int model_entries = model.filter_entries * model.num_filters * model.num_classes;
    const unsigned int model_entry_bytes = sizeof(entry_t);
    const unsigned int model_entries_aligned = aligned_count(model_entries, model_entry_bytes);

    // Calculate input size
    const unsigned int num_samples = p.num_samples;
    const unsigned int dpu_num_samples = num_samples / nr_of_dpus;

    const unsigned int hashes_per_image = model.num_filters * model.filter_hashes;
    const unsigned int dpu_num_hashes = hashes_per_image * dpu_num_samples;
    const unsigned hash_entry_bytes = sizeof(entry_t);
    const unsigned int dpu_num_hashes_aligned = aligned_count(dpu_num_hashes, hash_entry_bytes);

    // Input/output allocation in host main memory
    printf("Input/output allocation in host main memory\n");
    tensor_init(&hashes, num_samples, model.num_filters, model.filter_hashes);
    predictions = (uint64_t *) calloc(num_samples, sizeof(*predictions));
    predictions_host = (uint64_t *) calloc(num_samples, sizeof(*predictions_host));

    unsigned int i = 0;

    const unsigned int model_bytes = model_entries_aligned * model_entry_bytes;

    const unsigned int hashes_bytes_per_dpu =  dpu_num_hashes_aligned * hash_entry_bytes;
    const unsigned int hashes_bytes_total = num_samples * hashes_per_image * hash_entry_bytes;

    const unsigned int predictions_per_dpu = dpu_num_samples;
    const unsigned int prediction_entry_bytes = sizeof(uint64_t);
    const unsigned int prediction_bytes_per_dpu = predictions_per_dpu * prediction_entry_bytes;
    const unsigned int prediction_bytes_total = num_samples * prediction_entry_bytes;

    unsigned int each_dpu = 0;

    printf("Batch hashing\n");
    batch_hashing(&hashes, &model, &reordered_binarized_test, num_samples);

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Compute output on CPU (verification purposes)
        if(rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup);

        printf("Batch prediction\n");        
        batch_prediction(predictions_host, &model, &binarized_test, num_samples);
        if(rep >= p.n_warmup)
            stop(&timer, 0);

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
                .input_transfer_size_bytes = dpu_input_transfer_size,

                .output_size_bytes = dpu_num_samples * bytes_per_prediction,
                .output_transfer_size_bytes = dpu_output_transfer_size,

                .nr_inputs = dpu_num_samples,

                .kernel = kernel,
                .model_params = model_params
            };
            log_input_args(input_arguments[i], i);
        }


        if(rep >= p.n_warmup)
            start(&timer, 1, rep - p.n_warmup); // Start timer (CPU-DPU transfers)
        i = 0;
		// Copy input arguments
        // Parallel transfers
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[0]), DPU_XFER_DEFAULT));

        transfer_data_to_dpus(dpu_set, nr_of_dpus, &base_input_argument, &last_input_argument);

        if(rep >= p.n_warmup)
            stop(&timer, 1); // Stop timer (CPU-DPU transfers)
		
        printf("Run program on DPU(s) \n");
        // Run DPU kernel
        if(rep >= p.n_warmup) {
            start(&timer, 2, rep - p.n_warmup); // Start timer (DPU kernel)
        }
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        if(rep >= p.n_warmup) {
            stop(&timer, 2); // Stop timer (DPU kernel)
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
            start(&timer, 3, rep - p.n_warmup); // Start timer (DPU-CPU transfers)
        i = 0;

        retrieve_data_from_dpus(dpu_set, nr_of_dpus, &base_input_argument, &last_input_argument, prediction_entry_bytes);

        if(rep >= p.n_warmup)
            stop(&timer, 3); // Stop timer (DPU-CPU transfers)

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
    }
#ifdef CYCLES
    printf("DPU cycles  = %g\n", cc / p.n_reps);
#elif INSTRUCTIONS
    printf("DPU instructions, %d, %d, %.0f\n", NR_TASKLETS, num_samples, cc / p.n_reps);
    // printf("DPU instructions (min) = %f\n", cc_min / p.n_reps);
#endif
	
    // Print timing results
    printf("CPU ");
    print(&timer, 0, p.n_reps);
    printf("CPU-DPU ");
    print(&timer, 1, p.n_reps);
    printf("DPU Kernel ");
    print(&timer, 2, p.n_reps);
    printf("DPU-CPU ");
    print(&timer, 3, p.n_reps);

    // Check output
    puts("");
    bool status = true;
    for (i = 0; i < num_samples; i++) {
        if(predictions_host[i] != predictions[i]) {
            status = false;
            // printf("Sample %d> %u -- %u_ \n", i, predictions[i], predictions_host[i]);
        }
        printf("Sample %d> %u -- %u_ \n", i, predictions[i], predictions_host[i]);
    }
    if (status) {
        printf("\n[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("\n[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    // Deallocation
    // free(X);
    // free(Y);
    // free(Y_host);
    DPU_ASSERT(dpu_free(dpu_set)); // Deallocate DPUs
	
    return status ? 0 : -1;
}
