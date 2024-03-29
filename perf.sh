#!/bin/bash

declare -a DPUS=(1 2 4 8 16 32 64)
declare -a TASKLETS=(1 2 3 4 5 6 9 12 16 20 24)
declare -a SAMPLES=(500 1000 5000 10000 15000 30000 60000)

# Scale the number of tasklets while keeping the number of samples 10k
echo "Tasklet scaling: 1 dpu, 1-24 tasklets, 60k samples"
for i in $(seq 0 10); do
    make clean &> /dev/null
    PRINT=0 PERF=CYCLES NR_TASKLETS=${TASKLETS[$i]} make &> /dev/null
    ./bin/host_code -i 60000 2> /dev/null | grep -a "results_and_timings"
done

# # Scale the number of samples while keeping the tasklet count 16
echo "Sample scaling: 1 dpu, 16 tasklets, 500-60k samples"
for i in $(seq 0 10); do
    make clean &> /dev/null
    PRINT=0 PERF=CYCLES NR_TASKLETS=16 make &> /dev/null
    ./bin/host_code -i ${SAMPLES[$i]} 2> /dev/null | grep -a "results_and_timings"
done

samples_total=60000
samples_min=$(($samples_total / 64))

echo "DPU strong scaling: 1-64 dpus, 16 tasklets, 60k samples"
for i in $(seq 0 6); do
    make clean &> /dev/null
    PRINT=0 PERF=CYCLES NR_TASKLETS=16 NR_DPUS=${DPUS[$i]} make &> /dev/null
    ./bin/host_code -i $samples_total 2> /dev/null | grep -a "results_and_timings"
done

echo "DPU weak scaling: 1-64 dpus, 16 tasklets, ${samples_min} samples per dpu"
for i in $(seq 0 6); do
    make clean &> /dev/null
    PRINT=0 PERF=CYCLES NR_TASKLETS=16 NR_DPUS=${DPUS[$i]} make &> /dev/null
    WORK=$((${DPUS[$i]} * $samples_min))
    ./bin/host_code -i $WORK 2> /dev/null | grep -a "results_and_timings"
done
