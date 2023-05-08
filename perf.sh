#!/bin/bash

declare -a TASKLETS=(1 2 3 4 5 6 9 12 16 20 24)
declare -a SAMPLES=(1 10 100 500 1000 2000 5000 10000)

# Scale the number of samples while keeping the tasklet count 16
# echo "Sample scaling"
# for i in $(seq 0 10); do
#     make clean &> /dev/null
#     PRINT=0 PERF=INSTRUCTIONS NR_TASKLETS=16 make &> /dev/null
#     ./bin/host_code -i ${SAMPLES[$i]} 2> /dev/null | grep -a "DPU instructions, "
# done

# Scale the number of tasklets while keeping the number of samples 10k
echo "Tasklet scaling"
for i in $(seq 8 10); do
    make clean &> /dev/null
    PRINT=0 PERF=INSTRUCTIONS NR_TASKLETS=${TASKLETS[$i]} make &> /dev/null
    ./bin/host_code -i 10000 2> /dev/null | grep -a "DPU instructions, "
done
