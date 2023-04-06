#!/bin/bash

declare -a TASKLETS=(1 2 3 4 5 6 9 12 16 20 24)
declare -a BLOCKS=(11 10 9 8 7)

for i in $(seq 0 10); do
    echo "#tasklets: ${TASKLETS[$i]}, block: 9"
    make clean &> /dev/null
    NR_DPUS=32 NR_TASKLETS=${TASKLETS[$i]} BLOCK=9 TYPE=INT32 TRANSFER=PARALLEL PRINT=1 PERF=INSTRUCTIONS make &> /dev/null
    ./bin/host_code -w 0 -e 1 -i 2097152 -a 20 2> /dev/null | grep "DPU instructions\|instr."
done

# for i in $(seq 0 0); do
#     echo "#tasklets: ${TASKLETS[$i]}, block: ${BLOCKS[$i]}"
#     make clean
#     NR_DPUS=32 NR_TASKLETS=${TASKLETS[$i]} BLOCK=${BLOCKS[$i]} TYPE=INT32 TRANSFER=PARALLEL PRINT=0 PERF=INSTRUCTIONS make
#     ./bin/host_code -w 0 -e 1 -i 1048576 -a 20 2> /dev/null
# done

