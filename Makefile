DPU_DIR := dpu
HOST_DIR := host
BUILDDIR ?= bin
CBTHOWEN_DIR := cbthowen
NR_DPUS ?= 1
NR_TASKLETS ?= 1
PRINT ?= 0
PERF ?= NO

define conf_filename
	${BUILDDIR}/.NR_DPUS_$(1)_NR_TASKLETS_$(2)_PRINT_$(6)_PERF_$(7).conf
endef
CONF := $(call conf_filename,${NR_DPUS},${NR_TASKLETS},${PRINT},${PERF})

HOST_TARGET := ${BUILDDIR}/host_code
DPU_TARGET := ${BUILDDIR}/dpu_code

COMMON_INCLUDES := support

ALL_HOST_SOURCES := $(wildcard ${CBTHOWEN_DIR}/*.c ${HOST_DIR}/*.c) # collect all sources..
HOST_SOURCES := $(filter-out ${CBTHOWEN_DIR}/main.c, $(ALL_HOST_SOURCES)) # ..and exclude the unwanted main.c from libcbthowen
DPU_SOURCES := $(wildcard ${DPU_DIR}/*.c)

.PHONY: all clean test

__dirs := $(shell mkdir -p ${BUILDDIR})

COMMON_FLAGS := -Wall -Wextra -g -I${COMMON_INCLUDES}
HOST_FLAGS := ${COMMON_FLAGS} -std=c11 -O3 -lm -DUSE_SIMULATOR `dpu-pkg-config --cflags --libs dpu` -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPUS=${NR_DPUS} -DPRINT=${PRINT} -D${PERF}
DPU_FLAGS := ${COMMON_FLAGS} -O2 -DNR_TASKLETS=${NR_TASKLETS} -DPRINT=${PRINT} -D${PERF}

all: ${HOST_TARGET} ${DPU_TARGET}

${CONF}:
	$(RM) $(call conf_filename,*,*)
	touch ${CONF}

${HOST_TARGET}: ${HOST_SOURCES} ${COMMON_INCLUDES} ${CONF}
	$(CC) -o $@ ${HOST_SOURCES} ${HOST_FLAGS}

${DPU_TARGET}: ${DPU_SOURCES} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${DPU_SOURCES}

clean:
	$(RM) -r $(BUILDDIR)

test: all
	./${HOST_TARGET}
