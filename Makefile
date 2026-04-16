# Makefile for batched-lenet-cuda
# Requires CUDA 11+ and an Ampere-class GPU (sm_80+; tested on sm_86).

NVCC      ?= nvcc
CUDA_ARCH ?= 86
NVCCFLAGS  = -O3 -std=c++17 -arch=sm_$(CUDA_ARCH) -Iinclude -lineinfo
LDFLAGS    = -lcudart

BIN_DIR    = bin
SRC_DIR    = src
CONV_DIR   = $(SRC_DIR)/conv

VARIANTS   = baseline fused tensor_cores register_tiled

BINARIES   = $(addprefix $(BIN_DIR)/, $(VARIANTS))

.PHONY: all clean test bench

all: $(BINARIES)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Each variant compiles main.cu with a different conv implementation and a
# preprocessor flag selecting which kernel to call.
$(BIN_DIR)/baseline: $(SRC_DIR)/main.cu $(CONV_DIR)/baseline.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -DVARIANT_BASELINE       $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/fused: $(SRC_DIR)/main.cu $(CONV_DIR)/fused.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -DVARIANT_FUSED          $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/tensor_cores: $(SRC_DIR)/main.cu $(CONV_DIR)/tensor_cores.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -DVARIANT_TENSOR_CORES   $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/register_tiled: $(SRC_DIR)/main.cu $(CONV_DIR)/register_tiled.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -DVARIANT_REGISTER_TILED $^ -o $@ $(LDFLAGS)

CORRECTNESS = $(addprefix $(BIN_DIR)/correctness_, $(VARIANTS))

test: $(CORRECTNESS)
	@set -e; for bin in $(CORRECTNESS); do \
	    echo "--- Running $$bin ---"; \
	    $$bin; \
	done
	@echo "All correctness tests PASSED."

$(BIN_DIR)/correctness_baseline: tests/correctness.cu $(CONV_DIR)/baseline.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/correctness_fused: tests/correctness.cu $(CONV_DIR)/fused.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/correctness_tensor_cores: tests/correctness.cu $(CONV_DIR)/tensor_cores.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/correctness_register_tiled: tests/correctness.cu $(CONV_DIR)/register_tiled.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

bench: all
	bash bench/run_all.sh

clean:
	rm -rf $(BIN_DIR)
