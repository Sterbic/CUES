FLAGS = -gencode arch=compute_30,code=sm_30
LINKER_FLAGS = -lcurand
SRC = src/cues.cu

cues:
	nvcc -o $@ $(SRC) $(FLAGS) $(LINKER_FLAGS)
