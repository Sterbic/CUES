FLAGS = -gencode arch=compute_30,code=sm_30
LINKER_FLAGS = -lcurand
SRC = src/cues.cu

cues:
	nvcc -g -G -o $@ $(SRC) $(FLAGS) $(LINKER_FLAGS)
