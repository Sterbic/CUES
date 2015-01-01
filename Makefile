FLAGS = -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
LINKER_FLAGS = -lcurand
SRC = src/cues.cu

cues:
	nvcc -o $@ $(SRC) $(FLAGS) $(LINKER_FLAGS)


