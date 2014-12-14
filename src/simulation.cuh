#ifndef SIMULATION_CUH_
#define SIMULATION_CUH_

#include <stdlib.h>
#include <curand.h>

#include "utils.cuh"
#include "graph.cuh"

#define SEED 5

/**
 * A structure to store all the data and parameters needed for a simualtion.
 * Parameters  are stored on host memory while data arrays are allocated on
 * device memory.
 * nodes: the number of nodes in the graph
 * CSize: the size of the C array
 * R[N+1]: contains the indices in C for the start of the adjacency lists
 * C[CSize]: concatenated adjacency lists
 * levels[N]: the iteration at which each node was discovered
 * inFrontierSize: the current size of the input frontier
 * inputFrontier[CSize]: the input edge frontier
 * outFrontierSize: the current size of the output frontier
 * outputFrontier[CSize]: the output edge frontier
 * prng: the random generator used in the simulations
 * pRand[N]: probabilities to be avaluated agains p (infect neighbors)
 * qRand[N]: probabilities to be avaluated agains q (recover)
 */
typedef struct {
	unsigned int nodes;
	unsigned int CSize;
	unsigned int *R;
	unsigned int *C;
	int *levels;
	unsigned int *inFrontierSize;
	unsigned int *inputFrontier;
	unsigned int *outFrontierSize;
	unsigned int *outputFrontier;
	curandGenerator_t prng;
	float *pRand;
	float *qRand;
} SimulationContext;

/**
 * Create a simualtion context given a network graph.
 * graph: a graph describing the network to be used in the simulations
 * returnes: a pointer to the created context
 */
SimulationContext *createSimulationContext(Graph *graph) {
	SimulationContext *context =
			(SimulationContext *) malloc(sizeof(SimulationContext));
	exitIf(context == NULL, "Error allocating simulation context.");

	context->nodes = graph->N;
	context->CSize = graph->CSize;

	CUDA_CHECK_RETURN(cudaGetDeviceCopy(
			(void *) graph->R,
			(void **) &context->R,
			(graph->N + 1) * sizeof(unsigned int)
	));

	CUDA_CHECK_RETURN(cudaGetDeviceCopy(
			(void *) graph->C,
			(void **) &context->C,
			(graph->CSize) * sizeof(unsigned int)
	));

	CUDA_CHECK_RETURN(cudaMalloc(
			&context->levels, context->nodes * sizeof(int)
	));

	CUDA_CHECK_RETURN(cudaMalloc(
			&context->inFrontierSize, sizeof(unsigned int)
	));

	CUDA_CHECK_RETURN(cudaMalloc(
			&context->inputFrontier, context->CSize * sizeof(unsigned int)
	));

	CUDA_CHECK_RETURN(cudaMalloc(
			&context->outFrontierSize, sizeof(unsigned int)
	));

	CUDA_CHECK_RETURN(cudaMalloc(
			&context->outputFrontier, context->CSize * sizeof(unsigned int)
	));

    CURAND_CHECK_RETURN(curandCreateGenerator(
    		&context->prng, CURAND_RNG_PSEUDO_DEFAULT
	));

    CURAND_CHECK_RETURN(curandSetPseudoRandomGeneratorSeed(
    		context->prng, SEED
	));

    CUDA_CHECK_RETURN(cudaMalloc(
    		&context->pRand, context->nodes * sizeof(float)
	));

    CUDA_CHECK_RETURN(cudaMalloc(
    		&context->qRand, context->nodes * sizeof(float)
	));

	return context;
}

/**
 * Fills the pRand and qRand device arrays of the given context with uniform
 * random floats from the <0-1] interval.
 * context: the context pointer returned by createSimulationContext
 */
void generatePQRandoms(SimulationContext *context) {
    CURAND_CHECK_RETURN(curandGenerateUniform(
    		context->prng, context->pRand, context->nodes
	));

    CURAND_CHECK_RETURN(curandGenerateUniform(
    		context->prng, context->qRand, context->nodes
	));
}

/**
 * Sets the initial values of the context before each simulation.
 * context: the context pointer returned by createSimulationContext
 * src: the source node for the current simulation
 */
void prepareSimulationContext(SimulationContext *context, unsigned int src) {
	static unsigned int one = 1;

	CUDA_CHECK_RETURN(cudaMemset(
			context->levels, -1, context->nodes * sizeof(int)
	));

	CUDA_CHECK_RETURN(cudaMemcpy(
			context->inFrontierSize,
			&one,
			sizeof(unsigned int),
			cudaMemcpyHostToDevice
	));

	CUDA_CHECK_RETURN(cudaMemcpy(
			context->inputFrontier,
			&src,
			sizeof(unsigned int),
			cudaMemcpyHostToDevice
	));

	CUDA_CHECK_RETURN(cudaMemset(
			context->outFrontierSize, 0, sizeof(unsigned int)
	));
}

/**
 * Free the memory allocated by a call to createSimulationContext
 * context: the context pointer returned by createSimulationContext
 */
void freeSimulationContext(SimulationContext *context) {
	if(context != NULL) {
		CUDA_CHECK_RETURN(cudaFree(context->R));
		CUDA_CHECK_RETURN(cudaFree(context->C));
		CUDA_CHECK_RETURN(cudaFree(context->levels));
		CUDA_CHECK_RETURN(cudaFree(context->inFrontierSize));
		CUDA_CHECK_RETURN(cudaFree(context->inputFrontier));
		CUDA_CHECK_RETURN(cudaFree(context->outFrontierSize));
		CUDA_CHECK_RETURN(cudaFree(context->outputFrontier));
		CUDA_CHECK_RETURN(cudaFree(context->pRand));
		CUDA_CHECK_RETURN(cudaFree(context->qRand));

		CURAND_CHECK_RETURN(curandDestroyGenerator(context->prng));

		free(context);
	}
}

#endif
