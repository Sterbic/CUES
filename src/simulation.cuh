#ifndef SIMULATION_CUH_
#define SIMULATION_CUH_

#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

#include "globals.cuh"
#include "utils.cuh"
#include "graph.cuh"

/**
 * A structure to store all the data and parameters needed for a simualtion.
 * Parameters  are stored on host memory while data arrays are allocated on
 * device memory.
 * nodes: the number of nodes in the graph
 * CSize: the size of the C array
 * R[N+1]: contains the indices in C for the start of the adjacency lists
 * C[CSize]: concatenated adjacency lists
 * infected[N]: the iteration at which each node was infected
 * immune[N]: bool array, true if the node is immune
 * didInfectNeighbors[stateSize]: bool array, true if node infected neighbours
 * inFrontierSize: the current size of the input frontier
 * inputFrontier[CSize]: the input edge frontier
 * outFrontierSize: the current size of the output frontier
 * outputFrontier[CSize]: the output edge frontier
 * randState[MAX_THREADS]: the state of the prng for each thread
 * pRand[N]: probabilities to be avaluated agains p (infect neighbors)
 * qRand[N]: probabilities to be avaluated agains q (recover)
 */
typedef struct {
	unsigned int nodes;
	unsigned int CSize;
	unsigned int *R;
	unsigned int *C;
	int *infected;
	bool *immune;
	bool *didInfectNeighbors;
	unsigned int *inFrontierSize;
	unsigned int *inputFrontier;
	unsigned int *outFrontierSize;
	unsigned int *outputFrontier;
	curandState *randStates;
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

	context->nodes = graph->N + 1;
	context->CSize = graph->CSize;

	CUDA_CHECK_RETURN(cudaGetDeviceCopy(
			graph->R,
			&context->R,
			(graph->RSize) * sizeof(unsigned int)
	));

	CUDA_CHECK_RETURN(cudaGetDeviceCopy(
			graph->C,
			&context->C,
			(graph->CSize) * sizeof(unsigned int)
	));

	CUDA_CHECK_RETURN(cudaMalloc(
			&context->infected, context->nodes * sizeof(int)
	));

	CUDA_CHECK_RETURN(cudaMalloc(
			&context->immune, context->nodes * sizeof(bool)
	));

	CUDA_CHECK_RETURN(cudaMalloc(
			&context->didInfectNeighbors, context->nodes * sizeof(bool)
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

	CUDA_CHECK_RETURN(cudaMalloc(
			&context->randStates,
			BLOCK_SIZE * MAX_GRID_SIZE * sizeof(curandState)
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
 * Sets the initial values of the context before each simulation.
 * context: the context pointer returned by createSimulationContext
 * src: the source node for the current simulation
 */
void prepareSimulationContext(SimulationContext *context, unsigned int src) {
	static unsigned int one = 1;

	CUDA_CHECK_RETURN(cudaMemset(
			context->infected, -1, context->nodes * sizeof(int)
	));

	CUDA_CHECK_RETURN(cudaMemset(
			context->immune, 0, context->nodes * sizeof(bool)
	));

	CUDA_CHECK_RETURN(cudaMemset(
			context->didInfectNeighbors, 0, context->nodes * sizeof(bool)
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
 * Called when an iteration of the simulaion is completed. Prepares the context
 * for the next iteration.
 * context: the context pointer returned by createSimulationContext
 */
void iterationDone(SimulationContext *context) {
	unsigned int *temp = context->inFrontierSize;
	context->inFrontierSize = context->outFrontierSize;
	context->outFrontierSize = temp;

	temp = context->inputFrontier;
	context->inputFrontier = context->outputFrontier;
	context->outputFrontier = temp;

	CUDA_CHECK_RETURN(cudaMemset(
			context->outFrontierSize, 0, sizeof(unsigned int)
	));
}

/**
 * Returns the size of the input frontier stored on the device.
 * context: the context pointer returned by createSimulationContext
 * returns: the size of the input frontier
 */
unsigned int getInputFrontierSize(SimulationContext *context) {
	unsigned int frontierSize;

	CUDA_CHECK_RETURN(cudaMemcpy(
			&frontierSize,
			context->inFrontierSize,
			sizeof(unsigned int),
			cudaMemcpyDeviceToHost
	));

	return frontierSize;
}

/**
 * Free the memory allocated by a call to createSimulationContext
 * context: the context pointer returned by createSimulationContext
 */
void freeSimulationContext(SimulationContext *context) {
	if(context != NULL) {
		CUDA_CHECK_RETURN(cudaFree(context->R));
		CUDA_CHECK_RETURN(cudaFree(context->C));
		CUDA_CHECK_RETURN(cudaFree(context->infected));
		CUDA_CHECK_RETURN(cudaFree(context->immune));
		CUDA_CHECK_RETURN(cudaFree(context->didInfectNeighbors));
		CUDA_CHECK_RETURN(cudaFree(context->inFrontierSize));
		CUDA_CHECK_RETURN(cudaFree(context->inputFrontier));
		CUDA_CHECK_RETURN(cudaFree(context->outFrontierSize));
		CUDA_CHECK_RETURN(cudaFree(context->outputFrontier));
		CUDA_CHECK_RETURN(cudaFree(context->randStates));
		CUDA_CHECK_RETURN(cudaFree(context->pRand));
		CUDA_CHECK_RETURN(cudaFree(context->qRand));

		free(context);
	}
}

#endif
