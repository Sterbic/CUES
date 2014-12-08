#ifndef SIMULATION_CUH_
#define SIMULATION_CUH_

#include <stdlib.h>

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
 * levels[N]: the iteration at which each node was discovered
 * inputFrontier[CSize]: the input edge frontier
 * outputFrontier[CSize]: the output edge frontier
 */
typedef struct {
	unsigned int nodes;
	unsigned int CSize;
	unsigned int *R;
	unsigned int *C;
	int *levels;
	unsigned int *inputFrontier;
	unsigned int *outputFrontier;
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

	CUDA_CHECK_RETURN(cudaGetSpaceAndSet(
			(void **) &context->levels,
			context->nodes,
			-1));

	CUDA_CHECK_RETURN(cudaGetSpaceAndSet(
			(void **) &context->inputFrontier,
			context->CSize,
			-1));

	CUDA_CHECK_RETURN(cudaGetSpaceAndSet(
			(void **) &context->outputFrontier,
			context->CSize,
			-1));

	return context;
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
		CUDA_CHECK_RETURN(cudaFree(context->inputFrontier));
		CUDA_CHECK_RETURN(cudaFree(context->outputFrontier));

		free(context);
	}
}

#endif
