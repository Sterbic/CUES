/**
 * Main module of the CUES project.
 */

#include <cstdio>
#include <cstdlib>
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "utils.cuh"
#include "graph.cuh"
#include "simulation.cuh"
#include "kernels.cuh"

__device__ unsigned int bitreverse(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void bitreverse(void *data) {
	unsigned int *idata = (unsigned int*) data;
	idata[threadIdx.x] = bitreverse(idata[threadIdx.x]);
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
/*int main(void) {
	void *d = NULL;
	int i;
	unsigned int idata[WORK_SIZE], odata[WORK_SIZE];

	for (i = 0; i < WORK_SIZE; i++)
		idata[i] = (unsigned int) i;

	CUDA_CHECK_RETURN(cudaMalloc((void**) &d, sizeof(int) * WORK_SIZE));
	CUDA_CHECK_RETURN(
			cudaMemcpy(d, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice));

	bitreverse<<<1, WORK_SIZE, WORK_SIZE * sizeof(int)>>>(d);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost));

	for (i = 0; i < WORK_SIZE; i++)
		printf("Input value: %u, device output: %u\n", idata[i], odata[i]);

	CUDA_CHECK_RETURN(cudaFree((void*) d));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}*/

int main(int argc, char **argv) {
	if(argc != 6) {
		printUsage();
		exit(1);
	}

	char *graphPath = argv[1];
	int patientZero = atoi(argv[2]);
	double p = atof(argv[3]);
	double q = atof(argv[4]);
	int simulations = atoi(argv[5]);

	exitIf(p < 0.0 || p > 1.0,
			"The p parameter should be from the interval [0-1].");
	exitIf(q < 0.0 || q > 1.0,
			"The q parameter should be from the interval [0-1].");
	exitIf(simulations < 1, "The number of simulations should be at least 1.");

	printf("Input parameters:\n");

	printf("\t%-25s %s\n", "Graph file:", graphPath);
	printf("\t%-25s %d\n", "Source node:", patientZero);
	printf("\t%-25s %.2f\n", "Q:", q);
	printf("\t%-25s %.2f\n", "P:", p);
	printf("\t%-25s %d\n", "Simulations:", simulations);

	printf("\nSearching for best device... ");

	int devideID = cudaGetMaxGflopsDeviceID();
	CUDA_CHECK_RETURN(cudaSetDevice(devideID));

	cudaDeviceProp deviceProperties;
	CUDA_CHECK_RETURN(cudaGetDeviceProperties(&deviceProperties, devideID));

	int deviceMajor = deviceProperties.major;
	int deviceMinor = deviceProperties.minor;
	int deviceMPs = deviceProperties.multiProcessorCount;
	int residentThreadsPerMP = deviceProperties.maxThreadsPerMultiProcessor;
	int totalResidentThreads = deviceMPs * residentThreadsPerMP;

	printf("DONE\n");
	printf("\t%-25s %s\n", "Device:", deviceProperties.name);
	printf("\t%-25s %d.%d\n", "Capability", deviceMajor, deviceMinor);
	printf("\t%-25s %d\n", "Multiprocessors", deviceMPs);
	printf("\t%-25s %d\n", "Total CUDA cores", deviceMPs
			* convertSMVersion2Cores(deviceMajor, deviceMinor));
	printf("\t%-25s %d\n", "Total resident threads:", totalResidentThreads);

	printf("\nLoading graph... ");
	Graph *graph = loadGraph(graphPath);
	printf("DONE\n");

	printIntArray((int *)graph->R, graph->RSize, true);
	printIntArray((int *) graph->C, graph->CSize, true);

	exitIf(patientZero < 0 || patientZero > graph->N - 1,
			"Source node is not present in the input graph.");

	printf("\t%-25s %u\n", "Nodes:", graph->N);
	printf("\t%-25s %u\n", "Edges:", graph->M);

	printf("\nCreating simulation context... ");
	SimulationContext *context = createSimulationContext(graph);
	printf("DONE\n");

	printf("\nInitializing random generator states... ");
	initRandoms<<<MAX_GRID_SIZE, BLOCK_SIZE>>>(context->randStates, SEED);
	printf("DONE\n\n");

	for(int simulation = 1; simulation <= simulations; simulation++) {
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		printf("Running %d. simulation... ", simulation);
		prepareSimulationContext(context, patientZero);

		int iteration = 0;
		unsigned int inputSize = 1;

		do {
			unsigned int blocks = (inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
			blocks = min(blocks, MAX_GRID_SIZE);

			printf("Input size, blocks: %d %d\n", inputSize, blocks);

			generateRandForFrontier<<<blocks, BLOCK_SIZE>>>(
					context->randStates,
					context->inputFrontier,
					context->inFrontierSize,
					context->pRand,
					context->qRand
			);

			contractExpand<<<blocks, BLOCK_SIZE>>>(
					iteration,
					p,
					q,
					context->nodes,
					context->R,
					context->C,
					context->inFrontierSize,
					context->inputFrontier,
					context->outFrontierSize,
					context->outputFrontier,
					context->infected,
					context->immune,
					context->didInfectNeighbors,
					context->pRand,
					context->qRand
			);

			iterationDone(context);
			iteration++;

			inputSize = getInputFrontierSize(context);
			printf("Frontier size: %u -> ", inputSize);
			printIntArray((int *) context->inputFrontier, inputSize, false);
		} while(inputSize > 0); // inputSize != 0

		printIntArray(context->infected, context->nodes - 1, false);
		printBoolArray(context->immune, context->nodes - 1, false);
		printBoolArray(context->didInfectNeighbors, context->nodes - 1, false);
		printf(" DONE\n");
	}

	freeSimulationContext(context);
	freeGraph(graph);

	CUDA_CHECK_RETURN(cudaDeviceReset());
}
