/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>

#include "utils.cuh"
#include "graph.cuh"
#include "simulation.cuh"

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

	printf("\t%-20s %s\n", "Graph file:", graphPath);
	printf("\t%-20s %d\n", "Source node:", patientZero);
	printf("\t%-20s %.2f\n", "Q:", q);
	printf("\t%-20s %.2f\n", "P:", p);
	printf("\t%-20s %d\n", "Simulations:", simulations);

	printf("\nSearching for best device... ");

	int devideID = cudaGetMaxGflopsDeviceID();
	cudaDeviceProp deviceProperties;
	CUDA_CHECK_RETURN(cudaGetDeviceProperties(&deviceProperties, devideID));

	int deviceMajor = deviceProperties.major;
	int deviceMinor = deviceProperties.minor;
	int deviceMPs = deviceProperties.multiProcessorCount;

	printf("DONE\n");
	printf("\t%-20s %s\n", "Device:", deviceProperties.name);
	printf("\t%-20s %d.%d\n", "Capability", deviceMajor, deviceMinor);
	printf("\t%-20s %d\n", "Multiprocessors", deviceMPs);
	printf("\t%-20s %d\n", "Total CUDA cores", deviceMPs
			* convertSMVersion2Cores(deviceMajor, deviceMinor));

	printf("\nLoading graph... ");
	Graph *graph = loadGraph(graphPath);
	printf("DONE\n");

	exitIf(patientZero < 0 || patientZero > graph->N - 1,
			"Source node is not present in the input graph.");

	printf("\t%-20s %u\n", "Nodes:", graph->N);
	printf("\t%-20s %u\n", "Edges:", graph->M);

	printf("\nCreating simulation context... ");
	SimulationContext *context = createSimulationContext(graph);
	printf("DONE\n");

	freeSimulationContext(context);
	freeGraph(graph);
}
