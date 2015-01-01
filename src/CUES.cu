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

/**
 * Usage: ./cues <graph_path> <source_node> <p> <q> <simulations>
 * graph_path: the path to the graph in edge list format of the network that
 * will be used in the simulation
 * source_node: the node ID of the start of the epidemics, patient zero
 * p: the probability that a node will infect its neighbors
 * q: the probability that a node will recover and become immune
 * simulations: the number of simulations to run
 */
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

	clock_t globalClock = clock();

	for(int simulation = 1; simulation <= simulations; simulation++) {
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		clock_t simulationClock = clock();

		printf("Running %d. simulation...\n", simulation);
		prepareSimulationContext(context, patientZero);

		int iteration = 0;
		unsigned int inputSize = 1;

		// loop while the input frontier is not empty
		do {
			clock_t iterationClock = clock();

			unsigned int blocks = (inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
			blocks = min(blocks, MAX_GRID_SIZE);

			// refresh random test values for each node in the input frontier
			generateRandForFrontier<<<blocks, BLOCK_SIZE>>>(
					context->randStates,
					context->inputFrontier,
					context->inFrontierSize,
					context->pRand,
					context->qRand
			);

			// run the contract-expand kernel
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

			printf("\t%3d. iteration output - frontier size: %u, ", iteration,
					inputSize);
			printf("time elapsed %.3f ms\n", getElapsedTimeMS(iterationClock));
		} while(inputSize > 0);

		printf("Simulation ended - elapsed time %.3f ms\n\n",
				getElapsedTimeMS(simulationClock));
	}

	printf("All simulations ended - elapsed time %.3f ms\n",
			getElapsedTimeMS(globalClock));

	freeSimulationContext(context);
	freeGraph(graph);

	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}
