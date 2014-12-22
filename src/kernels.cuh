/**
 * Device kernels for the CUES project.
 */

#ifndef KERLENS_CUH_
#define KERLENS_CUH_

#include <curand_kernel.h>

#define HISTORY_SIZE 256

/**
 * Initializes the random generator state for each thread.
 * states[MAX_THREADS]: the state of the prng for each thread
 * seed: the seed to initialize the prng
 */
__global__ void initRandoms(curandState *states, unsigned int seed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, tid, 0, &states[tid]);
}

/**
 * Generate p and q random values for all nodes in the given frontier.
 * states[MAX_THREADS]: the state of the prng for each thread
 * frontier[CSize]: the input edge frontier
 * frontierSize: the number of valid elements in the frontier
 * pRand[N]: probabilities to be avaluated agains p (infect neighbors)
 * qRand[N]: probabilities to be avaluated agains q (recover)
 */
__global__ void generateRandForFrontier(curandState *states,
		unsigned int *frontier, unsigned int *frontierSize, float *pRand,
		float *qRand) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = states[tid];

	while(tid < *frontierSize) {
		unsigned int node = frontier[tid];
		tid += blockDim.x * gridDim.x;

		float p = curand_uniform(&localState);
		float q = curand_uniform(&localState);

		pRand[node] = p;
		qRand[node] = q;
	}

	states[tid] = localState;
}

__global__ void contractExpand(unsigned int *R, unsigned int *C,
		unsigned int *inFrontierSize, unsigned int *inputFrontier, int *levels,
		int iteration, float *pRand, float *qRand) {
	// structure for warp culling
	__shared__ unsigned int scratch[WARPS_PER_BLOCK][128];

	// structure for history culling
	__shared__ unsigned int history[HISTORY_SIZE];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int warpID = blockIdx.x / WARP_SIZE;

	while(tid < *inFrontierSize) {
		// get the node and level for the current thread
		unsigned int node = inputFrontier[tid];
		int level = levels[node];

		tid += gridDim.x * blockDim.x;

		// if node was not visited in previous iterations do more checks
		if(level != -1) {
			// do warp culling
			unsigned int hash = node & 127;
			scratch[warpID][hash] = node;

			if(scratch[warpID][hash] == node) {
				scratch[warpID][hash] = threadIdx.x;

				if(scratch[warpID][hash] != threadIdx.x) {
					continue;
				}
			}
		}

		// level test and warp culling passed, test history
		unsigned int historyHash = node & (HISTORY_SIZE - 1);
		if(history[historyHash] == node) {
			continue;
		}

		// visit node
		history[historyHash] = node;
		levels[node] = iteration;

		// fetch adjacency list offsets
		unsigned int rStart = R[node];
		unsigned int rEnd = R[node + 1];

		// fetch test values
		float pTest = pRand[node];
		float qTest = qRand[node];

		// to be continued ... add immune array to context
	}
}

#endif
