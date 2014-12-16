/**
 * Device kernels for the CUES project.
 */

#ifndef KERLENS_CUH_
#define KERLENS_CUH_

#include <curand_kernel.h>

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
}

#endif
