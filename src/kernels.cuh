/**
 * Device kernels for the CUES project.
 */

#ifndef KERLENS_CUH_
#define KERLENS_CUH_

#include <curand_kernel.h>

// HISTORY_SIZE must be a power of 2
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

__global__ void contractExpand(int iteration, float p, float q,
		unsigned int nodes,	unsigned int *R, unsigned int *C,
		unsigned int *inFrontierSize, unsigned int *inputFrontier,
		int *infected, bool *didInfectNeighbors, float *pRand, float *qRand) {
	// structure for warp culling
	__shared__ unsigned int scratch[WARPS_PER_BLOCK][128];

	// structure for history culling
	__shared__ unsigned int history[HISTORY_SIZE];

	// strucutre for the prescan calculations
	__shared__ unsigned int scanBuffer[BLOCK_SIZE][2];

	// init indices
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int localTid = threadIdx.x;
	int warpId = blockIdx.x / WARP_SIZE;

	int offset = 0;
	unsigned int frontierSize = *inFrontierSize;

	while(offset < frontierSize) {
		// get the node and all the flags for the current thread
		// initially assign to dummy node
		unsigned int node = nodes;

		// test if the thread is assigned to a valid node
		if(tid + offset < frontierSize) {
			node = inputFrontier[tid + offset];
		}

		offset += gridDim.x * blockDim.x;

		// test if the node is a duplicate
		bool duplicate = false;

		// do warp culling
		unsigned int hash = node & 127;
		scratch[warpId][hash] = node;

		if(scratch[warpId][hash] == node) {
			scratch[warpId][hash] = localTid;

			if(scratch[warpId][hash] != localTid) {
				duplicate = true;
			}
		}

		// test history if node is not a duplicate
		unsigned int historyHash = node & (HISTORY_SIZE - 1);

		if(!duplicate && history[historyHash] == node) {
			duplicate = true;
		}

		bool shouldVisit = false;
		bool shouldExpand = false;

		// if the node is not a duplicate init action flags
		if(!duplicate) {
			shouldVisit = infected[node] == -1;
			shouldExpand = !didInfectNeighbors[node];
		}

		// visit node if it should be visited
		if(shouldVisit) {
			history[historyHash] = node;
			infected[node] = iteration;
		}

		// try to recover the current node
		float qTest = qRand[node];
		bool didRecover = qTest < q;

		// fetch adjacency list offsets if needed
		unsigned int rStart = 0;
		unsigned int rEnd = 0;

		if(shouldExpand) {
			rStart = R[node];
			rEnd = R[node + 1];
		}

		// calculate size for coarse and fine grained node gathering

		unsigned int rLength = rEnd - rStart + !didRecover;
		unsigned int coarseSize = 0;
		unsigned int fineSize = 0;

		if(rLength > WARP_SIZE) {
			coarseSize = rLength;
		} else {
			fineSize = rLength;
		}

		// do prescan to determin local enqueue offsets
		__syncthreads();

		int offset = 1;
		scanBuffer[localTid][0] = coarseSize;
		scanBuffer[localTid][1] = fineSize;

		// reduce phase
		for(int d = BLOCK_SIZE >> 1; d > 0; d >>= 1, offset <<= 1) {
			__syncthreads();

			if(localTid < d) {
				int ai = offset * (2 * localTid + 1) - 1;
				int bi = offset * (2 * localTid + 2) - 1;

				scanBuffer[bi][0] += scanBuffer[ai][0];
				scanBuffer[bi][1] += scanBuffer[ai][1];
			}
		}

		if(localTid == 0) {
			scanBuffer[BLOCK_SIZE - 1][0] = 0;
			scanBuffer[BLOCK_SIZE - 1][1] = 0;
		}

		// up-sweep phase
		for(int d = 1; d < BLOCK_SIZE; d <<= 1) {
			offset >>= 1;
			__syncthreads();

			if(localTid < d) {
				int ai = offset * (2 * localTid + 1) - 1;
				int bi = offset * (2 * localTid + 2) - 1;

				unsigned int temp = scanBuffer[ai][0];
				scanBuffer[ai][0] = scanBuffer[bi][0];
				scanBuffer[bi][0] += temp;

				temp = scanBuffer[ai][1];
				scanBuffer[ai][1] = scanBuffer[bi][1];
				scanBuffer[bi][1] += temp;
		}

		__syncthreads();

		// to be continued ...

		// fetch test values if they will be needed
		float pTest = 0;


		if(shouldExpand) {
			pTest = pRand[node];
		}
	}
}

#endif
