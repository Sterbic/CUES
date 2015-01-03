/**
 * Device kernels for the CUES project.
 */

#ifndef KERLENS_CUH_
#define KERLENS_CUH_

#define DEBUG false

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
	unsigned int size = *frontierSize;
	unsigned int offset = 0;

	// read the prng state for the current thread from global memory
	curandState localState = states[tid];

	while(tid + offset < size) {
		unsigned int node = frontier[tid + offset];
		offset += blockDim.x * gridDim.x;

		float p = curand_uniform(&localState);
		float q = curand_uniform(&localState);

		pRand[node] = p;
		qRand[node] = q;
	}

	// write the updates prng state for the local thread back to global memory
	states[tid] = localState;
}

__global__ void contractExpand(int iteration, float p, float q,
		unsigned int nodes,	unsigned int *R, unsigned int *C,
		unsigned int *inFrontierSize, unsigned int *inputFrontier,
		unsigned int *outFrontierSize, unsigned int *outputFrontier,
		int *infected, unsigned char *nodeState, float *pRand, float *qRand) {
	// structure for warp culling
	__shared__ volatile unsigned int warpScratch[WARPS_PER_BLOCK][128];

	// structure for history culling
	__shared__ volatile int history[HISTORY_SIZE];

	// strucutre for the prescan calculations
	__shared__ unsigned int threadScratch[BLOCK_SIZE][3];

	// init indices
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int localTid = threadIdx.x;
	unsigned int warpId = threadIdx.x / WARP_SIZE;
	unsigned int laneId = threadIdx.x % WARP_SIZE;

	// init history hash table
	unsigned int historyOffset = 0;

	while(localTid + historyOffset < HISTORY_SIZE) {
		history[localTid + historyOffset] = -1;
		historyOffset += BLOCK_SIZE;
	}

	__syncthreads();

	int offset = 0;
	unsigned int frontierSize = *inFrontierSize;

	while(offset < frontierSize) {
		// get the node and all the flags for the current thread
		// initially assign to dummy node
		unsigned int node = nodes - 1;
		bool duplicate = true;

		// test if the thread is assigned to a valid node
		if(tid + offset < frontierSize) {
			node = inputFrontier[tid + offset];
			duplicate = false;
		}

		if(DEBUG && (tid < 5)) {
			printf("1 # Thread: %u, Node: %u\n", localTid, node);
		}

		offset += gridDim.x * blockDim.x;

		// do warp culling
		if(!duplicate) {
			unsigned int hash = node & 127;
			warpScratch[warpId][hash] = node;

			unsigned int retrivedNode = warpScratch[warpId][hash];

			if(DEBUG && (tid < 5)) {
				printf("1.5 # Thread: %u, node: %u, retrived: %u\n", localTid, node, retrivedNode);
			}

			if(retrivedNode == node) {
				warpScratch[warpId][hash] = localTid;
				unsigned int retrivedTid = warpScratch[warpId][hash];

				if(DEBUG && (tid < 5)) {
					printf("1.6 # Thread: %u, Retrived: %u\n", localTid, retrivedTid);
				}

				if(retrivedTid != localTid) {
					duplicate = true;
				}
			}
		}

		if(DEBUG && (tid < 5)) {
			printf("2 # Thread: %u, Node: %u, dup: %d\n", localTid, node, duplicate);
		}

		// test history if node is not a duplicate
		unsigned int historyHash = node % HISTORY_SIZE;
		unsigned int retrivedNode = history[historyHash];

		if(node == retrivedNode) {
			duplicate = true;
		} else {
			history[historyHash] = node;
		}

		if(DEBUG && (tid < 5)) {
			printf("3 # Thread: %u, Node: %u, dup: %d\n", localTid, node, duplicate);
		}

		// state flags are set to true by default
		unsigned char state = IMMUNE_AND_INFECTED_NEIGHBORS_MASK;
		bool immune = true;
		bool infectedNeighbors = true;

		// fetch node state if it is not a duplicate
		if(!duplicate) {
			state = nodeState[node];

			immune = state & IMMUNE_MASK;
			infectedNeighbors = state & DID_INFECT_NEIGHBORS_MASK;

			// node should be considered as a duplicate if it is immune
			duplicate = duplicate || immune;
		}

		if(DEBUG && (tid < 5)) {
			printf("3.1 # Thread: %u, Node: %u, dup: %d, immune: %d, infct: %d\n", threadIdx.x, node, duplicate, immune, infectedNeighbors);
		}

		// visit node if it should be visited
		if(!duplicate && infected[node] == -1) {
			infected[node] = iteration;
		}

		// try to recover the current node
		bool didRecover = true;

		if(!duplicate) {
			didRecover = qRand[node] < q;
		}

		// fetch adjacency list offsets if needed
		unsigned int rStart = 0;
		unsigned int rEnd = 0;

		// do test for expansion
		if(!duplicate && !infectedNeighbors) {
			float pTest = pRand[node];

			// if test is successful load offsets in C and set flag
			if(pTest < p) {
				rStart = R[node];
				rEnd = R[node + 1];
				infectedNeighbors = true;
			}
		}

		if(DEBUG && (tid < 5)) {
			printf("4 # Thread: %u, Node: %u, dup: %d\n", threadIdx.x, node, duplicate);
		}

		// calculate size for coarse and fine grained node gathering
		unsigned int rLength = rEnd - rStart;
		unsigned int coarseSize = 0;
		unsigned int fineSize = 0;
		unsigned int recoverySize = !didRecover;

		if(rLength > WARP_SIZE) {
			coarseSize = rLength;
		} else {
			fineSize = rLength;
		}

		if(DEBUG && (tid < 5 || !duplicate)) {
			printf("5 # Thread: %u, Node: %u, coarse: %u, fine: %u, rec: %u\n", threadIdx.x, node, coarseSize, fineSize, recoverySize);
		}

		// do prescan to determin local enqueue offsets
		__syncthreads();

		int offset = 1;
		threadScratch[localTid][0] = coarseSize;
		threadScratch[localTid][1] = fineSize;
		threadScratch[localTid][2] = recoverySize;

		// reduce phase
		for(int d = BLOCK_SIZE >> 1; d > 0; d >>= 1, offset <<= 1) {
			__syncthreads();

			if(localTid < d) {
				int ai = offset * (2 * localTid + 1) - 1;
				int bi = offset * (2 * localTid + 2) - 1;

				threadScratch[bi][0] += threadScratch[ai][0];
				threadScratch[bi][1] += threadScratch[ai][1];
				threadScratch[bi][2] += threadScratch[ai][2];
			}
		}

		if(localTid == 0) {
			threadScratch[BLOCK_SIZE - 1][0] = 0;
			threadScratch[BLOCK_SIZE - 1][1] = 0;
			threadScratch[BLOCK_SIZE - 1][2] = 0;
		}

		// up-sweep phase
		for(int d = 1; d < BLOCK_SIZE; d <<= 1) {
			offset >>= 1;
			__syncthreads();

			if(localTid < d) {
				int ai = offset * (2 * localTid + 1) - 1;
				int bi = offset * (2 * localTid + 2) - 1;

				unsigned int temp = threadScratch[ai][0];
				threadScratch[ai][0] = threadScratch[bi][0];
				threadScratch[bi][0] += temp;

				temp = threadScratch[ai][1];
				threadScratch[ai][1] = threadScratch[bi][1];
				threadScratch[bi][1] += temp;

				temp = threadScratch[ai][2];
				threadScratch[ai][2] = threadScratch[bi][2];
				threadScratch[bi][2] += temp;
			}
		}

		__syncthreads();

		// calculate global enqueue offset
		if(localTid == BLOCK_SIZE - 1) {
			unsigned int coarseTotal = threadScratch[BLOCK_SIZE - 1][0] +
					coarseSize;
			unsigned int fineTotal = threadScratch[BLOCK_SIZE - 1][1] +
					fineSize;
			unsigned int recoveryTotal = threadScratch[BLOCK_SIZE - 1][2] +
					recoverySize;

			unsigned int total = coarseTotal + fineTotal + recoveryTotal;
			unsigned int baseOffset = atomicAdd(outFrontierSize, total);

			if(DEBUG) {
				printf("6 # base: %u, coarse: %u, fine: %u\n", baseOffset, coarseTotal, fineTotal);
			}

			warpScratch[0][0] = baseOffset;
			warpScratch[0][1] = coarseTotal;
			warpScratch[0][2] = fineTotal;
		}

		__syncthreads();

		// broadcast base enqueue offset and totals
		unsigned int baseOffset = warpScratch[0][0];
		unsigned int coarseTotal = warpScratch[0][1];
		unsigned int fineTotal = warpScratch[0][2];

		// set the flag for block coarse gathering
		if(localTid == 0) {
			warpScratch[0][0] = BLOCK_SIZE;
		}

		__syncthreads();

		// perform coarse grained block gathering
		while(true) {
			// all threads with enough nodes vie for control
			if(coarseSize > BLOCK_SIZE) {
				warpScratch[0][0] = localTid;
			}

			__syncthreads();

			// get the winner and wxit the loop if all threads are done
			unsigned int winner = warpScratch[0][0];
			if(winner == BLOCK_SIZE) {
				break;
			}

			// winner thread broadcasts its offsets
			if(localTid == winner) {
				warpScratch[0][0] = BLOCK_SIZE;
				warpScratch[0][1] = rStart;
				warpScratch[0][2] = rEnd;
				coarseSize = 0;
			}

			__syncthreads();

			unsigned int outOffset = baseOffset + threadScratch[winner][0] +
								localTid;

			// get bounds is C
			unsigned int cIndex = warpScratch[0][1] + localTid;
			unsigned int cLast = warpScratch[0][2];

			// gather nodes, BLOCK_SIZE at a time
			while(cIndex < cLast) {
				unsigned int neighbor = C[cIndex];

				// store gathered neighbor in output frontier
				outputFrontier[outOffset] = neighbor;

				outOffset += BLOCK_SIZE;
				cIndex += BLOCK_SIZE;
			}
		}

		// perform coarse grained warp gathering
		while(__any(coarseSize)) {
			// vie for control of the warp
			if(coarseSize != 0) {
				warpScratch[warpId][0] = localTid;
			}

			// winner broadcasts its offsets
			if(warpScratch[warpId][0] == localTid) {
				warpScratch[warpId][1] = rStart;
				warpScratch[warpId][2] = rEnd;
				coarseSize = 0;
			}

			// get the local tid of the winner and its scatter offset
			unsigned int winner = warpScratch[warpId][0];
			unsigned int outOffset = baseOffset + threadScratch[winner][0] +
					laneId;

			// get bounds is C
			unsigned int cIndex = warpScratch[warpId][1] + laneId;
			unsigned int cLast = warpScratch[warpId][2];

			// gather nodes, WARP_SIZE at a time
			while(cIndex < cLast) {
				unsigned int neighbor = C[cIndex];

				// store gathered neighbor in output frontier
				outputFrontier[outOffset] = neighbor;

				outOffset += WARP_SIZE;
				cIndex += WARP_SIZE;
			}
		}

		// perform fine grained gathering
		unsigned int blockProgress = 0;
		unsigned int fineOffset = threadScratch[localTid][1];
		int remain = fineTotal;

		if(DEBUG && (tid < 5)) {
			printf("7 # Thread: %u, rStart: %u, rEnd: %u, remain: %u, offset: %d\n", threadIdx.x, rStart, rEnd, remain, fineOffset);
		}

		__syncthreads();

		// loop while there are nodes to gather
		while(remain > 0) {
			// load positions in shared memory
			while((fineOffset < blockProgress + BLOCK_SIZE) &&
					(rStart < rEnd)) {
				if(DEBUG) {
					printf("8 # Thread: %u, rStart: %u, index: %u\n", threadIdx.x, rStart, fineOffset - blockProgress);
				}

				threadScratch[fineOffset - blockProgress][0] = rStart;
				fineOffset++;
				rStart++;
			}

			// wait for all threads to load the shared buffer
			__syncthreads();

			// gather nodes
			if(localTid < remain) {
				unsigned int cIndex = threadScratch[localTid][0];
				unsigned int neighbor = C[cIndex];

				// store gathered neighbor in output frontier
				unsigned int outOffset = baseOffset + coarseTotal +
						blockProgress + localTid;
				outputFrontier[outOffset] = neighbor;

				if(DEBUG) {
					printf("9 # Thread: %u, Node: %u, index: %u\n", threadIdx.x, neighbor, outOffset);
				}
			}

			blockProgress += BLOCK_SIZE;
			remain -= BLOCK_SIZE;

			__syncthreads();
		}

		// perform recovery gahtering if needed
		if(!duplicate) {
			if(didRecover) {
				immune = true;
			} else {
				unsigned int outOffset = baseOffset + coarseTotal + fineTotal +
						threadScratch[localTid][2];
				outputFrontier[outOffset] = node;
			}
		}

		// update node state in global memory if needed, harcoded bit movements
		unsigned char newState = immune | infectedNeighbors << 1;

		if(DEBUG && (tid < 5)) {
			printf("3.1 # Thread: %u, Node: %u, old: %d, new: %d\n", threadIdx.x, node, state, newState);
		}

		if(newState != state) {
			nodeState[node] = newState;
		}
	}
}

#endif
