/**
 * Global constants for the CUES project
 */

#ifndef GLOBALS_CUH_
#define GLOBALS_CUH_

// current configuration uses 13312 bytes of shred memory per block

#define WARP_SIZE 32
#define BLOCK_SIZE 512
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define MAX_GRID_SIZE 128

#define SEED 17

#endif
