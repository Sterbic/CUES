#ifndef UTILS_CUH_
#define UTILS_CUH_

#include <stdio.h>

/**
 * Standard CUDA error check macro.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	}}

/**
 * Print the given message to stderr and exit the program.
 * message: the error message to print
 */
void exitWithMessage(const char *message) {
	fprintf(stderr, "%s", message);
	exit(1);
}

/**
 * If the given condition is true, print the given message and exit.
 * condition: true if the program should exit, false otherwise
 * message: the error message to print
 */
void exitIf(int condition, const char *message) {
	if(condition) {
		exitWithMessage(message);
	}
}

/**
 * Prints the CUES usage to stdout.
 */
void printUsage() {
	printf("Usage: ./cues <graph_path> <source_node> <p> <q> <simulations>\n");
	printf("See README.md for details.\n");
}

/**
 * Allocate device memoery and set it to the given value.
 * devicePointer: pointer to a device pointer
 * size: the size of the memory to allocate in bytes
 * setTo: the value used to set the allocated memory
 * returns: the CUDA error state after the performed operations
 */
cudaError_t cudaGetSpaceAndSet(void **devicePointer, size_t size, int setTo) {
	cudaError_t status = cudaMalloc(devicePointer, size);

	if(status != cudaSuccess) {
		return status;
	}

	return cudaMemset(*devicePointer, setTo, size);
}

/**
 * Creates a device copy of the given array.
 * src: the source array in host memory
 * dst: pointer to a device pointer where the copy will be placed
 * size: the size of the source array in bytes
 * returns: the CUDA error state after the performed operations
 */
cudaError_t cudaGetDeviceCopy(void *src, void** dst, size_t size) {
	cudaError_t status = cudaMalloc(dst, size);

	if(status != cudaSuccess) {
		return status;
	}

    return cudaMemcpy(*dst, src, size, cudaMemcpyHostToDevice);
}

#endif
