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

#endif /* UTILS_CUH_ */
