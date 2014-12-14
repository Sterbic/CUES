/**
 * Utilities for the CUES project.
 * This software contains source code provided by NVIDIA Corporation.
 * Please refer to the NVIDIA end user license agreement (EULA) for details.
 */

#ifndef UTILS_CUH_
#define UTILS_CUH_

#include <cstdio>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand.h>

/**
 * Standard CUDA error check macro. Will exit the program on detected error.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	}}

/**
 * Macro for safe curand calls. Will exit the program on detected error.
 */
#define CURAND_CHECK_RETURN(value) {										\
	curandStatus_t curandResult = value;									\
	if (curandResult != CURAND_STATUS_SUCCESS) {							\
		fprintf(stderr, "Error %d in curand call at line %d in file %s\n",	\
				curandResult, __LINE__, __FILE__);							\
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
 * Converts the given device computing capability version to the number of
 * cores per SM present in the device.
 * major: the major version of the device
 * minor: the minor version of the device
 * returns: the number of cores per SM
 */
int convertSMVersion2Cores(int major, int minor) {
    typedef struct {
        int SM;
        int cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        {   -1, -1 }
    };

    int index = 0;

    while(nGpuArchCoresPerSM[index].SM != -1) {
        if(nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].cores;
        }

        index++;
    }

    return nGpuArchCoresPerSM[index-1].cores;
}

/**
 * Returns the device ID of the device with highest Gflops installed on the
 * current machine with computing capabilities 3.0 or greater.
 * returns: best device ID
 */
int cudaGetMaxGflopsDeviceID() {
    int currentDevice = 0;
    int smPerMultiprocessor = 0;
    int bestDeviceID = 0;
    int deviceCount = 0;
    int bestSMArchitecture = 0;
    int prohibitedDevices = 0;

    unsigned long long maxPerformance = 0;
    cudaDeviceProp deviceProperties;

    CUDA_CHECK_RETURN(cudaGetDeviceCount(&deviceCount));
    exitIf(deviceCount == 0, "No CUDA capable device detected.");

    while (currentDevice < deviceCount) {
    	 CUDA_CHECK_RETURN(cudaGetDeviceProperties(
    			 &deviceProperties, currentDevice));

        if(deviceProperties.computeMode != cudaComputeModeProhibited) {
            if(deviceProperties.major > 0 && deviceProperties.major < 9999) {
                bestSMArchitecture = max(bestSMArchitecture, deviceProperties.major);
            }
        } else {
            prohibitedDevices++;
        }

        currentDevice++;
    }

    exitIf(prohibitedDevices == deviceCount,
    		"No CUDA capable devices currently available.");
    exitIf(bestSMArchitecture < 3,
    		"No CUDA device with computing capabilities 3.0 or greater.");

    currentDevice = 0;

    while(currentDevice < deviceCount) {
    	CUDA_CHECK_RETURN(cudaGetDeviceProperties(
    			&deviceProperties, currentDevice));

        if(deviceProperties.computeMode != cudaComputeModeProhibited) {
        	int major = deviceProperties.major;
        	int minor = deviceProperties.minor;

            if (major == 9999 && minor == 9999) {
                smPerMultiprocessor = 1;
            } else {
                smPerMultiprocessor = convertSMVersion2Cores(major, minor);
            }

            unsigned long long perfromance = (unsigned long long)
            		deviceProperties.multiProcessorCount * smPerMultiprocessor
            		* deviceProperties.clockRate;

            if(perfromance > maxPerformance && major == bestSMArchitecture) {
				maxPerformance = perfromance;
				bestDeviceID = currentDevice;
            }
        }

        currentDevice++;
    }

    return bestDeviceID;
}

/**
 * Creates a device copy of the given host array.
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

/**
 * Creates a host copy of the given device array.
 * src: the source array in device memory
 * dst: pointer to a host pointer where the copy will be placed
 * size: the size of the source array in bytes
 * returns: the CUDA error state after the performed operations
 */
cudaError_t cudaGetHostCopy(void *src, void** dst, size_t size) {
	*dst = malloc(size);
	exitIf(*dst == NULL, "Error allocating host copy array.");
	return cudaMemcpy(*dst, src, size, cudaMemcpyDeviceToHost);
}

/**
 * Prints the given float array to stdout.
 * array: the source array
 * size: the size of the array
 * onHost: true if the array is in host memory, false if it's on the device
 */
void printFloatArray(float *array, int size, bool onHost) {
	float *printArray = array;

	if(!onHost) {
		CUDA_CHECK_RETURN(cudaGetHostCopy(
				(void *) array,
				(void **) &printArray,
				size * sizeof(float)
				));
	}

	for(int i = 0; i < size; i++) {
		printf("%.5f", printArray[i]);
		printf(i == size - 1 ? "\n" : ", ");
	}
}

#endif
