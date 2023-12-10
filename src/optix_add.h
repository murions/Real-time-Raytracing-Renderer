#ifndef OPTIX_ADD_H
#define OPTIX_ADD_H

#define _CRT_SECURE_NO_WARNINGS

#include <optix.h>
#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <stdlib.h>

#define OPTIX_CHECK( call )\
{\
	OptixResult res = call;\
	if(res != OPTIX_SUCCESS)\
	{\
		fprintf(stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__);\
		exit(2);\
	}\
}
#define CUDA_CHECK(call)\
{\
	cudaError_t rc = cuda##call;\
	if(rc != cudaSuccess){\
		std::stringstream txt;\
		cudaError_t err = rc;\
		txt << "CUDA Error " << cudaGetErrorName(err)\
			<< " (" << cudaGetErrorString(err) << ")";\
		throw std::runtime_error(txt.str());\
	}\
}
#define CUDA_CHECK_NOEXCEPT(call)\
{\
	cuda##call;\
}
#define CUDA_SYNC_CHECK()\
{\
	cudaDeviceSynchronize();\
	cudaError_t error = cudaGetLastError();\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error));\
		exit(2);\
	}\
}

#endif // OPTEX_ADD_H