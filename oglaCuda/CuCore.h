/*
 * File:   CuCore.h
 * Author: mmatula
 *
 * Created on March 3, 2015, 10:35 PM
 */

#ifndef CU_CORE_H
#define CU_CORE_H

#include <cuda.h>

#ifdef CUDATEST

#include "Dim3.h"
#include <pthread.h>

#define __hostdeviceinline__ __inline__
#define __hostdevice__ __inline__
#define __shared__

#define CUDA_TEST_INIT() uint3 threadIdx = ThreadIdx::m_threadIdxs[pthread_self()].getThreadIdx();

#define CUDA_TEST_CODE(code) code

#define threads_sync() ThreadIdx::wait();

#else

#define __hostdeviceinline__ extern "C" __device__ __forceinline__
#define __hostdevice__ extern "C" __device__

#define threads_sync() __syncthreads()

#define CUDA_TEST_INIT()

#define CUDA_TEST_CODE(code)

#endif
#endif /* CUCORE_H */
