/*
 * File:   CuCore.h
 * Author: mmatula
 *
 * Created on March 3, 2015, 10:35 PM
 */

#ifndef CU_CORE_H
#define CU_CORE_H

#ifdef CUDA

#include <cuda.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#define __hostdeviceinline__ extern "C" __device__ __forceinline__
#define __hostdevice__ extern "C" __device__

#define threads_sync() __syncthreads()

#define CUDA_TEST_INIT()

#define CUDA_TEST_CODE(code)

#else

#include "Dim3.h"
#include <pthread.h>

#define __hostdeviceinline__ __inline__
#define __hostdevice__ __inline__
#define __shared__

#define CUDA_TEST_INIT()                                   \
  ThreadIdx& ti = ThreadIdx::m_threadIdxs[pthread_self()]; \
  uint3 threadIdx = ti.getThreadIdx();                     \
  dim3 blockIdx = ti.getBlockIdx();                        \
  dim3 blockDim = ti.getBlockDim();                        \
  dim3 gridDim = ti.getGridDim();

#define CUDA_TEST_CODE(code) code

#define threads_sync() ThreadIdx::wait();

#endif
#endif /* CUCORE_H */
