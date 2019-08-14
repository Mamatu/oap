/*
 * Copyright 2016 - 2019 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef OAP_CU_CORE_H
#define OAP_CU_CORE_H

#ifdef OAP_CUDA_BUILD

#include <cuda.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#define __hostdeviceinline__ extern "C" __device__ __forceinline__
#define __hostdevice__ extern "C" __device__

#define threads_sync() __syncthreads()

#define HOST_INIT()

#define HOST_INIT_SHARED(type, buffer) extern __shared__ type oap_shared_buffer[]; buffer = oap_shared_buffer;

#define HOST_CODE(code)

#else

#include "Dim3.h"
#include <pthread.h>

#define __hostdeviceinline__ __inline__
#define __hostdevice__ __inline__
#define __shared__

#define HOST_INIT()                                   \
  ThreadIdx& ti = ThreadIdx::m_threadIdxs[pthread_self()]; \
  uint3 threadIdx = ti.getThreadIdx();                     \
  dim3 blockIdx = ti.getBlockIdx();                        \
  dim3 blockDim = ti.getBlockDim();                        \
  dim3 gridDim = ti.getGridDim();

#define HOST_INIT_SHARED(type, buffer)  ThreadIdx& his_ti = ThreadIdx::m_threadIdxs[pthread_self()]; buffer = static_cast<type*>(his_ti.getSharedBuffer());

#define HOST_CODE(code) code

#define threads_sync() ThreadIdx::wait();

#endif

#define THREAD_INDEX_X_INIT() uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;

#define THREAD_INDEX_Y_INIT() uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;

#define THREAD_INDICES_INIT() \
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x; \
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;

#endif /* CUCORE_H */
