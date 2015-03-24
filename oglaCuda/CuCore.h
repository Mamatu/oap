/* 
 * File:   CuCore.h
 * Author: mmatula
 *
 * Created on March 3, 2015, 10:35 PM
 */

#ifndef CU_CORE_H
#define	CU_CORE_H

#include <cuda.h>

#ifndef CUDATEST
#define __hostdeviceinline__ extern "C" __device__ __forceinline__
#define __hostdevice__ extern "C" __device__
#else

#include "Dim3.h"

#define __hostdeviceinline__ __inline__
#define __hostdevice__ __inline__
#endif

#endif	/* CUCORE_H */

