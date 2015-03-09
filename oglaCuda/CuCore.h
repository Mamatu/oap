/* 
 * File:   CuCore.h
 * Author: mmatula
 *
 * Created on March 3, 2015, 10:35 PM
 */

#ifndef CU_CORE_H
#define	CU_CORE_H

#include <cuda.h>

#ifdef __CUDACC__
#define __hostdeviceinline__ __device__ __forceinline__
#define __hostdevice__ __device__
#else
#define __hostdeviceinline__
#define __hostdevice__
#endif

#endif	/* CUCORE_H */

