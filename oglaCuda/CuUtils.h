/* 
 * File:   CuUtils.h
 * Author: mmatula
 *
 * Created on October 26, 2014, 2:33 PM
 */

#ifndef CU_UTILS_H
#define	CU_UTILS_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h" 
#include <stdio.h>

#include "Types.h"

#ifdef DEBUG
#define CUDA_DEBUG() \
{ \
    const uintt tx = blockIdx.x * blockDim.x + threadIdx.x;\
    const uintt ty = blockIdx.y * blockDim.y + threadIdx.y;\
    if (tx == 0 && ty == 0) { \
        printf("%s %s %d \n", __FUNCTION__, __FILE__, __LINE__);\
    }\
}
#else
#define CUDA_DEBUG()
#endif


#endif	/* CUUTILS_H */

