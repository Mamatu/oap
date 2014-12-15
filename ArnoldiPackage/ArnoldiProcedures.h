/* 
 * File:   CuProcedures.h
 * Author: mmatula
 *
 * Created on August 17, 2014, 1:20 AM
 */

#ifndef OGLA_CU_ARNOLDIPROCEDURES_H
#define	OGLA_CU_ARNOLDIPROCEDURES_H

#include "KernelExecutor.h"

/**
 * 
 * @param params
 * @param kernel
 * @param image
 */
void CUDAKernel_CalculateHQ(void** params, ::cuda::Kernel& kernel, void* image);

#endif	/* CUPROCEDURES_H */

