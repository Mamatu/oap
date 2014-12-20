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
 * @param params 9 matrices, where 1. H matrix 2. Q matrix, 3. R matrix. Rest is auxaliary matrix.
 * @param kernel
 * @param image
 */
void CuProcedure_CalculateTriangularH(void** params, ::cuda::Kernel& kernel, void* image);

void CuProcedure_CalculateH(void** params, ::cuda::Kernel& kernel, void* image);

#endif	/* CUPROCEDURES_H */

