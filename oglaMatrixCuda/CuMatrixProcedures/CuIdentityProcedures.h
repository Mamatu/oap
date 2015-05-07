/* 
 * File:   CuIdentityProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:28 PM
 */

#ifndef CUIDENTITYPROCEDURES_H
#define	CUIDENTITYPROCEDURES_H

#include "CuCore.h"

__hostdevice__ void CUDA_SetIdentityReMatrix(math::Matrix* dst,
    uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + dst->columns * threadIndexY;
    floatt v = threadIndexX == threadIndexY ? 1 : 0;
    dst->reValues[index] = v;
    threads_sync();
}

__hostdevice__ void CUDA_SetIdentityImMatrix(math::Matrix* dst,
    uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + dst->columns * threadIndexY;
    floatt v = threadIndexX == threadIndexY ? 1 : 0;
    dst->imValues[index] = v;
    threads_sync();
}

__hostdevice__ void CUDA_SetIdentityMatrix(math::Matrix* dst,
    uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + dst->columns * threadIndexY;
    floatt v = threadIndexX == threadIndexY ? 1 : 0;
    dst->reValues[index] = v;
    if (NULL != dst->imValues) {
        dst->imValues[index] = 0;
    }
    threads_sync();
}

#endif	/* CUIDENTITYPROCEDURES_H */

