/* 
 * File:   CuCopyProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:11 PM
 */

#ifndef CUCOPYPROCEDURES_H
#define	CUCOPYPROCEDURES_H

#include "CuCore.h"

__hostdevice__ void CUDA_copyReMatrix(
    math::Matrix* dst,
    math::Matrix* src,
    uintt threadIndexX,
    uintt threadIndexY) {
    CUDA_TEST_INIT();
    dst->reValues[threadIndexX + dst->columns * threadIndexY] =
        src->reValues[threadIndexX + src->columns * threadIndexY];
    threads_sync();
}

__hostdevice__ void CUDA_copyImMatrix(
    math::Matrix* dst,
    math::Matrix* src,
    uintt threadIndexX,
    uintt threadIndexY) {
    CUDA_TEST_INIT();
    dst->imValues[threadIndexX + dst->columns * threadIndexY] =
        src->imValues[threadIndexX + src->columns * threadIndexY];
    threads_sync();
}

__hostdevice__ void CUDA_copyMatrix(
    math::Matrix* dst,
    math::Matrix* src,
    uintt threadIndexX,
    uintt threadIndexY) {
    CUDA_TEST_INIT();
    if (dst->reValues) {
        CUDA_copyReMatrix(dst, src, threadIndexX, threadIndexY);
    }
    if (dst->imValues) {
        CUDA_copyImMatrix(dst, src, threadIndexX, threadIndexY);
    }
}

#endif	/* CUCOPYPROCEDURES_H */

