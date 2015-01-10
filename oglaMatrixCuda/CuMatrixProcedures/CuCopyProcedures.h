/* 
 * File:   CuCopyProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:11 PM
 */

#ifndef CUCOPYPROCEDURES_H
#define	CUCOPYPROCEDURES_H

extern "C" __device__ void CUDA_CopyReMatrix(
    math::Matrix* dst,
    math::Matrix* src,
    uintt threadIndexX,
    uintt threadIndexY) {
    dst->reValues[threadIndexX + dst->columns * threadIndexY] =
        src->reValues[threadIndexX + src->columns * threadIndexY];
    __syncthreads();
}

extern "C" __device__ void CUDA_CopyImMatrix(
    math::Matrix* dst,
    math::Matrix* src,
    uintt threadIndexX,
    uintt threadIndexY) {
    dst->imValues[threadIndexX + dst->columns * threadIndexY] =
        src->imValues[threadIndexX + src->columns * threadIndexY];
    __syncthreads();
}

extern "C" __device__ void CUDA_CopyMatrix(
    math::Matrix* dst,
    math::Matrix* src,
    uintt threadIndexX,
    uintt threadIndexY) {
    if (dst->reValues) {
        CUDA_CopyReMatrix(dst, src, threadIndexX, threadIndexY);
    }
    if (dst->imValues) {
        CUDA_CopyImMatrix(dst, src, threadIndexX, threadIndexY);
    }
}

#endif	/* CUCOPYPROCEDURES_H */

