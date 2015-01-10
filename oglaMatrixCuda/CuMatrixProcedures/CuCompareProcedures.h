/* 
 * File:   CuCompareProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:08 PM
 */

#ifndef CUCOMPAREPROCEDURES_H
#define	CUCOMPAREPROCEDURES_H


#include <cuda.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "CuMatrixUtils.h"
#include <stdio.h>
#include "Matrix.h"
#include "MatrixEx.h"

#define cuda_compare_re(buffer, m1, m2)\
uintt index = tindex * 2;\
uintt c = length & 1;\
if (tindex < length / 2) {\
    buffer[tindex] = m1->reValues[index] == m2->reValues[index];\
    buffer[tindex] += m1->reValues[index + 1] == m2->reValues[index + 1];\
    if (c == 1 && tindex == length - 3) {buffer[tindex] += m1->reValues[index + 2] == m2->reValues[index + 2];}\
}\
length = length / 2;

#define cuda_compare_real(buffer, m1, m2)\
uintt index = tindex * 2;\
uintt c = length & 1;\
if (tindex < length / 2) {\
    buffer[tindex] = m1->reValues[index] == m2->reValues[index];\
    buffer[tindex] += m1->imValues[index] == m2->imValues[index];\
    buffer[tindex] += m1->reValues[index + 1] == m2->reValues[index + 1];\
    buffer[tindex] += m1->imValues[index + 1] == m2->imValues[index + 1];\
    if (c == 1 && tindex == length - 3) {\
        buffer[tindex] += m1->reValues[index + 2] == m2->reValues[index + 2];\
        buffer[tindex] += m1->imValues[index + 2] == m2->imValues[index + 2];\
    }\
}\
length = length / 2;

#define cuda_compare_im(buffer, m1, m2)\
uintt index = tindex * 2;\
uintt c = length & 1;\
if (tindex < length / 2) {\
    buffer[tindex] += m1->imValues[index] == m2->imValues[index];\
    buffer[tindex] += m1->imValues[index + 1] == m2->imValues[index + 1];\
    if (c == 1 && tindex == length - 3) {\
        buffer[tindex] += m1->imValues[index + 2] == m2->imValues[index + 2];\
    }\
}\
length = length / 2;


#define cuda_compare_step_2(buffer)\
uintt index = tindex * 2;\
uintt c = length & 1;\
if (tindex < length / 2) {\
    buffer[tindex] += buffer[index + 1];\
    if (c == 1 && index == length - 3) {buffer[tindex] += buffer[index + 2];}\
}\
length = length / 2;

extern "C" __device__ void CUDA_compareRealMatrix(
    uintt& sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer,
    uintt tx, uintt ty) {
    uintt tindex = ty * matrix1->columns + tx;
    uintt length = matrix1->columns * matrix1->rows;
    cuda_compare_real(buffer, matrix1, matrix2);
    __syncthreads();
    do {
        cuda_compare_step_2(buffer);
        __syncthreads();
    } while (length > 1);
    sum = buffer[0] / 2;
}

extern "C" __device__ void CUDA_compareImMatrix(
    uintt& sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer,
    uintt tx, uintt ty) {
    uintt tindex = ty * matrix1->columns + tx;
    uintt length = matrix1->columns * matrix1->rows;
    cuda_compare_im(buffer, matrix1, matrix2);
    __syncthreads();
    do {
        cuda_compare_step_2(buffer);
        __syncthreads();
    } while (length > 1);
    sum = buffer[0];
}

extern "C" __device__ void CUDA_compareReMatrix(
    uintt& sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer,
    uintt tx, uintt ty) {
    uintt tindex = ty * matrix1->columns + tx;
    uintt length = matrix1->columns * matrix1->rows;
    cuda_compare_re(buffer, matrix1, matrix2);
    __syncthreads();
    do {
        cuda_compare_step_2(buffer);
        __syncthreads();
    } while (length > 1);
    sum = buffer[0];
}

extern "C" __device__ void CUDA_compare(
    uintt& sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer,
    uintt tx, uintt ty) {
    bool isre = matrix1->reValues != NULL;
    bool isim = matrix1->imValues != NULL;
    if (isre && isim) {
        CUDA_compareRealMatrix(sum, matrix1, matrix2, buffer, tx, ty);
    } else if (isre) {
        CUDA_compareReMatrix(sum, matrix1, matrix2, buffer, tx, ty);
    } else if (isim) {
        CUDA_compareImMatrix(sum, matrix1, matrix2, buffer, tx, ty);
    }
}


#endif	/* CUCOMPAREPROCEDURES_H */

