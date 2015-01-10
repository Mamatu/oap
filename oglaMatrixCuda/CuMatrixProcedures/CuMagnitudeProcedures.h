/* 
 * File:   CuMagnitudeProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:27 PM
 */

#ifndef CUMAGNITUDEPROCEDURES_H
#define	CUMAGNITUDEPROCEDURES_H

#define cuda_magnite_step_real(buffer, re, im)\
uintt index = tindex * 2;\
uintt c = length & 1;\
if (tindex < length / 2) {\
    buffer[tindex] =\
    + re[index] * re[index]\
    + im[index] * im[index];\
    + re[index + 1] * re[index + 1]\
    + im[index + 1] * im[index + 1];\
    if (c == 1 && tindex == length - 3) {\
        buffer[tindex] += re[index + 2] * re[index + 2] + im[index + 2] * im[index + 2];\
    }\
    length = length / 2;\
}

#define cuda_magnite_step(buffer, values)\
uintt index = tindex * 2;\
uintt c = length & 1;\
if (tindex < length / 2) {\
    buffer[tindex] = values[index] * values[index] + values[index + 1] * values[index + 1];\
    if (c == 1 && tindex == length - 3) {buffer[tindex] += values[index + 2] * values[index + 2];}\
}\
length = length / 2;

#define cuda_magnite_step_2(buffer)\
uintt index = tindex * 2;\
uintt c = length & 1;\
if (tindex < length / 2) {\
    buffer[tindex] = buffer[index] + buffer[index + 1];\
    if (c == 1 && index == length - 3) {buffer[tindex] += buffer[index + 2];}\
}\
length = length / 2;

extern "C" __device__ void CUDA_magnitudeReal(floatt& value, math::Matrix* src,
    floatt* buffer,
    uintt tx, uintt ty) {
    if (tx == 0 && ty == 0) {
        value = 0;
    }
    uintt tindex = tx + src->columns * ty;
    uintt length = src->columns * src->rows;
    cuda_magnite_step_real(buffer, src->reValues, src->imValues);
    __syncthreads();
    do {
        cuda_magnite_step_2(buffer);
        __syncthreads();
    } while (length > 1);
    value = sqrt(buffer[0]);
}

extern "C" __device__ void CUDA_magnitudeRe(floatt& value, math::Matrix* src,
    floatt* buffer,
    uintt tx, uintt ty) {
    if (tx == 0 && ty == 0) {
        value = 0;
    }
    uintt length = src->columns * src->rows;
    uintt tindex = tx + src->columns * ty;
    cuda_magnite_step(buffer, src->reValues);
    __syncthreads();
    do {
        cuda_magnite_step_2(buffer);
        __syncthreads();
    } while (length > 1);
    value = sqrt(buffer[0]);
}

extern "C" __device__ void CUDA_magnitudeIm(floatt& value, math::Matrix* src,
    floatt* buffer,
    uintt tx, uintt ty) {
    if (tx == 0 && ty == 0) {
        value = 0;
    }
    uintt length = src->columns * src->rows;
    uintt tindex = tx + src->columns * ty;
    cuda_magnite_step(buffer, src->imValues);
    __syncthreads();
    do {
        cuda_magnite_step_2(buffer);
        __syncthreads();
    } while (length > 1);
    value = sqrt(buffer[0]);
}

extern "C" __device__ void CUDA_magnitudeRealOpt(floatt& value, math::Matrix* src,
    uintt tx, uintt ty) {
    if (tx == 0 && ty == 0) {
        value = 0;
    }
    extern __shared__ floatt buffer[];
    uintt length = src->columns * src->rows;
    uintt tindex = tx + src->columns * ty;
    cuda_magnite_step_real(buffer, src->reValues, src->imValues);
    while (length > 1) {
        cuda_magnite_step_2(buffer);
    }
    value = sqrt(buffer[0]);
}

extern "C" __device__ void CUDA_magnitudeReOpt(floatt& value, math::Matrix* src,
    uintt tx, uintt ty) {
    if (tx == 0 && ty == 0) {
        value = 0;
    }
    extern __shared__ floatt buffer[];
    uintt length = src->columns * src->rows;
    uintt tindex = tx + src->columns * ty;
    cuda_magnite_step(buffer, src->reValues);
    while (length > 1) {
        cuda_magnite_step_2(buffer);
    }
    value = sqrt(buffer[0]);
}

extern "C" __device__ void CUDA_magnitudeImOpt(floatt& value, math::Matrix* src,
    uintt tx, uintt ty) {
    if (tx == 0 && ty == 0) {
        value = 0;
    }
    extern __shared__ floatt buffer[];
    uintt length = src->columns * src->rows;
    uintt tindex = tx + src->columns * ty;
    cuda_magnite_step(buffer, src->imValues);
    while (length > 1) {
        cuda_magnite_step_2(buffer);
    }
    value = sqrt(buffer[0]);
}

extern "C" __device__ void CUDA_magnitudeOpt(floatt& value, math::Matrix* src,
    uintt tx, uintt ty) {
    if (tx == 0 && ty == 0) {
        value = 0;
    }
    bool isre = src->reValues != NULL;
    bool isim = src->imValues != NULL;
    if (isre && isim) {
        CUDA_magnitudeRealOpt(value, src, tx, ty);
    } else if (isre) {
        CUDA_magnitudeReOpt(value, src, tx, ty);
    } else if (isim) {
        CUDA_magnitudeImOpt(value, src, tx, ty);
    }
}

extern "C" __device__ void CUDA_magnitude(floatt& value, math::Matrix* src,
    floatt* buffer,
    uintt tx, uintt ty) {
    bool isre = src->reValues != NULL;
    bool isim = src->imValues != NULL;
    if (isre && isim) {
        CUDA_magnitudeReal(value, src, buffer, tx, ty);
    } else if (isre) {
        CUDA_magnitudeRe(value, src, buffer, tx, ty);
    } else if (isim) {
        CUDA_magnitudeIm(value, src, buffer, tx, ty);
    }
}

#endif	/* CUMAGNITUDEPROCEDURES_H */

