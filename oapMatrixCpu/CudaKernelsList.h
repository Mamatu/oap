/*
 * Copyright 2016 - 2021 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef OAP_CUDA_KERNELS_LIST_H
#define OAP_CUDA_KERNELS_LIST_H

#include "CuMatrixProcedures.h"

template<typename T>
T* getParam (const void* param)
{
  return *static_cast<T* const*> (param);
}

template<typename T>
T* getParam (const void** params, size_t index)
{
  return getParam<T> (params[index]);
}

template<typename T>
T* getParam (void* param)
{
  return *static_cast<T**> (param);
}

template<typename T>
T* getParam (void** params, size_t index)
{
  return getParam<T> (params[index]);
}

#define DEFINE_1M(arg_hostKernel, arg_cudaKernel)                 \
void arg_hostKernel (math::ComplexMatrix* output)                        \
{                                                                 \
  arg_cudaKernel (output);                                        \
}                                                                 \
                                                                  \
void proxy_##arg_hostKernel (const void** params)                       \
{                                                                 \
  math::ComplexMatrix* output = getParam<math::ComplexMatrix> (params[0]);      \
  arg_hostKernel (output);                                        \
}

#define DEFINE_1M_EX(arg_hostKernel, arg_cudaKernel)                          \
void arg_hostKernel (math::ComplexMatrix* output, uintt* ex)                         \
{                                                                             \
  arg_cudaKernel (output, ex);                                                \
}                                                                             \
                                                                              \
void proxy_##arg_hostKernel (const void** params)                                   \
{                                                                             \
  math::ComplexMatrix* output = getParam<math::ComplexMatrix> (params[0]);                  \
  uintt* ex = getParam<uintt> (params[1]);                                    \
  arg_hostKernel (output, ex);                                                \
}

#define DEFINE_2M(arg_hostKernel, arg_cudaKernel)                 \
void arg_hostKernel (math::ComplexMatrix* output, math::ComplexMatrix* param1)  \
{                                                                 \
  arg_cudaKernel (output, param1);                                \
}                                                                 \
                                                                  \
void proxy_##arg_hostKernel (const void** params)                       \
{                                                                 \
  math::ComplexMatrix* output = getParam<math::ComplexMatrix> (params[0]);      \
  math::ComplexMatrix* matrix = getParam<math::ComplexMatrix> (params[1]);      \
  arg_hostKernel (output, matrix);                                \
}

#define DEFINE_2M_EX(arg_hostKernel, arg_cudaKernel)                          \
void arg_hostKernel (math::ComplexMatrix* output, math::ComplexMatrix* param1, uintt* ex)   \
{                                                                             \
  arg_cudaKernel (output, param1, ex);                                        \
}                                                                             \
                                                                              \
void proxy_##arg_hostKernel (const void** params)                                   \
{                                                                             \
  math::ComplexMatrix* output = getParam<math::ComplexMatrix> (params[0]);                  \
  math::ComplexMatrix* matrix = getParam<math::ComplexMatrix> (params[1]);                  \
  uintt* ex = getParam<uintt> (params[2]);                                    \
  arg_hostKernel (output, matrix, ex);                                        \
}

#define DEFINE_3M(arg_hostKernel, arg_cudaKernel)                                       \
void arg_hostKernel (math::ComplexMatrix* output, math::ComplexMatrix* param1, math::ComplexMatrix* param2)  \
{                                                                                       \
  arg_cudaKernel (output, param1, param2);                                              \
}                                                                                       \
                                                                                        \
void proxy_##arg_hostKernel (const void** params)                                             \
{                                                                                       \
  math::ComplexMatrix* output = getParam<math::ComplexMatrix> (params[0]);                            \
  math::ComplexMatrix* matrix = getParam<math::ComplexMatrix> (params[1]);                            \
  math::ComplexMatrix* matrix1 = getParam<math::ComplexMatrix> (params[2]);                           \
  arg_hostKernel (output, matrix, matrix1);                                             \
}

#define DEFINE_3M_EX(arg_hostKernel, arg_cudaKernel)                                                \
void arg_hostKernel (math::ComplexMatrix* output, math::ComplexMatrix* param1, math::ComplexMatrix* param2, uintt* ex)   \
{                                                                                                   \
  arg_cudaKernel (output, param1, param2, ex);                                                      \
}                                                                                                   \
                                                                                                    \
void proxy_##arg_hostKernel (const void** params)                                                         \
{                                                                                                   \
  math::ComplexMatrix* output = getParam<math::ComplexMatrix> (params[0]);                                        \
  math::ComplexMatrix* matrix = getParam<math::ComplexMatrix> (params[1]);                                        \
  math::ComplexMatrix* matrix1 = getParam<math::ComplexMatrix> (params[2]);                                       \
  uintt* ex = getParam<uintt> (params[3]);                                                          \
  arg_hostKernel (output, matrix, matrix1, ex);                                                     \
}

void HOSTKernel_SumShared (floatt* rebuffer, floatt* imbuffer, math::ComplexMatrix* matrix)
{
  CUDA_sumShared (rebuffer, imbuffer, matrix);
}

void proxy_HOSTKernel_SumShared (const void** params)
{
  floatt* param1 = getParam<floatt> (params[0]);
  floatt* param2 = getParam<floatt> (params[1]);
  math::ComplexMatrix* param3 = getParam<math::ComplexMatrix> (params[2]);
  HOSTKernel_SumShared (param1, param2, param3);
}

DEFINE_1M(HOSTKernel_SetIdentity, CUDA_setIdentityMatrix)

DEFINE_3M(HOSTKernel_CrossEntropy, CUDA_crossEntropy)
DEFINE_3M(HOSTKernel_specific_DotProduct, CUDA_specific_dotProduct)
DEFINE_3M(HOSTKernel_DotProductShared, CUDAKernel_dotProductShared)
DEFINE_3M(HOSTKernel_Convolve, CudaKernel_convolve)
DEFINE_2M_EX(HOSTKernel_PoolAverage, CudaKernel_poolAverage)

DEFINE_2M(HOSTKernel_Tanh, cuda_tanh)
DEFINE_2M(HOSTKernel_DTanh, cuda_dtanh)
DEFINE_2M(HOSTKernel_Sigmoid, cuda_sigmoid)
DEFINE_2M(HOSTKernel_Sin, cuda_sin)

DEFINE_2M(HOSTKernel_Relu, cuda_relu)
DEFINE_2M(HOSTKernel_DRelu, cuda_drelu)
DEFINE_2M(HOSTKernel_PRelu, cuda_prelu)
DEFINE_2M(HOSTKernel_DPRelu, cuda_dprelu)

DEFINE_2M_EX(HOSTKernel_ReluDim, cuda_reluDim)
DEFINE_2M_EX(HOSTKernel_DReluDim, cuda_dreluDim)
DEFINE_2M_EX(HOSTKernel_PReluDim, cuda_preluDim)
DEFINE_2M_EX(HOSTKernel_DPReluDim, cuda_dpreluDim)

DEFINE_2M_EX(HOSTKernel_ReluDimPeriodic, cuda_reluDimPeriodic)
DEFINE_2M_EX(HOSTKernel_DReluDimPeriodic, cuda_dreluDimPeriodic)
DEFINE_2M_EX(HOSTKernel_PReluDimPeriodic, cuda_preluDimPeriodic)
DEFINE_2M_EX(HOSTKernel_DPReluDimPeriodic, cuda_dpreluDimPeriodic)

DEFINE_2M_EX(HOSTKernel_TanhDim, cuda_tanhDim)
DEFINE_2M_EX(HOSTKernel_SigmoidDim, cuda_sigmoidDim)
DEFINE_2M_EX(HOSTKernel_SinDim, cuda_sinDim)

DEFINE_2M_EX(HOSTKernel_TanhDimPeriodic, cuda_tanhDimPeriodic)
DEFINE_2M_EX(HOSTKernel_SigmoidDimPeriodic, cuda_sigmoidDimPeriodic)
DEFINE_2M_EX(HOSTKernel_SinDimPeriodic, cuda_sinDimPeriodic)

DEFINE_3M_EX(HOSTKernel_DotProductDim, CUDA_dotProductDim)
DEFINE_3M(HOSTKernel_DotProductPeriodic, CUDA_dotProductPeriodic)
DEFINE_3M_EX(HOSTKernel_DotProductDimPeriodic, CUDA_dotProductDimPeriodic)

DEFINE_3M_EX(HOSTKernel_TensorProductDim, CUDA_tensorProductDim)

void HOSTKernel_QRHT (math::ComplexMatrix* Q, math::ComplexMatrix* R, math::ComplexMatrix* A, math::ComplexMatrix* V, math::ComplexMatrix* VT, math::ComplexMatrix* P, math::ComplexMatrix* VVT)
{
  CudaKernel_QRHT(Q, R, A, V, VT, P, VVT);
}

void proxy_HOSTKernel_QRHT (const void** params)
{
  size_t i = 0;
  math::ComplexMatrix* Q = getParam<math::ComplexMatrix> (params, i++);
  math::ComplexMatrix* R = getParam<math::ComplexMatrix> (params, i++);
  math::ComplexMatrix* A = getParam<math::ComplexMatrix> (params, i++);
  math::ComplexMatrix* V = getParam<math::ComplexMatrix> (params, i++);
  math::ComplexMatrix* VT = getParam<math::ComplexMatrix> (params, i++);
  math::ComplexMatrix* P = getParam<math::ComplexMatrix> (params, i++);
  math::ComplexMatrix* VVT = getParam<math::ComplexMatrix> (params, i++);

  HOSTKernel_QRHT (Q, R, A, V, VT, P, VVT);
}

void HOSTKernel_setVector (math::ComplexMatrix* V, uintt column,
                           math::ComplexMatrix* v, uintt length)
{
  CUDAKernel_setVector (V, column, v, length);
}

void proxy_HOSTKernel_setVector (const void** params)
{
  math::ComplexMatrix* V = getParam<math::ComplexMatrix> (params[0]);
  uintt column = *static_cast<const uintt*> (params[1]);
  math::ComplexMatrix* v = getParam<math::ComplexMatrix> (params[2]);
  uintt length = *static_cast<const uintt*> (params[3]);

  HOSTKernel_setVector (V, column, v, length);
}

void HOSTKernel_getVector (math::ComplexMatrix* v, uintt length,
                                 math::ComplexMatrix* V, uintt column)
{
  CUDAKernel_getVector (v, length, V, column);
}

void proxy_HOSTKernel_getVector (const void** params)
{
  math::ComplexMatrix* v = getParam<math::ComplexMatrix> (params[0]);
  uintt length = *static_cast<const uintt*> (params[1]);
  math::ComplexMatrix* V = getParam<math::ComplexMatrix> (params[2]);
  uintt column = *static_cast<const uintt*> (params[3]);

  HOSTKernel_getVector (v, length, V, column);
}

void HOSTKernel_PHadamardProduct (math::ComplexMatrix* outputs, math::ComplexMatrix* params1, math::ComplexMatrix* params2)
{
  CUDA_phadamardProduct (outputs, params1, params2);
}

void proxy_HOSTKernel_PHadamardProduct (const void** params)
{
  math::ComplexMatrix* outputs = getParam<math::ComplexMatrix> (params[0]);
  math::ComplexMatrix* params1 = getParam<math::ComplexMatrix> (params[1]);
  math::ComplexMatrix* params2 = getParam<math::ComplexMatrix> (params[2]);

  HOSTKernel_PHadamardProduct (outputs, params1, params2);
}

void HOSTKernel_AddMatrices (math::ComplexMatrix* outputs, math::ComplexMatrix* params1, math::ComplexMatrix* params2)
{
  CUDA_addMatrices (outputs, params1, params2);
}

void proxy_HOSTKernel_AddMatrices (const void** params)
{
  math::ComplexMatrix* outputs = getParam<math::ComplexMatrix> (params[0]);
  math::ComplexMatrix* params1 = getParam<math::ComplexMatrix> (params[1]);
  math::ComplexMatrix* params2 = getParam<math::ComplexMatrix> (params[2]);

  HOSTKernel_AddMatrices (outputs, params1, params2);
}

void HOSTKernel_MultiplyConstantRe (math::ComplexMatrix* outputs, math::ComplexMatrix* params1, floatt re)
{
  CUDA_multiplyConstantReMatrix (outputs, params1, re);
}

void proxy_HOSTKernel_MultiplyConstantRe (const void** params)
{
  math::ComplexMatrix* outputs = getParam<math::ComplexMatrix> (params[0]);
  math::ComplexMatrix* params1 = getParam<math::ComplexMatrix> (params[1]);
  floatt re = *static_cast<const floatt*> (params[2]);

  HOSTKernel_MultiplyConstantRe (outputs, params1, re);
}

void HOSTKernel_GenericApi_AddConst (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, floatt params2, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_AddConst (outputs, params1, params2, mapper);
}

void proxy_HOSTKernel_GenericApi_AddConst (const void** params)
{
  math::ComplexMatrix** outputs = getParam<math::ComplexMatrix*> (params[0]);
  math::ComplexMatrix* const* params1 = getParam<math::ComplexMatrix*> (params[1]);
  floatt params2 = *static_cast<const floatt*> (params[2]);
  oap::ThreadsMapperS* mapper = getParam<oap::ThreadsMapperS> (params[3]);

  HOSTKernel_GenericApi_AddConst (outputs, params1, params2, mapper);
}

void HOSTKernel_GenericApi_Add (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, math::ComplexMatrix* const* params2, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_Add (outputs, params1, params2, mapper);
}

void proxy_HOSTKernel_GenericApi_Add (const void** params)
{
  math::ComplexMatrix** outputs = getParam<math::ComplexMatrix*> (params[0]);
  math::ComplexMatrix* const* params1 = getParam<math::ComplexMatrix*> (params[1]);
  math::ComplexMatrix* const* params2 = getParam<math::ComplexMatrix*> (params[2]);
  oap::ThreadsMapperS* mapper = getParam<oap::ThreadsMapperS> (params[3]);

  HOSTKernel_GenericApi_Add (outputs, params1, params2, mapper);
}

void HOSTKernel_GenericApi_Subtract (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, math::ComplexMatrix* const* params2, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_Subtract (outputs, params1, params2, mapper);
}

void proxy_HOSTKernel_GenericApi_Subtract (const void** params)
{
  math::ComplexMatrix** outputs = getParam<math::ComplexMatrix*> (params[0]);
  math::ComplexMatrix* const* params1 = getParam<math::ComplexMatrix*> (params[1]);
  math::ComplexMatrix* const* params2 = getParam<math::ComplexMatrix*> (params[2]);
  oap::ThreadsMapperS* mapper = getParam<oap::ThreadsMapperS> (params[3]);

  HOSTKernel_GenericApi_Subtract (outputs, params1, params2, mapper);
}

void HOSTKernel_GenericApi_DotProduct (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, math::ComplexMatrix* const* params2, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_DotProduct (outputs, params1, params2, mapper);
}

void proxy_HOSTKernel_GenericApi_DotProduct (const void** params)
{
  math::ComplexMatrix** outputs = getParam<math::ComplexMatrix*> (params[0]);
  math::ComplexMatrix* const* params1 = getParam<math::ComplexMatrix*> (params[1]);
  math::ComplexMatrix* const* params2 = getParam<math::ComplexMatrix*> (params[2]);
  oap::ThreadsMapperS* mapper = getParam<oap::ThreadsMapperS> (params[3]);

  HOSTKernel_GenericApi_DotProduct (outputs, params1, params2, mapper);
}

void HOSTKernel_GenericApi_HadamardProduct (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, math::ComplexMatrix* const* params2, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_HadamardProduct (outputs, params1, params2, mapper);
}

void proxy_HOSTKernel_GenericApi_HadamardProduct (const void** params)
{
  math::ComplexMatrix** outputs = getParam<math::ComplexMatrix*> (params[0]);
  math::ComplexMatrix* const* params1 = getParam<math::ComplexMatrix*> (params[1]);
  math::ComplexMatrix* const* params2 = getParam<math::ComplexMatrix*> (params[2]);
  oap::ThreadsMapperS* mapper = getParam<oap::ThreadsMapperS> (params[3]);

  HOSTKernel_GenericApi_HadamardProduct (outputs, params1, params2, mapper);
}

void HOSTKernel_GenericApi_PHadamardProduct (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, math::ComplexMatrix* const* params2, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_PartialHadamardProduct (outputs, params1, params2, mapper);
}

void proxy_HOSTKernel_GenericApi_PHadamardProduct (const void** params)
{
  math::ComplexMatrix** outputs = getParam<math::ComplexMatrix*> (params[0]);
  math::ComplexMatrix* const* params1 = getParam<math::ComplexMatrix*> (params[1]);
  math::ComplexMatrix* const* params2 = getParam<math::ComplexMatrix*> (params[2]);
  oap::ThreadsMapperS* mapper = getParam<oap::ThreadsMapperS> (params[3]);

  HOSTKernel_GenericApi_PHadamardProduct (outputs, params1, params2, mapper);
}

void HOSTKernel_GenericApi_TensorProduct (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, math::ComplexMatrix* const* params2, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_TensorProduct (outputs, params1, params2, mapper);
}

void proxy_HOSTKernel_GenericApi_TensorProduct (const void** params)
{
  math::ComplexMatrix** outputs = getParam<math::ComplexMatrix*> (params[0]);
  math::ComplexMatrix* const* params1 = getParam<math::ComplexMatrix*> (params[1]);
  math::ComplexMatrix* const* params2 = getParam<math::ComplexMatrix*> (params[2]);
  oap::ThreadsMapperS* mapper = getParam<oap::ThreadsMapperS> (params[3]);

  HOSTKernel_GenericApi_TensorProduct (outputs, params1, params2, mapper);
}

void HOSTKernel_GenericApi_Transpose (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, oap::ThreadsMapperS* mapper)
{
  CUDA_GenericApi_Transpose (outputs, params1, mapper);
}

void proxy_HOSTKernel_GenericApi_Transpose (const void** params)
{
  math::ComplexMatrix** outputs = getParam<math::ComplexMatrix*> (params[0]);
  math::ComplexMatrix* const* params1 = getParam<math::ComplexMatrix*> (params[1]);
  oap::ThreadsMapperS* mapper = getParam<oap::ThreadsMapperS> (params[2]);

  HOSTKernel_GenericApi_Transpose (outputs, params1, mapper);
}

void HOSTKernel_GenericApi_Sigmoid (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, oap::ThreadsMapperS* mapper)
{
  cuda_genericApi_sigmoid (outputs, params1, mapper);
}

void proxy_HOSTKernel_GenericApi_Sigmoid (const void** params)
{
  math::ComplexMatrix** outputs = getParam<math::ComplexMatrix*> (params[0]);
  math::ComplexMatrix* const* params1 = getParam<math::ComplexMatrix*> (params[1]);
  oap::ThreadsMapperS* mapper = getParam<oap::ThreadsMapperS> (params[2]);

  HOSTKernel_GenericApi_Sigmoid (outputs, params1, mapper);
}

void HOSTKernel_GenericApi_Tanh (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, oap::ThreadsMapperS* mapper)
{
  cuda_genericApi_tanh (outputs, params1, mapper);
}

void proxy_HOSTKernel_GenericApi_Tanh (const void** params)
{
  math::ComplexMatrix** outputs = getParam<math::ComplexMatrix*> (params[0]);
  math::ComplexMatrix* const* params1 = getParam<math::ComplexMatrix*> (params[1]);
  oap::ThreadsMapperS* mapper = getParam<oap::ThreadsMapperS> (params[2]);

  HOSTKernel_GenericApi_Tanh (outputs, params1, mapper);
}

void HOSTKernel_GenericApi_DTanh (math::ComplexMatrix** outputs, math::ComplexMatrix* const* params1, oap::ThreadsMapperS* mapper)
{
  cuda_genericApi_dtanh (outputs, params1, mapper);
}

void proxy_HOSTKernel_GenericApi_DTanh (const void** params)
{
  math::ComplexMatrix** outputs = getParam<math::ComplexMatrix*> (params[0]);
  math::ComplexMatrix* const* params1 = getParam<math::ComplexMatrix*> (params[1]);
  oap::ThreadsMapperS* mapper = getParam<oap::ThreadsMapperS> (params[2]);

  HOSTKernel_GenericApi_DTanh (outputs, params1, mapper);
}

#endif
