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

#ifndef HOSTPROCEDURE_H
#define HOSTPROCEDURE_H

#include "Matrix.h"
#include "HostKernel.h"
#include "HostKernelExecutor.h"

#include "GenericProceduresApi.h"
#include "GenericProceduresNewApi.h"
#include "oapHostMatrixUtils.h"

class HostProcedures {
 public:
  HostProcedures(uint maxThreadsPerBlock = 1024);
  virtual ~HostProcedures();

  void setMaxThreadsPerBlock (uintt maxThreadsPerBlock);

  bool compare(math::Matrix* matrix1, math::Matrix* matrix2);

  bool isEqual(math::Matrix* matrix1, math::Matrix* matrix2);

  void subtract(math::Matrix* output, math::Matrix* matrix1,
                 math::Matrix* matrix2);

  void dotProduct (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2);
  void dotProductShared (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2);

  void dotProductPeriodic (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2);

  void dotProductDimPeriodic (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2, oap::generic::Dim32 dim, uintt periodicRows);

  void dotProductDimPeriodic (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2, oap::generic::Dim32 dim)
  {
    uintt periodicRows = oap::host::GetRows (matrix1);
    dotProductDimPeriodic (output, matrix1, matrix2, dim, periodicRows);
  }

  void dotProduct (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2, oap::generic::Dim32 dim);

  void dotProduct (math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2,
                   uintt outputDim[2], uintt params0Dim[2], uintt params1Dim[2])
  {
    oap::generic::Dim32 dim {{{outputDim[0], outputDim[1]}, {params0Dim[0], params0Dim[1]}, {params1Dim[0], params1Dim[1]}}};
    dotProduct (output, matrix1, matrix2, dim);
  }

  void transpose(math::Matrix* output, math::Matrix* matrix);

  void tanh (math::Matrix* output, math::Matrix* matrix);
  void sigmoid (math::Matrix* output, math::Matrix* matrix);
  void linear (math::Matrix* output, math::Matrix* matrix);
  void sin (math::Matrix* output, math::Matrix* matrix);
  void prelu (math::Matrix* output, math::Matrix* matrix);
  void relu (math::Matrix* output, math::Matrix* matrix);
  void softplus (math::Matrix* output, math::Matrix* matrix);

  void tanh (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim);
  void sigmoid (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim);
  void linear (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim);
  void sin (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim);
  void prelu (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim);
  void relu (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim);
  void softplus (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim);

  void tanh (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim);
  void sigmoid (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim);
  void linear (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim);
  void sin (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim);
  void prelu (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim);
  void relu (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim);
  void softplus (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim);

  void dprelu (math::Matrix* output, math::Matrix* matrix);
  void drelu (math::Matrix* output, math::Matrix* matrix);
  void dprelu (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim);
  void drelu (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim);
  void dprelu (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim);
  void drelu (math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim);

  void sum (floatt& reoutput, floatt& imoutput, math::Matrix* params0);

  void crossEntropy (math::Matrix* output, math::Matrix* params0, math::Matrix* params1);

  void tensorProduct (math::Matrix* matrix, math::Matrix* params0, math::Matrix* params1, oap::generic::Dim32 dim);

  inline void tensorProduct (math::Matrix* matrix, math::Matrix* params0, math::Matrix* params1, uintt dim1[2], uintt dim2[2], uintt dim3[2])
  {
    oap::generic::Dim32 dim {{{dim1[0], dim1[1]}, {dim2[0], dim2[1]}, {dim3[0], dim3[1]}}};
    tensorProduct (matrix, params0, params1, dim);
  }

  void QRHT (math::Matrix* Q, math::Matrix* R, math::Matrix* A, math::Matrix* V, math::Matrix* VT, math::Matrix* P, math::Matrix* VVT);
  void setIdentity (math::Matrix* matrix);

  void setVector (math::Matrix* V, uintt column, math::Matrix* v, uintt length);

  void getVector (math::Matrix* vector, uintt length, math::Matrix* matrix, uintt column);

  void getVector (math::Matrix* vector, math::Matrix* matrix, uintt column);

  void convolve (math::Matrix* output, const math::Matrix* matrix, const math::Matrix* kernel);
  void poolAverage (math::Matrix* output, const math::Matrix* matrix, const math::MatrixDim& kernel);


  template<typename Matrices>
  void v2_add (Matrices& output, const Matrices& params1, floatt value)
  {
    oap::generic::addConstant (output, params1, value, &m_kernel, oap::host::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_add (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    oap::generic::add (output, params1, params2, &m_kernel, oap::host::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_dotProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    oap::generic::dotProduct (output, params1, params2, &m_kernel, oap::host::CreateThreadsMapper, malloc, free, memcpy, oap::host::GetMatrixInfo);
  }

  template<typename Matrices>
  void v2_multiply (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    v2_dotProduct<Matrices> (output, params1, params2);
  }

  template<typename Matrices>
  void v2_hadamardProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    oap::generic::hadamardProduct (output, params1, params2, &m_kernel, oap::host::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_tensorProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    oap::generic::tensorProduct (output, params1, params2, &m_kernel, oap::host::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_sigmoid (Matrices& output, const Matrices& params1)
  {
    oap::generic::sigmoid (output, params1, &m_kernel, oap::host::CreateThreadsMapper, malloc, free, memcpy);
  }

 private:
  uint m_threads[2];
  uint m_blocks[2];

  uintt m_maxThreadsPerBlock;
  HostKernelExecutor m_kernel;

  void prepare(math::Matrix* matrix, HostKernel& hostKernel);
  void prepare(size_t w, size_t h, HostKernel& hostKernel);

  oap::generic::BasicMatrixApi<decltype(oap::host::GetMatrixInfo)> m_bmApi;

  uintt* createKernelArray (uintt* hostArray, size_t length)
  {
    return hostArray;
  }

  void _funcDim (const std::string& kname, math::Matrix* output, math::Matrix* matrix, oap::generic::Dim2 dim);
  void _funcDimPeriodic (const std::string& kname, math::Matrix* output, math::Matrix* matrix, oap::generic::Dim22 dim);
  std::function<uintt*(uintt*, uintt)> m_createKernelArray;
};

#endif  // HOSTCOMPAREPROCEDURE_H
