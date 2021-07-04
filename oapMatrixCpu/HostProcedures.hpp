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

#include "Matrix.hpp"
#include "HostKernel.hpp"
#include "HostKernelExecutor.hpp"

#include "GenericProceduresApi.hpp"
#include "GenericProceduresNewApi.hpp"
#include "oapHostComplexMatrixApi.hpp"

#include "oapProcedures.hpp"
#include "HostBuffer.hpp"

namespace oap
{

class HostProcedures : public oap::generic::SingleMatrixProcedures {
 private:
  oap::host::HostBuffer<floatt> m_dsumsReBuffer;
  oap::host::HostBuffer<floatt> m_dsumsImBuffer;
  oap::host::HostBuffer<floatt> m_hsumsReBuffer;
  oap::host::HostBuffer<floatt> m_hsumsImBuffer;
 
 public:
  HostProcedures(uint maxThreadsPerBlock = 1024);
  virtual ~HostProcedures();

  void setMaxThreadsPerBlock (uintt maxThreadsPerBlock);

  bool compare(math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2);

  bool isEqual(math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2);

  void subtract (math::ComplexMatrix* output, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2) override;

  void dotProduct (math::ComplexMatrix* output, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2);
  void dotProductShared (math::ComplexMatrix* output, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2);

  void dotProductPeriodic (math::ComplexMatrix* output, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2);

  void dotProductDimPeriodic (math::ComplexMatrix* output, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2, oap::generic::Dim32 dim, uintt periodicRows) override;

  void dotProductDimPeriodic (math::ComplexMatrix* output, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2, oap::generic::Dim32 dim)
  {
    uintt periodicRows = oap::chost::GetRows (matrix1);
    dotProductDimPeriodic (output, matrix1, matrix2, dim, periodicRows);
  }

  void dotProduct (math::ComplexMatrix* output, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2, oap::generic::Dim32 dim) override;

  void dotProduct (math::ComplexMatrix* output, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2,
                   uintt outputDim[2], uintt params0Dim[2], uintt params1Dim[2])
  {
    oap::generic::Dim32 dim {{{outputDim[0], outputDim[1]}, {params0Dim[0], params0Dim[1]}, {params1Dim[0], params1Dim[1]}}};
    dotProduct (output, matrix1, matrix2, dim);
  }

  void transpose(math::ComplexMatrix* output, math::ComplexMatrix* matrix) override;

  void tanh (math::ComplexMatrix* output, math::ComplexMatrix* matrix) override;
  void sigmoid (math::ComplexMatrix* output, math::ComplexMatrix* matrix) override;
  void linear (math::ComplexMatrix* output, math::ComplexMatrix* matrix) override;
  void sin (math::ComplexMatrix* output, math::ComplexMatrix* matrix) override;
  void prelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix) override;
  void relu (math::ComplexMatrix* output, math::ComplexMatrix* matrix) override;
  void softplus (math::ComplexMatrix* output, math::ComplexMatrix* matrix) override;

  void tanh (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim2 dim) override;
  void sigmoid (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim2 dim) override;
  void linear (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim2 dim) override;
  void sin (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim2 dim) override;
  void prelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim2 dim) override;
  void relu (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim2 dim) override;
  void softplus (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim2 dim) override;

  void tanh (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim22 dim) override;
  void sigmoid (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim22 dim) override;
  void linear (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim22 dim) override;
  void sin (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim22 dim) override;
  void prelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim22 dim) override;
  void relu (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim22 dim) override;
  void softplus (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim22 dim) override;

  void dprelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix) override;
  void drelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix) override;
  void dprelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim2 dim) override;
  void drelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim2 dim) override;
  void dprelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim22 dim) override;
  void drelu (math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim22 dim) override;

  void dsigmoid (math::ComplexMatrix* output, math::ComplexMatrix* input) override;
  void dlinear (math::ComplexMatrix* output, math::ComplexMatrix* input) override;
  void dtanh (math::ComplexMatrix* output, math::ComplexMatrix* input) override;
  void dsin (math::ComplexMatrix* output, math::ComplexMatrix* input) override;
  void dsoftplus (math::ComplexMatrix* output, math::ComplexMatrix* input) override;
  void dsigmoid (math::ComplexMatrix* output, math::ComplexMatrix* input, oap::generic::Dim2 dim) override;
  void dlinear (math::ComplexMatrix* output, math::ComplexMatrix* input, oap::generic::Dim2 dim) override;
  void dtanh (math::ComplexMatrix* output, math::ComplexMatrix* input, oap::generic::Dim2 dim) override;
  void dsin (math::ComplexMatrix* output, math::ComplexMatrix* input, oap::generic::Dim2 dim) override;
  void dsoftplus (math::ComplexMatrix* output, math::ComplexMatrix* input, oap::generic::Dim2 dim) override;
  void dsigmoid (math::ComplexMatrix* output, math::ComplexMatrix* input, oap::generic::Dim22 dim) override;
  void dlinear (math::ComplexMatrix* output, math::ComplexMatrix* input, oap::generic::Dim22 dim) override;
  void dtanh (math::ComplexMatrix* output, math::ComplexMatrix* input, oap::generic::Dim22 dim) override;
  void dsin (math::ComplexMatrix* output, math::ComplexMatrix* input, oap::generic::Dim22 dim) override;
  void dsoftplus (math::ComplexMatrix* output, math::ComplexMatrix* input, oap::generic::Dim22 dim) override;
  void hadamardProductVec (math::ComplexMatrix* output, math::ComplexMatrix* param1, math::ComplexMatrix* param2) override;
  void add (math::ComplexMatrix* output, math::ComplexMatrix* param1, math::ComplexMatrix* param2) override;
  void multiplyReConstant (math::ComplexMatrix* output, math::ComplexMatrix* param1, floatt re) override;
  void sum (floatt& reoutput, floatt& imoutput, const math::ComplexMatrix* param) override;
  void setZeroMatrix (math::ComplexMatrix* param) override;

  //void sum (floatt& reoutput, floatt& imoutput, math::ComplexMatrix* params0);

  void crossEntropy (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1) override;

  void tensorProduct (math::ComplexMatrix* matrix, math::ComplexMatrix* params0, math::ComplexMatrix* params1, oap::generic::Dim32 dim) override;

  inline void tensorProduct (math::ComplexMatrix* matrix, math::ComplexMatrix* params0, math::ComplexMatrix* params1, uintt dim1[2], uintt dim2[2], uintt dim3[2])
  {
    oap::generic::Dim32 dim {{{dim1[0], dim1[1]}, {dim2[0], dim2[1]}, {dim3[0], dim3[1]}}};
    tensorProduct (matrix, params0, params1, dim);
  }

  void QRHT (math::ComplexMatrix* Q, math::ComplexMatrix* R, math::ComplexMatrix* A, math::ComplexMatrix* V, math::ComplexMatrix* VT, math::ComplexMatrix* P, math::ComplexMatrix* VVT);
  void setIdentity (math::ComplexMatrix* matrix);

  void setVector (math::ComplexMatrix* V, uintt column, math::ComplexMatrix* v, uintt length);

  void getVector (math::ComplexMatrix* vector, uintt length, math::ComplexMatrix* matrix, uintt column);

  void getVector (math::ComplexMatrix* vector, math::ComplexMatrix* matrix, uintt column);

  void convolve (math::ComplexMatrix* output, const math::ComplexMatrix* matrix, const math::ComplexMatrix* kernel);
  void poolAverage (math::ComplexMatrix* output, const math::ComplexMatrix* matrix, const math::MatrixDim& kernel);


  template<typename Matrices>
  void v2_add (Matrices& output, const Matrices& params1, floatt value)
  {
    oap::generic::addConstant (output, params1, value, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_add (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    oap::generic::add (output, params1, params2, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_subtract (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    oap::generic::subtract (output, params1, params2, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_dotProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    oap::generic::dotProduct (output, params1, params2, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy, oap::chost::GetMatrixInfo);
  }

  template<typename Matrices>
  void v2_multiply (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    v2_dotProduct<Matrices> (output, params1, params2);
  }

  template<typename Matrices>
  void v2_hadamardProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    oap::generic::hadamardProduct (output, params1, params2, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_hadamardProductVec (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    oap::generic::hadamardProductVec (output, params1, params2, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy, oap::chost::GetMatrixInfo);
  }

  template<typename Matrices>
  void v2_transpose (Matrices& output, const Matrices& params1)
  {
    oap::generic::transpose (output, params1, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_dsigmoid (Matrices& output, const Matrices& params1)
  {
    oap::generic::dsigmoid (output, params1, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_linear (Matrices& output, const Matrices& params1)
  {
    oap::generic::linear (output, params1, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_dlinear (Matrices& output, const Matrices& params1)
  {
    oap::generic::dlinear (output, params1, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);

  }

  template<typename Matrices>
  void v2_tanh (Matrices& output, const Matrices& params1)
  {
    oap::generic::tanh (output, params1, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_dtanh (Matrices& output, const Matrices& params1)
  {
    oap::generic::dtanh (output, params1, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_sin (Matrices& output, const Matrices& params1)
  {
    oap::generic::sin (output, params1, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_dsin (Matrices& output, const Matrices& params1)
  {
    oap::generic::dsin (output, params1, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_relu (Matrices& output, const Matrices& params1)
  {
    oap::generic::relu (output, params1, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_drelu (Matrices& output, const Matrices& params1)
  {
    oap::generic::drelu (output, params1, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_prelu (Matrices& output, const Matrices& params1)
  {
    oap::generic::relu (output, params1, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_dprelu (Matrices& output, const Matrices& params1)
  {
    oap::generic::drelu (output, params1, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_softplus (Matrices& output, const Matrices& params1)
  {
    oap::generic::relu (output, params1, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_dsoftplus (Matrices& output, const Matrices& params1)
  {
    oap::generic::drelu (output, params1, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_tensorProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    oap::generic::tensorProduct (output, params1, params2, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

  template<typename Matrices>
  void v2_sigmoid (Matrices& output, const Matrices& params1)
  {
    oap::generic::sigmoid (output, params1, &m_kernel, oap::chost::CreateThreadsMapper, malloc, free, memcpy);
  }

 private:
  uint m_threads[2];
  uint m_blocks[2];

  uintt m_maxThreadsPerBlock;
  HostKernelExecutor m_kernel;

  void prepare(math::ComplexMatrix* matrix, HostKernel& hostKernel);
  void prepare(size_t w, size_t h, HostKernel& hostKernel);

  oap::generic::BasicMatrixApi<decltype(oap::chost::GetMatrixInfo)> m_bmApi;

  uintt* createKernelArray (uintt* hostArray, size_t length)
  {
    return hostArray;
  }

  void _funcDim (const std::string& kname, math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim2 dim);
  void _funcDimPeriodic (const std::string& kname, math::ComplexMatrix* output, math::ComplexMatrix* matrix, oap::generic::Dim22 dim);
  std::function<uintt*(uintt*, uintt)> m_createKernelArray;
};
}
#endif  // HOSTCOMPAREPROCEDURE_H
