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

#ifndef OAP_CU_PROCEDURES_API_H
#define OAP_CU_PROCEDURES_API_H

#include <unordered_map>
#include <sstream>

#include "CudaBuffer.hpp"
#include "HostBuffer.hpp"

#include "Matrix.hpp"
#include "MatrixEx.hpp"
#include "CudaUtils.hpp"
#include "KernelExecutor.hpp"

#include "oapCudaMatrixUtils.hpp"

#include "RecToSquareApi.hpp"

#include "oapMemory.hpp"

#include "GenericCoreApi.hpp"
#include "GenericProceduresApi.hpp"
#include "GenericProceduresNewApi.hpp"

#include "oapProcedures.hpp"

#define CHECK_MATRIX(m) debugAssertMsg (m != NULL, "ComplexMatrix is nullptr.");

namespace oap
{

class CuProceduresApi : public oap::generic::SingleMatrixProcedures
{
 public:
  CuProceduresApi();
  virtual ~CuProceduresApi();

  CuProceduresApi(const CuProceduresApi&) = delete;
  CuProceduresApi(CuProceduresApi&&) = delete;
  CuProceduresApi& operator=(const CuProceduresApi&) = delete;
  CuProceduresApi& operator=(CuProceduresApi&&) = delete;

  void addDotProduct(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1);

  void tensorProduct(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1);

  void tensorProduct (math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1, generic::Dim32 dim) override;

  void tensorProduct (math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1, uintt outputD[2], uintt matrix1D[2], uintt matrix2D[2]);

  void hadamardProduct(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1);
  void elementWiseProduct(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1);
  void schurProduct(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1);

  /**
   *  @brief Calculates hadamard product of matrix and vector of the second matrix.
   *  @param outputs - outputs matrix (dimension n columns x n rows)
   *  @param params0 - matrix (dimension n columns x n rows)
   *  @param params1 - vector of the second matrix. The second matrix consists of repeated column vector (dimenstion must be 1 columns x n rows).
   *
   *            |a00 a10 a20|                 |v0 v0 v0|                                               |v0|
   *  outputs = |a01 a11 a21| hadamardProduct |v1 v1 v1| where a__ are values of params0 and params1 = |v1|
   *            |a02 a12 a22|                 |v2 v2 v2|                                               |v2|
   *
   *  Example of use: oapPartialHadamardProductTests.cpp
   */
  void hadamardProductVec(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1) override;

  void dotProduct (math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1);
//  void dotProduct (oap::MemoryRegionPtrs* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1);

  void dotProductShared (math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1);

  void addDotProduct(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1, uintt columns, uintt rows);
  void tensorProduct(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1, uintt columns, uintt rows);

  void calculateQTHQ(math::ComplexMatrix* outputs, math::ComplexMatrix* H, math::ComplexMatrix* Q,
                     math::ComplexMatrix* aux);

  void dotProductEx(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1, MatrixEx* matrixEx);

  /**
   * If outputs and matrix2 have more rows than columns of matrix1, then next following rows will be multiply as separated matrix.
   * This process will be continue to end of outputs's rows.
   *
   * If C = A * B and rows of A are lower than rows of C and
   * columns of A are lower than rows of B fo example:
   *
   *              B00 B01 B02
   *              B10 B11 B12
   *              B20 B21 B22
   *              B30 B31 B32
   *              B40 B41 B42
   *              B50 B51 B52
   *
   * A00 A01 A02  C00 C01 C02
   * A10 A11 A12  C10 C11 C12
   * A20 A21 A22  C20 C21 C22
   *              C30 C31 C32
   *              C40 C41 C42
   *              C50 C51 C52
   *
   * then behaviour of this procedure is
   *
   *              B00 B01 B02
   *              B10 B11 B12
   *              B20 B21 B22
   *              B30 B31 B32
   *              B40 B41 B42
   *              B50 B51 B52
   *
   * A00 A01 A02  C00 C01 C02
   * A10 A11 A12  C10 C11 C12
   * A20 A21 A22  C20 C21 C22
   * A00 A01 A02  C30 C31 C32
   * A10 A11 A12  C40 C41 C42
   * A20 A21 A22  C50 C51 C52
   *
   * so for example
   *
   * C10 = A10 * B00 + A11 * B10 + A12 * B30
   * C40 = A10 * B30 + A11 * B40 + A12 * B50
   *
   */
  void dotProductPeriodic (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2);

  /**
  * The same like in dotProductPeriodic but dimensions by matrices are defined by user.
  */
  void dotProductDimPeriodic (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2, generic::Dim32 dim, uintt periodicRows) override;

  void dotProductDimPeriodic (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2, generic::Dim32 dim)
  {
    uintt periodicRows = oap::cuda::GetRows(matrix1);
    dotProductDimPeriodic (outputs, matrix1, matrix2, dim, periodicRows);
  }

  void dotProduct (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2, generic::Dim32 dim) override;

  void dotProduct (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2,
                   uintt outputD[2], uintt matrix1D[2], uintt matrix2D[2])
  {
    generic::Dim32 dims {{{outputD[0], outputD[1]}, {matrix1D[0], matrix1D[1]}, {matrix2D[0], matrix2D[1]}}};

    dotProduct (outputs, matrix1, matrix2, dims);
  }

  void dotProductEx(math::ComplexMatrix* outputs, math::ComplexMatrix* params0,
                    math::ComplexMatrix* params1, MatrixEx* matrixEx, uintt columns,
                    uintt rows);

  void dotProductOpt(math::ComplexMatrix* outputs, math::ComplexMatrix* params0,
                            math::ComplexMatrix* params1);

  void dotProductOpt(math::ComplexMatrix* outputs, math::ComplexMatrix* params0,
                     math::ComplexMatrix* params1, uintt ocolumns, uintt orows,
                     uintt p1rows, uintt p2columns);

  void dotProductExOpt(math::ComplexMatrix* outputs, math::ComplexMatrix* params0,
                       math::ComplexMatrix* params1, MatrixEx* matrixEx);

  void transposeEx(math::ComplexMatrix* outputs, math::ComplexMatrix* params0,
                         MatrixEx* matrixEx);

  void transpose(math::ComplexMatrix* outputs, math::ComplexMatrix* params0) override;

  void conjugateTranspose(math::ComplexMatrix* outputs, math::ComplexMatrix* params0);

  void subtract(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1) override;
  void addSubstract(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1);

  void crossEntropy(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1) override;

  void subtract(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1, uintt columns, uintt rows);
  void addSubstract(math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1, uintt columns, uintt rows);

  void add (math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1) override;

  void add (math::ComplexMatrix* outputs, math::ComplexMatrix* params0, math::ComplexMatrix* params1, uintt columns, uintt rows);
  void add (math::ComplexMatrix* outputs, const math::ComplexMatrix* params0, floatt value);

  void setVector(math::ComplexMatrix* outputs, uintt column, math::ComplexMatrix* params0,
                 uintt length);

  void getVector(math::ComplexMatrix* vector, uintt length, math::ComplexMatrix* matrix,
                 uintt column);

  void getVector(math::ComplexMatrix* vector, math::ComplexMatrix* matrix, uintt column);

  void magnitude (floatt& outputs, math::ComplexMatrix* params0);

  void sum (floatt& reoutput, floatt& imoutput, const math::ComplexMatrix* matrix) override;
  //void sum (floatt& reoutput, const math::ComplexMatrix* matrix);
  //void sum (floatt& reoutput, const floatt* values, size_t count);
  //void sumShared (floatt& outputs, math::ComplexMatrix* params0);

  void magnitudeOpt(floatt& outputs, math::ComplexMatrix* params0);

  void magnitudeOptVer2(floatt& outputs, math::ComplexMatrix* params0);

  void magnitude2(floatt& outputs, math::ComplexMatrix* params0);

  void magnitude2Opt(floatt& outputs, math::ComplexMatrix* params0);

  void magnitude2OptVer2(floatt& outputs, math::ComplexMatrix* params0);

  void multiplyReConstant(math::ComplexMatrix* v, math::ComplexMatrix* f, floatt re) override;

  void multiplyConstant(math::ComplexMatrix* v, math::ComplexMatrix* f, floatt re, floatt im);

  void setDiagonal(math::ComplexMatrix* matrix, floatt re, floatt im);

  void setIdentity(math::ComplexMatrix* matrix);

  void setZeroMatrix(math::ComplexMatrix* matrix) override;

  bool compare(math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2, floatt tolerance = 0);

  bool compareVer2(math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2, floatt tolerance = 0);

  // Sigmoid function and derivatives
  void sigmoid (math::ComplexMatrix* matrix);
  void sigmoid (math::ComplexMatrix* matrix, generic::Dim2 dim);
  void sigmoid (math::ComplexMatrix* matrix, generic::Dim22 dim);

  void sigmoid (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix) override;
  void sigmoid (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim2 dim) override;
  void sigmoid (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim22 dim) override;

  void dsigmoid (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix) override;
  void dsigmoid (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, generic::Dim2 dim) override;
  void dsigmoid (math::ComplexMatrix* omatrix, math::ComplexMatrix* imatrix, generic::Dim22 dim) override;

  void multiplyDSigmoid (math::ComplexMatrix* omatrix, math::ComplexMatrix* matrix);
  void multiplyDSigmoid (math::ComplexMatrix* omatrix, math::ComplexMatrix* matrix, generic::Dim2 dim);
  void multiplyDSigmoid (math::ComplexMatrix* omatrix, math::ComplexMatrix* matrix, generic::Dim22 dim);

  // Linear function and derivatives
  void linear (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix) override;
  void linear (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim2 dim) override;
  void linear (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim22 dim) override;

  void dlinear (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix) override;
  void dlinear (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim2 dim) override;
  void dlinear (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim22 dim) override;

  // Tanh/tanh function and derivatives
  void tanh (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix) override;
  void tanh (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim2 dim) override;
  void tanh (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim22 dim) override;

  void dtanh (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix) override;
  void dtanh (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim2 dim) override;
  void dtanh (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim22 dim) override;

  // Sin/sin function and derivatives
  void sin (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix) override;
  void sin (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim2 dim) override;
  void sin (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim22 dim) override;

  void dsin (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix) override;
  void dsin (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim2 dim) override;
  void dsin (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim22 dim) override;

  void multiplyDSin (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix);
  void multiplyDSin (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim2 dim);
  void multiplyDSin (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim22 dim);

  // Relu/relu function and derivatives
  void relu (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix) override;
  void relu (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim2 dim) override;
  void relu (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim22 dim) override;

  void drelu (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix) override;
  void drelu (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim2 dim) override;
  void drelu (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim22 dim) override;

  // PRelu/prelu function and derivatives where paramters is 0.01
  void prelu (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix) override;
  void prelu (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim2 dim) override;
  void prelu (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim22 dim) override;

  void dprelu (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix) override;
  void dprelu (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim2 dim) override;
  void dprelu (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim22 dim) override;

  // Softplus/softplus function and derivatives
  void softplus (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix) override;
  void softplus (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim2 dim) override;
  void softplus (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim22 dim) override;

  void dsoftplus (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix) override;
  void dsoftplus (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim2 dim) override;
  void dsoftplus (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix, generic::Dim22 dim) override;

  /**
   * \brief Convolution operation
   */
  void convolve (math::ComplexMatrix* outputs, const math::ComplexMatrix* matrix, const math::ComplexMatrix* kernel);

  /**
   * \brief Pooling operation
   */
  void poolAverage (math::ComplexMatrix* outputs, const math::ComplexMatrix* matrix, const math::MatrixDim& kernel);

  /**
   * \brief mean of values in matrix
   */
  floatt mean (const math::ComplexMatrix* matrix);

  /**
   * \brief standard deviation of values in matrix
   */
  floatt stddv (const math::ComplexMatrix* matrix, floatt mean);

  /**
   * \brief standard deviation of values in matrix
   */
  floatt stddv (const math::ComplexMatrix* matrix);

  /**
   * \brief Scale matrix in the way: (x - mean) / standard_deviation
   */
  void scale (math::ComplexMatrix* matrix);

  void dotProduct (oap::Memory& outputs, const oap::Memory& arg1, const oap::Memory& arg2, const oap::MemoryRegion_3_Args* regions);

  inline void cos (math::ComplexMatrix* outputs, math::ComplexMatrix* matrix)
  {
    dsin (outputs, matrix);
  }

  floatt getCompareOperationSum() const;

  void QRGR(math::ComplexMatrix* Q, math::ComplexMatrix* R, math::ComplexMatrix* H, math::ComplexMatrix* R1,
            math::ComplexMatrix* Q1, math::ComplexMatrix* G, math::ComplexMatrix* GT);

  void QRHT (math::ComplexMatrix* Q, math::ComplexMatrix* R, math::ComplexMatrix* A, math::ComplexMatrix* V, math::ComplexMatrix* VT, math::ComplexMatrix* P, math::ComplexMatrix* VVT);

  bool isUpperTriangular(math::ComplexMatrix* matrix);

  void calcTriangularH (math::ComplexMatrix* H, math::ComplexMatrix* Q, math::ComplexMatrix* R,
                        math::ComplexMatrix* aux1, math::ComplexMatrix* aux2, math::ComplexMatrix* aux3,
                        math::ComplexMatrix* aux4, math::ComplexMatrix* aux5, math::ComplexMatrix* aux6);

  void calcTriangularHStep (math::ComplexMatrix* H, math::ComplexMatrix* Q, math::ComplexMatrix* R,
                            math::ComplexMatrix* aux1, math::ComplexMatrix* aux2, math::ComplexMatrix* aux3,
                            math::ComplexMatrix* aux4, math::ComplexMatrix* aux5, math::ComplexMatrix* aux6);

  template<typename Matrices>
  void v2_add (Matrices& outputs, const Matrices& params1, floatt value);

  template<typename Matrices>
  void v2_add (Matrices& outputs, const Matrices& params1, const Matrices& params2);

  template<typename Matrices>
  void v2_subtract (Matrices& outputs, const Matrices& params1, const Matrices& params2);

  template<typename Matrices>
  void v2_dotProduct (Matrices& outputs, const Matrices& params1, const Matrices& params2);

  template<typename Matrices>
  void v2_multiply (Matrices& outputs, const Matrices& params1, const Matrices& params2);

  template<typename Matrices>
  void v2_hadamardProduct (Matrices& outputs, const Matrices& params1, const Matrices& params2);

  template<typename Matrices>
  void v2_hadamardProductVec (Matrices& outputs, const Matrices& params1, const Matrices& params2);

  template<typename Matrices>
  void v2_tensorProduct (Matrices& outputs, const Matrices& params1, const Matrices& params2);

  template<typename Matrices>
  void v2_transpose (Matrices& outputs, const Matrices& params1);

  template<typename Matrices>
  void v2_sigmoid (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_dsigmoid (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_multiplyDSigmoid (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_linear (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_dlinear (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_multiplyDLinear (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_tanh (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_dtanh (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_multiplyDTanh (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_sin (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_dsin (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_multiplyDSin (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_relu (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_drelu (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_multiplyDRelu (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_prelu (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_dprelu (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_multiplyDPrelu (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_softplus (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_dsoftplus (Matrices& outputs, const Matrices& params);

  template<typename Matrices>
  void v2_multiplyDSoftplus (Matrices& outputs, const Matrices& params);

  std::string getMsgStatus() const;

 private:
  enum QRType { NORMAL, OPT };
  enum Type { HOST, CUDA };

  bool m_cuStatus;

  int* m_doutputIsTriangular;
  floatt* m_magnitudeOutput;

  bool m_isIntialized;
  oap::cuda::Kernel m_kernel;
  uintt m_maxThreadsPerBlock;
  floatt m_compareOperationOutput;

  uintt m_columns;
  uintt m_rows;

  bool m_isSetColumns;
  bool m_isSetRows;

  uint m_blocks[2];
  uint m_threads[2];

  void init();

  void calculateDims (uint blocks[2], uint threads[2], uintt w, uintt h);
  void prepareDims (uintt w, uintt h);

  enum Types
  {
    MATRIX,
    SCALAR
  };

  bool execute(const char* functionName, uintt w, uintt h, void** params, uintt sharedMemory, bool _prepareDims = true);


  void qrProcedure(QRType qrType, math::ComplexMatrix* Q, math::ComplexMatrix* R,
                   math::ComplexMatrix* A, math::ComplexMatrix* AT, math::ComplexMatrix* P,
                   math::ComplexMatrix* I, math::ComplexMatrix* v, math::ComplexMatrix* vt,
                   math::ComplexMatrix* vvt);

  floatt compareProcedure(const char* cuKernelName, math::ComplexMatrix* matrix1,
                        math::ComplexMatrix* matrix2, uintt w, uintt h, uintt wthreads,
                        uintt hthreads);

  floatt magnitude2Procedure(const char* cuKernelName, math::ComplexMatrix* matrix1,
                             uintt wthreads, uintt hthreads);

  floatt magnitude2Procedure_GetOutput(uint blocks[2], uintt outputSize) const;

  void deallocKernelArrays ();

  inline void resetFlags() {
    m_isSetRows = false;
    m_isSetColumns = false;
  }
private:

  static uintt GetColumns(const math::ComplexMatrix* matrix);
  static uintt GetRows(const math::ComplexMatrix* matrix);

  oap::TBuffer<floatt, oap::Type::CUDA> m_dcompareOutputBuffer;
  oap::TBuffer<floatt, oap::Type::CUDA> m_dcompareBuffer;
  oap::TBuffer<floatt, oap::Type::HOST> m_hcompareOutputBuffer;
  oap::TBuffer<floatt, oap::Type::CUDA> m_magnitudeBuffer;
  oap::TBuffer<floatt, oap::Type::CUDA> m_dmagnitudeOutputBuffer;
  oap::TBuffer<floatt, oap::Type::CUDA> m_dmagnitudeBuffer;
  oap::TBuffer<floatt, oap::Type::HOST> m_hmagnitudeOutputBuffer;
  oap::TBuffer<int, oap::Type::CUDA> m_disuppertriangularOutputBuffer;
  oap::TBuffer<int, oap::Type::HOST> m_hisuppertriangularOutputBuffer;
  oap::TBuffer<floatt, oap::Type::CUDA> m_dqrSums;
  oap::TBuffer<floatt, oap::Type::CUDA> m_dqrBuffer;
  oap::TBuffer<floatt, oap::Type::CUDA> m_dsumsReBuffer;
  oap::TBuffer<floatt, oap::Type::CUDA> m_dsumsImBuffer;
  oap::TBuffer<floatt, oap::Type::HOST> m_hsumsReBuffer;
  oap::TBuffer<floatt, oap::Type::HOST> m_hsumsImBuffer;

  MatrixEx* m_dMatrixEx = nullptr;
  std::unordered_map<size_t, uint*> m_kernelArrays;

  uintt* createKernelArray (uintt* hostArray, size_t length)
  {
    auto it = m_kernelArrays.find (length);
    if (it == m_kernelArrays.end ())
    {
      uintt* kernelArray = static_cast<uintt*>(CudaUtils::AllocDeviceMem (length * sizeof(uintt)));
      m_kernelArrays[length] = kernelArray;
    }

    uintt* array = m_kernelArrays [length];
    CudaUtils::CopyHostToDevice (array, hostArray, length * sizeof(uintt));

    return array;
  }
/*
  uintt* createKernelArray (oap::generic::Dim32 dim32)
  {
    uintt harray [6] =
    {
      dim32[0][0], dim32[0][1],
      dim32[1][0], dim32[1][1],
      dim32[2][0], dim32[2][1]
    };
    return createKernelArray (harray, 6);
  }

  uintt* createKernelArray (oap::generic::Dim22 dim22)
  {
    uintt harray [4] =
    {
      dim22[0][0], dim22[0][1],
      dim22[1][0], dim22[1][1],
    };
    return createKernelArray (harray, 4);
  }

  uintt* createKernelArray (oap::generic::Dim2 dim2)
  {
    return createKernelArray (dim2.data(), 2);
  }*/

  MatrixEx* createDeviceMatrixEx(const MatrixEx& host)
  {
    if (m_dMatrixEx == nullptr)
    {
      m_dMatrixEx = oap::cuda::NewDeviceMatrixExCopy (host);
    }

    CudaUtils::CopyHostToDevice (m_dMatrixEx, &host, sizeof(MatrixEx));
    return m_dMatrixEx;
  }

  oap::generic::BasicMatrixApi<decltype(oap::cuda::GetMatrixInfo)> m_bmApi;
  std::function<void()> m_preExecCallback;//(std::bind(&CuProceduresApi::resetFlags, this)
  std::function<uintt*(uintt*, uintt)> m_createKernelArray;
};

template<typename Matrices>
void CuProceduresApi::v2_add (Matrices& outputs, const Matrices& params1, floatt value)
{
  m_cuStatus = oap::generic::addConstant (outputs, params1, value, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_add (Matrices& outputs, const Matrices& params1, const Matrices& params2)
{
  m_cuStatus = oap::generic::add (outputs, params1, params2, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_subtract (Matrices& outputs, const Matrices& params1, const Matrices& params2)
{
  m_cuStatus = oap::generic::subtract (outputs, params1, params2, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_dotProduct (Matrices& outputs, const Matrices& params1, const Matrices& params2)
{
  m_cuStatus = oap::generic::dotProduct (outputs, params1, params2, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice, oap::cuda::GetMatrixInfo);
}

template<typename Matrices>
void CuProceduresApi::v2_multiply (Matrices& outputs, const Matrices& params1, const Matrices& params2)
{
  m_cuStatus = oap::generic::dotProduct (outputs, params1, params2, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice, oap::cuda::GetMatrixInfo);
}

template<typename Matrices>
void CuProceduresApi::v2_hadamardProduct (Matrices& outputs, const Matrices& params1, const Matrices& params2)
{
  m_cuStatus = oap::generic::hadamardProduct (outputs, params1, params2, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_hadamardProductVec (Matrices& outputs, const Matrices& params1, const Matrices& params2)
{
  m_cuStatus = oap::generic::hadamardProductVec (outputs, params1, params2, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice, oap::cuda::GetMatrixInfo);
}

template<typename Matrices>
void CuProceduresApi::v2_tensorProduct (Matrices& outputs, const Matrices& params1, const Matrices& params2)
{
  m_cuStatus = oap::generic::tensorProduct (outputs, params1, params2, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_transpose (Matrices& outputs, const Matrices& params1)
{
  m_cuStatus = oap::generic::transpose (outputs, params1, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_sigmoid (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::sigmoid (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_dsigmoid (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::dsigmoid (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_multiplyDSigmoid (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::multiplyDSigmoid (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_linear (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::linear (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_dlinear (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::dlinear (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_tanh (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::tanh (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_dtanh (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::dtanh (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_sin (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::sin (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_dsin (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::dsin (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_multiplyDSin (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::multiplyDSin (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_relu (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::relu (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_drelu (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::drelu (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_multiplyDRelu (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::multiplyDRelu (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_prelu (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::prelu (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_dprelu (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::dprelu (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_multiplyDPrelu (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::multiplyDPrelu (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_softplus (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::softplus (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_dsoftplus (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::dsoftplus (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

template<typename Matrices>
void CuProceduresApi::v2_multiplyDSoftplus (Matrices& outputs, const Matrices& params)
{
  m_cuStatus = oap::generic::multiplyDSoftplus (outputs, params, &m_kernel, oap::cuda::CreateThreadsMapper, CudaUtils::Malloc, CudaUtils::Free, CudaUtils::CopyHostToDevice);
}

}

#endif /* MATRIXPROCEDURES_H */
