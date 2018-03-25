/*
 * Copyright 2016, 2017 Marcin Matula
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

#ifndef OAP_CUPROCEDURESAPI_H
#define OAP_CUPROCEDURESAPI_H

#include "Matrix.h"
#include "MatrixEx.h"
#include "CudaUtils.h"
#include "KernelExecutor.h"

namespace oap
{

class CuProceduresApi
{
 public:
  CuProceduresApi();
  virtual ~CuProceduresApi();

  inline void dotProduct(math::Matrix* output, math::Matrix* params0,
                         math::Matrix* params1);

  void dotProduct(math::Matrix* output, math::Matrix* params0,
                  math::Matrix* params1, uintt columns, uintt rows);

  void calculateQTHQ(math::Matrix* output, math::Matrix* H, math::Matrix* Q,
                     math::Matrix* aux);

  inline void dotProductEx(math::Matrix* output, math::Matrix* params0,
                           math::Matrix* params1, MatrixEx* matrixEx);

  void dotProductEx(math::Matrix* output, math::Matrix* params0,
                    math::Matrix* params1, MatrixEx* matrixEx, uintt columns,
                    uintt rows);

  inline void dotProductOpt(math::Matrix* output, math::Matrix* params0,
                            math::Matrix* params1);

  void dotProductOpt(math::Matrix* output, math::Matrix* params0,
                     math::Matrix* params1, uintt ocolumns, uintt orows,
                     uintt p1rows, uintt p2columns);

  void dotProductExOpt(math::Matrix* output, math::Matrix* params0,
                       math::Matrix* params1, MatrixEx* matrixEx);

  void transposeEx(math::Matrix* output, math::Matrix* params0,
                         MatrixEx* matrixEx);

  void transpose(math::Matrix* output, math::Matrix* params0);

  void conjugateTranspose(math::Matrix* output, math::Matrix* params0);

  inline void substract(math::Matrix* output, math::Matrix* params0,
                        math::Matrix* params1);

  void substract(math::Matrix* output, math::Matrix* params0,
                 math::Matrix* params1, uintt columns, uintt rows);

  inline void add(math::Matrix* output, math::Matrix* params0,
                  math::Matrix* params1);

  void add(math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
           uintt columns, uintt rows);

  void setVector(math::Matrix* output, uintt column, math::Matrix* params0,
                 uintt length);

  void getVector(math::Matrix* vector, uintt length, math::Matrix* matrix,
                 uintt column);

  void getVector(math::Matrix* vector, math::Matrix* matrix, uintt column);

  void magnitude(floatt& output, math::Matrix* params0);

  void magnitudeOpt(floatt& output, math::Matrix* params0);

  void magnitudeOptVer2(floatt& output, math::Matrix* params0);

  void magnitude2(floatt& output, math::Matrix* params0);

  void magnitude2Opt(floatt& output, math::Matrix* params0);

  void magnitude2OptVer2(floatt& output, math::Matrix* params0);

  void multiplyReConstant(math::Matrix* v, math::Matrix* f, floatt re);

  void multiplyConstant(math::Matrix* v, math::Matrix* f, floatt re, floatt im);

  void setDiagonal(math::Matrix* matrix, floatt re, floatt im);

  void setIdentity(math::Matrix* matrix);

  void setZeroMatrix(math::Matrix* matrix);

  bool compare(math::Matrix* matrix1, math::Matrix* matrix2, floatt tolerance = 0);

  bool compareVer2(math::Matrix* matrix1, math::Matrix* matrix2, floatt tolerance = 0);

  floatt getCompareOperationSum() const;

  void QRGR(math::Matrix* Q, math::Matrix* R, math::Matrix* H, math::Matrix* R1,
            math::Matrix* Q1, math::Matrix* G, math::Matrix* GT);

  bool isUpperTriangular(math::Matrix* matrix);

  void calcTriangularHStep(math::Matrix* H, math::Matrix* Q, math::Matrix* R,
            math::Matrix* aux1, math::Matrix* aux2, math::Matrix* aux3,
            math::Matrix* aux4, math::Matrix* aux5, math::Matrix* aux6);

  void calcTriangularHStep(math::Matrix* H, math::Matrix* Q, math::Matrix* R,
            math::Matrix* aux1, math::Matrix* aux2, math::Matrix* aux3,
            math::Matrix* aux4, math::Matrix* aux5, math::Matrix* aux6,
            uint columns, uint rows);

  CUresult getStatus() const;

 private:
  enum QRType { NORMAL, OPT };
  enum Type { HOST, CUDA };

  CUresult m_cuResult;

  int* m_doutputIsTriangular;
  floatt* m_magniuteOutput;

  bool m_isIntialized;
  oap::cuda::Kernel m_kernel;
  uintt m_maxThreadsPerBlock;
  floatt m_compareOperationOutput;

  uintt m_columns;
  uintt m_rows;

  bool m_isSetColumns;
  bool m_isSetRows;

  CuProceduresApi(const CuProceduresApi&);

  void init();

  void prepareDims(uintt w, uintt h);

  CUresult execute(const char* functionName, uintt w, uintt h, void** params,
                   uintt sharedMemory, bool _prepareDims = true);


  void qrProcedure(QRType qrType, math::Matrix* Q, math::Matrix* R,
                   math::Matrix* A, math::Matrix* AT, math::Matrix* P,
                   math::Matrix* I, math::Matrix* v, math::Matrix* vt,
                   math::Matrix* vvt);

  floatt compareProcedure(const char* cuKernelName, math::Matrix* matrix1,
                        math::Matrix* matrix2, uintt w, uintt h, uintt wthreads,
                        uintt hthreads);

  floatt magnitude2Procedure(const char* cuKernelName, math::Matrix* matrix1,
                             uintt wthreads, uintt hthreads);

  floatt magnitude2Procedure_GetOutput(uint blocks[2], uintt outputSize) const;
  inline void resetFlags() {
    m_isSetRows = false;
    m_isSetColumns = false;
  }
private:

  template <typename T>
  class Buffer {
   public:
    T* m_buffer;
    uintt m_length;

    Buffer(oap::CuProceduresApi::Type type);
    ~Buffer();

    void realloc(uintt length);
    size_t GetSizeOfType() const { return sizeof(T); }

   private:
    Type m_type;
    void free(T* buffer);
    T* alloc(uintt length);
  };

  Buffer<floatt> m_dcompareOutputBuffer;
  Buffer<floatt> m_dcompareBuffer;
  Buffer<floatt> m_hcompareOutputBuffer;
  Buffer<floatt> m_magnitudeBuffer;
  Buffer<floatt> m_dmagnitudeOutputBuffer;
  Buffer<floatt> m_dmagnitudeBuffer;
  Buffer<floatt> m_hmagnitudeOutputBuffer;
  Buffer<int> m_disuppertriangularOutputBuffer;
  Buffer<int> m_hisuppertriangularOutputBuffer;
  Buffer<floatt> m_dqrSums;
  Buffer<floatt> m_dqrBuffer;
};

template <typename T>
CuProceduresApi::Buffer<T>::Buffer(oap::CuProceduresApi::Type type)
    : m_buffer(NULL), m_length(0), m_type(type) {
  // not implemented
}

template <typename T>
CuProceduresApi::Buffer<T>::~Buffer() {
  if (m_buffer != NULL && m_type == CUDA) {
    free(m_buffer);
  }
}

template <typename T>
void CuProceduresApi::Buffer<T>::realloc(uintt length) {
  if (length > m_length) {
    if (m_buffer != NULL) {
      free(m_buffer);
    }
    m_buffer = alloc(length);
    m_length = length;
  }
}

template <typename T>
void CuProceduresApi::Buffer<T>::free(T* buffer) {
  if (m_type == oap::CuProceduresApi::CUDA) {
    CudaUtils::FreeDeviceMem(m_buffer);
  } else if (m_type == oap::CuProceduresApi::HOST) {
    delete[] buffer;
  }
}

template <typename T>
T* CuProceduresApi::Buffer<T>::alloc(uintt length) {
  switch (m_type) {
    case oap::CuProceduresApi::CUDA:
      return static_cast<T*>(CudaUtils::AllocDeviceMem(length * sizeof(T)));
    case oap::CuProceduresApi::HOST:
      return new T[length];
  };
}

inline void CuProceduresApi::dotProduct(math::Matrix* output, math::Matrix* params0,
                                 math::Matrix* params1) {
  const uintt columns = CudaUtils::GetColumns(output);
  const uintt rows = CudaUtils::GetRows(output);
  dotProduct(output, params0, params1, columns, rows);
}

inline void CuProceduresApi::dotProductEx(math::Matrix* output, math::Matrix* params0,
                                   math::Matrix* params1, MatrixEx* matrixEx) {
  const uintt columns = CudaUtils::GetColumns(matrixEx);
  const uintt rows = CudaUtils::GetRows(matrixEx);
  dotProductEx(output, params0, params1, matrixEx, columns, rows);
}

inline void CuProceduresApi::dotProductOpt(math::Matrix* output, math::Matrix* params0,
                                    math::Matrix* params1) {
  const uintt ocolumns = CudaUtils::GetColumns(output);
  const uintt orows = CudaUtils::GetRows(output);
  const uintt p1rows = CudaUtils::GetRows(params0);
  const uintt p2columns = CudaUtils::GetColumns(params1);
  dotProductOpt(output, params0, params1, ocolumns, orows, p1rows, p2columns);
}

inline void CuProceduresApi::substract(math::Matrix* output, math::Matrix* params0,
                                math::Matrix* params1) {
  const uintt columns = CudaUtils::GetColumns(output);
  const uintt rows = CudaUtils::GetRows(output);
  substract(output, params0, params1, columns, rows);
}

inline void CuProceduresApi::add(math::Matrix* output, math::Matrix* params0,
                          math::Matrix* params1) {
  const uintt columns = CudaUtils::GetColumns(output);
  const uintt rows = CudaUtils::GetRows(output);
  add(output, params0, params1, columns, rows);
}

}

#endif /* MATRIXPROCEDURES_H */
