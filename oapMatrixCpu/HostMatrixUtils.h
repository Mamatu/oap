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

#ifndef OAP_HOST_MATRIX_UTILS_H
#define OAP_HOST_MATRIX_UTILS_H

#include "MatrixModules.h"
#include "Matrix.h"
#include "MatrixInfo.h"
#include "MatrixEx.h"
#include <stdio.h>
#include "ThreadUtils.h"

namespace host {

/**
 * @brief NewMatrixCopy
 * @param matrix
 * @return
 */
math::Matrix* NewMatrixCopy(const math::Matrix* matrix);

/**
 * @brief NewMatrixCopy
 * @param columns
 * @param rows
 * @param reArray
 * @param imArray
 * @return
 */
math::Matrix* NewMatrixCopy(uintt columns, uintt rows, floatt* reArray,
                            floatt* imArray);

/**
 * @brief NewReMatrixCopy
 * @param columns
 * @param rows
 * @param reArray
 * @return
 */
math::Matrix* NewReMatrixCopy(uintt columns, uintt rows, floatt* reArray);

/**
 * @brief NewImMatrixCopy
 * @param columns
 * @param rows
 * @param imArray
 * @return
 */
math::Matrix* NewImMatrixCopy(uintt columns, uintt rows, floatt* imArray);

/**
 * @brief NewMatrix
 * @param matrix
 * @param value
 * @return
 */
math::Matrix* NewMatrix(const math::Matrix* matrix, floatt value = 0);

/**
 * @brief NewMatrix
 * @param matrix
 * @param columns
 * @param rows
 * @param value
 * @return
 */
math::Matrix* NewMatrix(const math::Matrix* matrix, uintt columns, uintt rows,
                        floatt value = 0);

/**
 * @brief NewMatrix
 * @param columns
 * @param rows
 * @param value
 * @return
 */
math::Matrix* NewMatrix(uintt columns, uintt rows, floatt value = 0);

math::Matrix* NewMatrix(const math::MatrixInfo& matrixInfo, floatt value = 0);

/**
 * @brief NewMatrix
 * @param isre
 * @param isim
 * @param columns
 * @param rows
 * @param value
 * @return
 */
math::Matrix* NewMatrix(bool isre, bool isim, uintt columns, uintt rows,
                        floatt value = 0);

/**
 * @brief NewReMatrix
 * @param columns
 * @param rows
 * @param value
 * @return
 */
math::Matrix* NewReMatrix(uintt columns, uintt rows, floatt value = 0);

/**
 * @brief NewImMatrix
 * @param columns
 * @param rows
 * @param value
 * @return
 */
math::Matrix* NewImMatrix(uintt columns, uintt rows, floatt value = 0);

/**
 * @brief NewMatrix
 * @param text
 * @return
 */
math::Matrix* NewMatrix(const std::string& text);

inline void CopyBuffer(floatt* dst, floatt* src, uintt length) {
  memcpy(dst, src, length * sizeof(floatt));
}

/**
 * @brief CopyMatrix
 * @param dst
 * @param src
 */
void CopyMatrix(math::Matrix* dst, const math::Matrix* src);

/**
 * Copy data to dst matrix which has one column and row less than
 * src matrix. Row and column which will be omitted are added as params..
 * @param dst
 * @param src
 * @param column index of column which will be omitted
 * @param row index of row which will be omitted
 */
void Copy(math::Matrix* dst, const math::Matrix* src, uintt column, uintt row);

/**
 * @brief Copy
 * @param dst
 * @param src
 * @param MatrixEx
 */
void Copy(math::Matrix* dst, const math::Matrix* src, const MatrixEx& matrixEx);

/**
 * @brief CopyRe
 * @param dst
 * @param src
 */
void CopyRe(math::Matrix* dst, const math::Matrix* src);

/**
 * @brief CopyIm
 * @param dst
 * @param src
 */
void CopyIm(math::Matrix* dst, const math::Matrix* src);

/**
 * @brief DeleteMatrix
 * @param matrix
 */
void DeleteMatrix(math::Matrix* matrix);

/**
 * @brief GetReValue
 * @param matrix
 * @param column
 * @param row
 * @return
 */
floatt GetReValue(const math::Matrix* matrix, uintt column, uintt row);

/**
 * @brief SetReValue
 * @param matrix
 * @param column
 * @param row
 * @param value
 */
void SetReValue(const math::Matrix* matrix, uintt column, uintt row,
                floatt value);

/**
 * @brief GetImValue
 * @param matrix
 * @param column
 * @param row
 * @return
 */
floatt GetImValue(const math::Matrix* matrix, uintt column, uintt row);

/**
 * @brief SetImValue
 * @param matrix
 * @param column
 * @param row
 * @param value
 */
void SetImValue(const math::Matrix* matrix, uintt column, uintt row,
                floatt value);

std::string GetMatrixStr(const math::Matrix* matrix);

/**
 * @brief PrintMatrix
 * @param text
 * @param matrix
 */
void PrintMatrix(const std::string& text, const math::Matrix* matrix);

void PrintMatrix(FILE* stream, const std::string& text, const math::Matrix* matrix);

void PrintMatrix(FILE* stream, const math::Matrix* matrix);

/**
 * @brief PrintMatrix
 * @param matrix
 */
void PrintMatrix(const math::Matrix* matrix);

bool PrintMatrixToFile(const std::string& path, const std::string& text, const math::Matrix* matrix);

bool PrintMatrixToFile(const std::string& path, const math::Matrix* matrix);
/**
 * @brief PrintImMatrix
 * @param stream
 * @param matrix
 */
void PrintImMatrix(FILE* stream, const math::Matrix* matrix);

/**
 * @brief PrintImMatrix
 * @param matrix
 */
void PrintImMatrix(const math::Matrix* matrix);

/**
 * @brief PrintImMatrix
 * @param text
 * @param matrix
 */
void PrintImMatrix(const std::string& text, const math::Matrix* matrix);

void SetVector(math::Matrix* matrix, uintt column, math::Matrix* vector);

void SetVector(math::Matrix* matrix, uintt column, floatt* revector,
               floatt* imvector, uintt length);

/**
 * @brief SetReVector
 * @param matrix
 * @param column
 * @param vector
 * @param length
 */
void SetReVector(math::Matrix* matrix, uintt column, floatt* vector,
                 uintt length);

/**
 * @brief SetTransposeReVector
 * @param matrix
 * @param row
 * @param vector
 * @param length
 */
void SetTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector,
                          uintt length);

/**
 * @brief SetImVector
 * @param matrix
 * @param column
 * @param vector
 * @param length
 */
void SetImVector(math::Matrix* matrix, uintt column, floatt* vector,
                 uintt length);

/**
 * @brief SetTransposeImVector
 * @param matrix
 * @param row
 * @param vector
 * @param length
 */
void SetTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector,
                          uintt length);

/**
 * @brief SetReVector
 * @param matrix
 * @param column
 * @param vector
 */
void SetReVector(math::Matrix* matrix, uintt column, floatt* vector);

/**
 * @brief SetTransposeReVector
 * @param matrix
 * @param row
 * @param vector
 */
void SetTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector);

/**
 * @brief SetImVector
 * @param matrix
 * @param column
 * @param vector
 */
void SetImVector(math::Matrix* matrix, uintt column, floatt* vector);

/**
 * @brief SetTransposeImVector
 * @param matrix
 * @param row
 * @param vector
 */
void SetTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector);

void GetVector(math::Matrix* vector, math::Matrix* matrix, uintt column);

void GetVector(floatt* revector, floatt* imvector, uintt length,
               math::Matrix* matrix, uintt column);

/**
 * @brief GetReVector
 * @param vector
 * @param length
 * @param matrix
 * @param column
 */
void GetReVector(floatt* vector, uintt length, math::Matrix* matrix,
                 uintt column);

/**
 * @brief GetTransposeReVector
 * @param vector
 * @param length
 * @param matrix
 * @param row
 */
void GetTransposeReVector(floatt* vector, uintt length, math::Matrix* matrix,
                          uintt row);

/**
 * @brief GetImVector
 * @param vector
 * @param length
 * @param matrix
 * @param column
 */
void GetImVector(floatt* vector, uintt length, math::Matrix* matrix,
                 uintt column);

/**
 * @brief GetTransposeImVector
 * @param vector
 * @param length
 * @param matrix
 * @param row
 */
void GetTransposeImVector(floatt* vector, uintt length, math::Matrix* matrix,
                          uintt row);

/**
 * @brief GetReVector
 * @param vector
 * @param matrix
 * @param column
 */
void GetReVector(floatt* vector, math::Matrix* matrix, uintt column);

/**
 * @brief GetTransposeReVector
 * @param vector
 * @param matrix
 * @param row
 */
void GetTransposeReVector(floatt* vector, math::Matrix* matrix, uintt row);

/**
 * @brief GetImVector
 * @param vector
 * @param matrix
 * @param column
 */
void GetImVector(floatt* vector, math::Matrix* matrix, uintt column);

/**
 * @brief GetTransposeImVector
 * @param vector
 * @param matrix
 * @param row
 */
void GetTransposeImVector(floatt* vector, math::Matrix* matrix, uintt row);

/**
 * @brief SetIdentity
 * @param matrix
 */
void SetIdentity(math::Matrix* matrix);

/**
 * @brief SetIdentityMatrix
 * @param matrix
 */
void SetIdentityMatrix(math::Matrix* matrix);

/**
 * @brief SmallestDiff
 * @param matrix
 * @param matrix1
 * @return
 */
floatt SmallestDiff(math::Matrix* matrix, math::Matrix* matrix1);

/**
 * @brief LargestDiff
 * @param matrix
 * @param matrix1
 * @return
 */
floatt LargestDiff(math::Matrix* matrix, math::Matrix* matrix1);

/**
 * @brief GetTrace
 * @param matrix
 * @return
 */
floatt GetTrace(math::Matrix* matrix);

/**
 * @brief SetReZero
 * @param matrix
 */
void SetReZero(math::Matrix* matrix);

/**
 * @brief SetImZero
 * @param matrix
 */
void SetImZero(math::Matrix* matrix);

/**
 * @brief SetZero
 * @param matrix
 */
void SetZero(math::Matrix* matrix);

/**
 * @brief IsEquals
 * @param matrix
 * @param matrix1
 * @param diff
 * @return
 */
bool IsEquals(math::Matrix* matrix, math::Matrix* matrix1, floatt diff = 0.1);

/**
 * @brief SetSubs
 * @param matrix
 * @param subcolumns
 * @param subrows
 */
void SetSubs(math::Matrix* matrix, uintt subcolumns, uintt subrows);

/**
 * @brief SetSubColumns
 * @param matrix
 * @param subcolumns
 */
void SetSubColumns(math::Matrix* matrix, uintt subcolumns);

/**
 * @brief SetSubRows
 * @param matrix
 * @param subrows
 */
void SetSubRows(math::Matrix* matrix, uintt subrows);

/**
 * @brief SetSubsSafe
 * @param matrix
 * @param subcolumns
 * @param subrows
 */
void SetSubsSafe(math::Matrix* matrix, uintt subcolumns, uintt subrows);

/**
 * @brief SetSubColumnsSafe
 * @param matrix
 * @param subcolumns
 */
void SetSubColumnsSafe(math::Matrix* matrix, uintt subcolumns);

/**
 * @brief SetSubRowsSafe
 * @param matrix
 * @param subrows
 */
void SetSubRowsSafe(math::Matrix* matrix, uintt subrows);

/**
 * @brief SetDiagonalMatrix
 * @param matrix
 * @param a
 */
void SetDiagonalMatrix(math::Matrix* matrix, floatt a);

/**
 * @brief SetDiagonalReMatrix
 * @param matrix
 * @param a
 */
void SetDiagonalReMatrix(math::Matrix* matrix, floatt a);

/**
 * @brief SetDiagonalImMatrix
 * @param matrix
 * @param a
 */
void SetDiagonalImMatrix(math::Matrix* matrix, floatt a);

math::MatrixInfo GetMatrixInfo(const math::Matrix* matrix);

math::Matrix* ReadMatrix(const std::string& path);

math::Matrix* ReadRowVector(const std::string& path, size_t index);

bool WriteMatrix(const std::string& path, const math::Matrix* matrix);
};

class HostMatrixUtils : public MatrixUtils {
 public:
  HostMatrixUtils();
  ~HostMatrixUtils();
  void getReValues(floatt* dst, math::Matrix* matrix, uintt index,
                   uintt length);
  void getImValues(floatt* dst, math::Matrix* matrix, uintt index,
                   uintt length);
  void setReValues(math::Matrix* matrix, floatt* src, uintt index,
                   uintt length);
  void setImValues(math::Matrix* matrix, floatt* src, uintt index,
                   uintt length);
  void setDiagonalReMatrix(math::Matrix* matrix, floatt a);
  void setDiagonalImMatrix(math::Matrix* matrix, floatt a);
  void setZeroReMatrix(math::Matrix* matrix);
  void setZeroImMatrix(math::Matrix* matrix);
  uintt getColumns(const math::Matrix* matrix) const;
  uintt getRows(const math::Matrix* matrix) const;
  bool isMatrix(const math::Matrix* matrix) const;
  bool isReMatrix(const math::Matrix* matrix) const;
  bool isImMatrix(const math::Matrix* matrix) const;
};

class HostMatrixCopier : public MatrixCopier {
  HostMatrixUtils hmu;

 public:
  HostMatrixCopier();
  virtual ~HostMatrixCopier();
  void copyMatrixToMatrix(math::Matrix* dst, const math::Matrix* src);
  void copyReMatrixToReMatrix(math::Matrix* dst, const math::Matrix* src);
  void copyImMatrixToImMatrix(math::Matrix* dst, const math::Matrix* src);
  /**
   * Copy floatts where length is number of floatts (not bytes!).
   * @param dst
   * @param src
   * @param length number of numbers to copy
   */
  void copy(floatt* dst, const floatt* src, uintt length);

  void setReVector(math::Matrix* matrix, uintt column, floatt* vector,
                   uintt length);
  void setTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector,
                            uintt length);
  void setImVector(math::Matrix* matrix, uintt column, floatt* vector,
                   uintt length);
  void setTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector,
                            uintt length);

  void getReVector(floatt* vector, uintt length, math::Matrix* matrix,
                   uintt column);
  void getTransposeReVector(floatt* vector, uintt length, math::Matrix* matrix,
                            uintt row);
  void getImVector(floatt* vector, uintt length, math::Matrix* matrix,
                   uintt column);
  void getTransposeImVector(floatt* vector, uintt length, math::Matrix* matrix,
                            uintt row);
  void setVector(math::Matrix* matrix, uintt column, math::Matrix* vector,
                 uintt rows);
  void getVector(math::Matrix* vector, uintt rows, math::Matrix* matrix,
                 uintt column);
};

class HostMatrixAllocator : public MatrixAllocator {
  HostMatrixUtils hmu;
  HostMatrixCopier hmc;
  typedef std::vector<math::Matrix*> HostMatrices;
  static HostMatrices hostMatrices;
  static utils::sync::Mutex mutex;
  static math::Matrix* createHostMatrix(math::Matrix* matrix, uintt columns,
                                        uintt rows, floatt* values,
                                        floatt** valuesPtr);
  static void initMatrix(math::Matrix* matrix);
  static math::Matrix* createHostReMatrix(uintt columns, uintt rows,
                                          floatt* values);
  static math::Matrix* createHostImMatrix(uintt columns, uintt rows,
                                          floatt* values);

 public:
  HostMatrixAllocator();
  ~HostMatrixAllocator();
  math::Matrix* newReMatrix(uintt columns, uintt rows, floatt value = 0);
  math::Matrix* newImMatrix(uintt columns, uintt rows, floatt value = 0);
  math::Matrix* newMatrix(uintt columns, uintt rows, floatt value = 0);
  bool isMatrix(math::Matrix* matrix);
  math::Matrix* newMatrixFromAsciiFile(const char* path);
  math::Matrix* newMatrixFromBinaryFile(const char* path);
  void deleteMatrix(math::Matrix* matrix);
};

class HostMatrixPrinter : public MatrixPrinter {
 public:
  HostMatrixPrinter();
  ~HostMatrixPrinter();
  void getMatrixStr(std::string& str, const math::Matrix* matrix);
  void getReMatrixStr(std::string& str, const math::Matrix* matrix);
  void getImMatrixStr(std::string& str, const math::Matrix* matrix);
};

class HostMatrixModules : public MatrixModule {
  HostMatrixAllocator* m_hma;
  HostMatrixCopier* m_hmc;
  HostMatrixUtils* m_hmu;
  HostMatrixPrinter* m_hmp;
  static HostMatrixModules* hostMatrixModule;

 protected:
  HostMatrixModules();
  virtual ~HostMatrixModules();

 public:
  static HostMatrixModules* GetInstance();
  HostMatrixAllocator* getMatrixAllocator();
  HostMatrixCopier* getMatrixCopier();
  HostMatrixUtils* getMatrixUtils();
  HostMatrixPrinter* getMatrixPrinter();
};

#endif /* MATRIXALLOCATOR_H */
