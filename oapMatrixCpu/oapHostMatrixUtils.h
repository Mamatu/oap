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

#ifndef OAP_HOST_MATRIX_UTILS_H
#define OAP_HOST_MATRIX_UTILS_H

#include "Matrix.h"

#include "oapGenericMatrixApi.h"

#define PRINT_MATRIX(m) logInfo ("%s %p\n%s %s", #m, m, oap::host::to_string(m).c_str(), oap::host::GetMatrixInfo(m).toString().c_str());
#define PRINT_DIMS_3_2(m) logInfo ("%s dims = {{%u, %u}, {%u, %u}, {%u, %u}} ", #m, m[0][0], m[0][1], m[1][0], m[1][1], m[2][0], m[2][1]);
#define PRINT_DIMS_2_2_2(m) logInfo ("%s dims = {{{%u, %u}, {%u, %u}}, {{%u, %u}, {%u, %u}}} ", #m, m[0][0][0], m[0][0][1], m[0][1][0], m[0][1][1], m[1][0][0], m[1][0][1], m[1][1][0], m[1][1][1]);
#define PRINT_MATRIX_CARRAY(m) logInfo ("%s", oap::host::to_carraystr(m).c_str());

namespace oap
{
namespace host
{

/**
 * @brief NewMatrix
 * @param columns
 * @param rows
 * @param value
 * @return
 */
math::ComplexMatrix* NewMatrix (uintt columns, uintt rows);

math::ComplexMatrix* NewMatrixCopyOfArray (uintt columns, uintt rows, const floatt* rearray, const floatt* imarray);
math::ComplexMatrix* NewReMatrixCopyOfArray (uintt columns, uintt rows, const floatt* rearray);
math::ComplexMatrix* NewImMatrixCopyOfArray (uintt columns, uintt rows, const floatt* imarray);

math::ComplexMatrix* NewShareMatrix (uintt columns, uintt rows, math::ComplexMatrix* src);

/**
 * @brief NewMatrix - creates new matrix, which has the same size
 *                    and im and re part like passed matrix. Values
 *                    of output are specyfied by user.
 * @param matrix
 * @param value
 * @return
 */
math::ComplexMatrix* NewMatrixRef(const math::ComplexMatrix* matrix);

/**
 * @brief NewMatrix
 * @param matrix
 * @param columns
 * @param rows
 * @param value
 * @return
 */
math::ComplexMatrix* NewMatrix (const math::ComplexMatrix* matrix, uintt columns, uintt rows);


math::ComplexMatrix* NewMatrix(const math::MatrixInfo& matrixInfo);

/**
 * @brief NewMatrix
 * @param isre
 * @param isim
 * @param columns
 * @param rows
 * @param value
 * @return
 */
math::ComplexMatrix* NewMatrix (bool isre, bool isim, uintt columns, uintt rows);

math::ComplexMatrix* NewMatrixWithValue (bool isre, bool isim, uintt columns, uintt rows, floatt value);

inline math::ComplexMatrix* NewHostMatrix (bool isre, bool isim, uintt columns, uintt rows)
{
  return NewMatrix (isre, isim, columns, rows);
}

inline math::ComplexMatrix* NewHostReMatrix (uintt columns, uintt rows)
{
  return NewMatrix (true, false, columns, rows);
}

inline math::ComplexMatrix* NewHostMatrixFromMatrixInfo (const math::MatrixInfo& minfo)
{
  return NewMatrix (minfo.isRe, minfo.isIm, minfo.columns(), minfo.rows());
}

/**
 * @brief NewReMatrix
 * @param columns
 * @param rows
 * @param value
 * @return
 */
math::ComplexMatrix* NewReMatrix(uintt columns, uintt rows);

/**
 * @brief NewImMatrix
 * @param columns
 * @param rows
 * @param value
 * @return
 */
math::ComplexMatrix* NewImMatrix(uintt columns, uintt rows);

/**
 * @brief NewMatrix
 * @param text
 * @return
 */

math::ComplexMatrix* NewMatrixWithValue (const math::MatrixInfo& minfo, floatt value);

math::ComplexMatrix* NewMatrixWithValue (uintt columns, uintt rows, floatt value);
math::ComplexMatrix* NewReMatrixWithValue (uintt columns, uintt rows, floatt value);
math::ComplexMatrix* NewImMatrixWithValue (uintt columns, uintt rows, floatt value);

math::ComplexMatrix* NewMatrix(const std::string& text);

inline void CopyBuffer(floatt* dst, floatt* src, uintt length)
{
  memcpy(dst, src, length * sizeof(floatt));
}

/**
 * @brief CopyMatrix
 * @param dst
 * @param src
 */
void CopyMatrix(math::ComplexMatrix* dst, const math::ComplexMatrix* src);
void CopyMatrixRegion (math::ComplexMatrix* dst, const oap::MemoryLoc& dstLoc, const math::ComplexMatrix* src, const oap::MemoryRegion& srcReg);

inline void CopyHostMatrixToHostMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
  CopyMatrix (dst, src);
}

inline void CopyHostMatrixToHostMatrixRegion (math::ComplexMatrix* dst, const oap::MemoryLoc& dstLoc, const math::ComplexMatrix* src, const oap::MemoryRegion& srcReg)
{
  CopyMatrixRegion (dst, dstLoc, src, srcReg);
}

#if 0
void CopyMatrixDims (math::ComplexMatrix* dst, const math::ComplexMatrix* src, uintt dims[2][2][2]);

inline void CopyHostMatrixToHostMatrixDims (math::ComplexMatrix* dst, const math::ComplexMatrix* src, uintt dims[2][2][2])
{
  CopyMatrixDims (dst, src, dims);
}
#endif
inline uintt GetColumns (const math::ComplexMatrix* matrix)
{
  return gColumns (matrix);
}

inline uintt GetRows (const math::ComplexMatrix* matrix)
{
  return gRows (matrix);
}

/**
 * Copy data to dst matrix which has one column and row less than
 * src matrix. Row and column which will be omitted are added as params..
 * @param dst
 * @param src
 * @param column index of column which will be omitted
 * @param row index of row which will be omitted
 */
void Copy(math::ComplexMatrix* dst, const math::ComplexMatrix* src, uintt column, uintt row);

/**
 * @brief Copy
 * @param dst
 * @param src
 * @param MatrixEx
 */
void Copy(math::ComplexMatrix* dst, const math::ComplexMatrix* src, const MatrixEx& matrixEx);

/**
 * @brief CopyRe
 * @param dst
 * @param src
 */
void CopyRe(math::ComplexMatrix* dst, const math::ComplexMatrix* src);

/**
 * @brief CopyIm
 * @param dst
 * @param src
 */
void CopyIm(math::ComplexMatrix* dst, const math::ComplexMatrix* src);

template<typename T>
void Copy(floatt* dst, T* src, uint length)
{
  debugFunc();
  if (dst == NULL || src == NULL)
  {
    return;
  }
  for (uint idx = 0; idx < length; ++idx)
  {
    dst[idx] = src[idx];
  }
}

template<>
inline void Copy<floatt>(floatt* dst, floatt* src, uint length)
{
  debugFunc();
  if (dst == NULL || src == NULL)
  {
    return;
  }
  memcpy(dst, src, sizeof(floatt) * length);
}

template<typename T>
void Copy(math::ComplexMatrix* dst, T* rearray, T* imarray)
{
  Copy(gReValues (dst), rearray, gColumns (dst) * gRows (dst));
  Copy(gImValues (dst), imarray, gColumns (dst) * gRows (dst));
}

template<typename T>
void CopyRe(math::ComplexMatrix* dst, T* array)
{
  Copy(gReValues (dst), array, gColumns (dst) * gRows (dst));
}

template<typename T>
void CopyIm(math::ComplexMatrix* dst, T* array)
{
  Copy(gImValues (dst), array, gColumns (dst) * gRows (dst));
}

/**
 * @brief NewMatrixCopy
 * @param matrix
 * @return
 */
math::ComplexMatrix* NewMatrixCopy(const math::ComplexMatrix* matrix);

template<typename T>
math::ComplexMatrix* NewMatrixCopy(uint columns, uint rows, T* reArray, T* imArray)
{
  math::ComplexMatrix* output = oap::host::NewMatrix(reArray != NULL, imArray != NULL, columns, rows);
  oap::host::Copy<T>(output, reArray, imArray);
  return output;
}

template<typename T>
math::ComplexMatrix* NewReMatrixCopy(uint columns, uint rows, T* reArray)
{
  math::ComplexMatrix* output = oap::host::NewMatrix(reArray != NULL, false, columns, rows);
  oap::host::CopyRe<T>(output, reArray);
  return output;
}

template<typename T>
math::ComplexMatrix* NewImMatrixCopy(uint columns, uint rows, T* imArray)
{
  math::ComplexMatrix* output = oap::host::NewMatrix(false, imArray != NULL, columns, rows);
  oap::host::CopyIm<T>(output, imArray);
  return output;
}

/**
 * @brief DeleteMatrix
 * @param matrix
 */
void DeleteMatrix(const math::ComplexMatrix* matrix);

void DeleteComplexMatrix(const math::ComplexMatrix* matrix);

void DeleteMatrix(const math::Matrix* matrix);

template<typename Matrices>
void deleteMatrices(const Matrices& matrices)
{
  for (uintt idx = 0; idx < matrices.size(); ++idx)
  {
    const math::ComplexMatrix* matrix = matrices[idx];
    DeleteMatrix (matrix);
  }
}

/**
 * @brief GetReValue
 * @param matrix
 * @param column
 * @param row
 * @return
 */
floatt GetReValue(const math::ComplexMatrix* matrix, uintt column, uintt row);

/**
 * @brief SetReValue
 * @param matrix
 * @param column
 * @param row
 * @param value
 */
void SetReValue(const math::ComplexMatrix* matrix, uintt column, uintt row,
                floatt value);

/**
 * @brief GetImValue
 * @param matrix
 * @param column
 * @param row
 * @return
 */
floatt GetImValue(const math::ComplexMatrix* matrix, uintt column, uintt row);

/**
 * @brief SetImValue
 * @param matrix
 * @param column
 * @param row
 * @param value
 */
void SetImValue(const math::ComplexMatrix* matrix, uintt column, uintt row, floatt value);

std::string GetMatrixStr(const math::ComplexMatrix* matrix);
math::ComplexMatrix GetRefHostMatrix (const math::ComplexMatrix* matrix);

inline std::string to_string (const math::ComplexMatrix* matrix)
{
  return GetMatrixStr (matrix);
}

void ToString(std::string& str, const math::ComplexMatrix* matrix);

void PrintMatrix(FILE* stream, const matrixUtils::PrintArgs& args, const math::ComplexMatrix* matrix);
void PrintMatrix(FILE* stream, const math::ComplexMatrix* matrix, const matrixUtils::PrintArgs& args = matrixUtils::PrintArgs());

void PrintMatrix(const matrixUtils::PrintArgs& args, const math::ComplexMatrix* matrix);
void PrintMatrix(const math::ComplexMatrix* matrix, const matrixUtils::PrintArgs& args = matrixUtils::PrintArgs());

bool PrintMatrixToFile(const std::string& path, const matrixUtils::PrintArgs& args, const math::ComplexMatrix* matrix);
bool PrintMatrixToFile(const std::string& path, const math::ComplexMatrix* matrix, const matrixUtils::PrintArgs& args = matrixUtils::PrintArgs());

/**
 * @brief PrintImMatrix
 * @param stream
 * @param matrix
 */
void PrintImMatrix(FILE* stream, const math::ComplexMatrix* matrix);

/**
 * @brief PrintImMatrix
 * @param matrix
 */
void PrintImMatrix(const math::ComplexMatrix* matrix);

/**
 * @brief PrintImMatrix
 * @param text
 * @param matrix
 */
void PrintImMatrix(const std::string& text, const math::ComplexMatrix* matrix);

void SetVector(math::ComplexMatrix* matrix, uintt column, math::ComplexMatrix* vector);

void SetVector(math::ComplexMatrix* matrix, uintt column, floatt* revector,
               floatt* imvector, uintt length);

/**
 * @brief SetReVector
 * @param matrix
 * @param column
 * @param vector
 * @param length
 */
void SetReVector(math::ComplexMatrix* matrix, uintt column, floatt* vector,
                 uintt length);

/**
 * @brief SetTransposeReVector
 * @param matrix
 * @param row
 * @param vector
 * @param length
 */
void SetTransposeReVector(math::ComplexMatrix* matrix, uintt row, floatt* vector,
                          uintt length);

/**
 * @brief SetImVector
 * @param matrix
 * @param column
 * @param vector
 * @param length
 */
void SetImVector(math::ComplexMatrix* matrix, uintt column, floatt* vector,
                 uintt length);

/**
 * @brief SetTransposeImVector
 * @param matrix
 * @param row
 * @param vector
 * @param length
 */
void SetTransposeImVector(math::ComplexMatrix* matrix, uintt row, floatt* vector,
                          uintt length);

/**
 * @brief SetReVector
 * @param matrix
 * @param column
 * @param vector
 */
void SetReVector(math::ComplexMatrix* matrix, uintt column, floatt* vector);

/**
 * @brief SetTransposeReVector
 * @param matrix
 * @param row
 * @param vector
 */
void SetTransposeReVector(math::ComplexMatrix* matrix, uintt row, floatt* vector);

/**
 * @brief SetImVector
 * @param matrix
 * @param column
 * @param vector
 */
void SetImVector(math::ComplexMatrix* matrix, uintt column, floatt* vector);

/**
 * @brief SetTransposeImVector
 * @param matrix
 * @param row
 * @param vector
 */
void SetTransposeImVector(math::ComplexMatrix* matrix, uintt row, floatt* vector);

void GetVector(math::ComplexMatrix* vector, math::ComplexMatrix* matrix, uintt column);

void GetVector(floatt* revector, floatt* imvector, uint length,
               math::ComplexMatrix* matrix, uint column);

void GetTransposeVector(math::ComplexMatrix* vector, math::ComplexMatrix* matrix, uint column);

void GetTransposeReVector(math::ComplexMatrix* vector, math::ComplexMatrix* matrix, uint column);

void GetTransposeImVector(math::ComplexMatrix* vector, math::ComplexMatrix* matrix, uint column);

/**
 * @brief GetReVector
 * @param vector
 * @param length
 * @param matrix
 * @param column
 */
void GetReVector(floatt* vector, uint length, math::ComplexMatrix* matrix, uint column);

/**
 * @brief GetTransposeReVector
 * @param vector
 * @param length
 * @param matrix
 * @param row
 */
void GetTransposeReVector(floatt* vector, uint length, math::ComplexMatrix* matrix, uint row);

/**
 * @brief GetImVector
 * @param vector
 * @param length
 * @param matrix
 * @param column
 */
void GetImVector(floatt* vector, uint length, math::ComplexMatrix* matrix, uint column);

/**
 * @brief GetTransposeImVector
 * @param vector
 * @param length
 * @param matrix
 * @param row
 */
void GetTransposeImVector(floatt* vector, uint length, math::ComplexMatrix* matrix, uint row);

/**
 * @brief GetReVector
 * @param vector
 * @param matrix
 * @param column
 */
void GetReVector(floatt* vector, math::ComplexMatrix* matrix, uint column);

/**
 * @brief GetTransposeReVector
 * @param vector
 * @param matrix
 * @param row
 */
void GetTransposeReVector(floatt* vector, math::ComplexMatrix* matrix, uint row);

/**
 * @brief GetImVector
 * @param vector
 * @param matrix
 * @param column
 */
void GetImVector(floatt* vector, math::ComplexMatrix* matrix, uint column);

/**
 * @brief GetTransposeImVector
 * @param vector
 * @param matrix
 * @param row
 */
void GetTransposeImVector(floatt* vector, math::ComplexMatrix* matrix, uint row);

/**
 * @brief SetIdentity
 * @param matrix
 */
void SetIdentity(math::ComplexMatrix* matrix);

/**
 * @brief SetIdentityMatrix
 * @param matrix
 */
void SetIdentityMatrix(math::ComplexMatrix* matrix);

/**
 * @brief SmallestDiff
 * @param matrix
 * @param matrix1
 * @return
 */
floatt SmallestDiff(math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1);

/**
 * @brief LargestDiff
 * @param matrix
 * @param matrix1
 * @return
 */
floatt LargestDiff(math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1);

/**
 * @brief GetTrace
 * @param matrix
 * @return
 */
floatt GetTrace(math::ComplexMatrix* matrix);

/**
 * @brief SetReZero
 * @param matrix
 */
void SetReZero(math::ComplexMatrix* matrix);

/**
 * @brief SetImZero
 * @param matrix
 */
void SetImZero(math::ComplexMatrix* matrix);

/**
 * @brief SetZero
 * @param matrix
 */
void SetZero(math::ComplexMatrix* matrix);

/**
 * @brief IsEquals
 * @param matrix
 * @param matrix1
 * @param diff
 * @return
 */
bool IsEquals(math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, floatt diff = 0.1);

/**
 * @brief SetSubs
 * @param matrix
 * @param subcolumns
 * @param subrows
 */
void SetSubs(math::ComplexMatrix* matrix, uintt subcolumns, uintt subrows);

/**
 * @brief SetSubColumns
 * @param matrix
 * @param subcolumns
 */
void SetSubColumns(math::ComplexMatrix* matrix, uintt subcolumns);

/**
 * @brief SetSubRows
 * @param matrix
 * @param subrows
 */
void SetSubRows(math::ComplexMatrix* matrix, uintt subrows);

/**
 * @brief SetSubsSafe
 * @param matrix
 * @param subcolumns
 * @param subrows
 */
void SetSubsSafe(math::ComplexMatrix* matrix, uintt subcolumns, uintt subrows);

/**
 * @brief SetSubColumnsSafe
 * @param matrix
 * @param subcolumns
 */
void SetSubColumnsSafe(math::ComplexMatrix* matrix, uintt subcolumns);

/**
 * @brief SetSubRowsSafe
 * @param matrix
 * @param subrows
 */
void SetSubRowsSafe(math::ComplexMatrix* matrix, uintt subrows);

/**
 * @brief SetDiagonalMatrix
 * @param matrix
 * @param a
 */
void SetDiagonalMatrix(math::ComplexMatrix* matrix, floatt a);

/**
 * @brief SetDiagonalReMatrix
 * @param matrix
 * @param a
 */
void SetDiagonalReMatrix(math::ComplexMatrix* matrix, floatt a);

/**
 * @brief SetDiagonalImMatrix
 * @param matrix
 * @param a
 */
void SetDiagonalImMatrix(math::ComplexMatrix* matrix, floatt a);

math::MatrixInfo CreateMatrixInfo (const math::ComplexMatrix* matrix);

math::MatrixInfo GetMatrixInfo (const math::ComplexMatrix* matrix);

math::ComplexMatrix* ReadMatrix(const std::string& path);

math::ComplexMatrix* ReadRowVector(const std::string& path, size_t index);

math::ComplexMatrix* ReadColumnVector(const std::string& path, size_t index);

void CopyReBuffer (math::ComplexMatrix* houput, math::ComplexMatrix* hinput);

void SetZeroRow (const math::ComplexMatrix* matrix, uintt index, bool re = true, bool im = true);
void SetReZeroRow (const math::ComplexMatrix* matrix, uintt index);
void SetImZeroRow (const math::ComplexMatrix* matrix, uintt index);

void SetValueToMatrix (math::ComplexMatrix* matrix, floatt re, floatt im);
void SetValueToReMatrix (math::ComplexMatrix* matrix, floatt v);
void SetValueToImMatrix (math::ComplexMatrix* matrix, floatt v);

void SetValueToMatrix (math::Matrix* matrix, floatt v);

void SetZeroMatrix (math::ComplexMatrix* matrix);
void SetZeroReMatrix (math::ComplexMatrix* matrix);
void SetZeroImMatrix (math::ComplexMatrix* matrix);

void SetZeroMatrix (math::Matrix* matrix);

/**
 * @brief Save matrix to file
 * 4 byte - size of boolean variable
 * 4 byte - size of uintt variable (where uintt - oap specyfied during compilation type)
 * 4 byte - size of floatt variable (where floatt - oap specyfied during compilation type)
 * size of uintt - count of columnts
 * size of uintt - count of rows
 * size of boolean - is re section
 * size of boolean - is im section
 * if the first boolean is true:
 *  columns * rows * sizeoffloatt - re part of matrix
 * if the second boolean is true:
 *  columns * rows * sizeoffloatt - im part of matrix
 */
bool WriteMatrix(const std::string& path, const math::ComplexMatrix* matrix);

inline bool IsReMatrix(math::ComplexMatrix* m)
{
  return m != NULL && gReValues (m) != NULL;
}

inline bool IsImMatrix(math::ComplexMatrix* m)
{
  return m != NULL && gImValues (m) != NULL;
}

inline void ToHost (void* dst, const void* src, size_t size)
{
  memcpy (dst, src, size);
}

void CopySubMatrix(math::ComplexMatrix* dst, const math::ComplexMatrix* src, uintt cindex, uintt rindex);

math::ComplexMatrix* NewSubMatrix (const math::ComplexMatrix* orig, uintt cindex, uintt rindex, uintt clength, uintt rlength);

math::ComplexMatrix* GetSubMatrix (const math::ComplexMatrix* orig, uintt cindex, uintt rindex, math::ComplexMatrix* matrix);

void SaveMatrix (const math::ComplexMatrix* matrix, utils::ByteBuffer& buffer);
void SaveMatrixInfo (const math::MatrixInfo& minfo, utils::ByteBuffer& buffer);

math::ComplexMatrix* LoadMatrix (const utils::ByteBuffer& buffer);
math::MatrixInfo LoadMatrixInfo (const utils::ByteBuffer& buffer);

void CopyArrayToMatrix (math::ComplexMatrix* matrix, const floatt* rebuffer, const floatt* imbuffer);
void CopyArrayToReMatrix (math::ComplexMatrix* matrix, const floatt* buffer);
void CopyArrayToImMatrix (math::ComplexMatrix* matrix, const floatt* buffer);

inline void SetReValueToMatrix (math::ComplexMatrix* matrix, floatt value, size_t idx = 0)
{
  gReValues (matrix)[idx] = value;
}

inline void SetReValuesToMatrix (math::ComplexMatrix* matrix, const std::vector<floatt>& vec)
{
  debugAssert (vec.size() <= gColumns (matrix) * gRows (matrix));
  memcpy (gReValues (matrix), vec.data(), sizeof(floatt) * vec.size());
}

inline void SetReValuesToMatrix (math::ComplexMatrix* matrix, floatt* array, size_t length)
{
  debugAssert (length <= gColumns (matrix) * gRows (matrix));
  memcpy (gReValues (matrix), array, sizeof(floatt) * length);
}

template<typename Tuple, size_t Tidx = 0>
void SetReValuesToMatrix (math::ComplexMatrix* matrix, const std::vector<Tuple>& vecl)
{
  std::vector<floatt> vec;
  vec.reserve (vecl.size());
  for (auto it = vecl.begin(); it != vecl.end(); ++it)
  {
    vec.push_back (std::get<Tidx>(*it));
  }

  SetReValuesToMatrix (matrix, vec.data(), vec.size());
}

void CopyHostArrayToHostMatrix (math::ComplexMatrix* matrix, const floatt* rebuffer, const floatt* imbuffer, size_t length);

void CopyHostArrayToHostReMatrix (math::ComplexMatrix* matrix, const floatt* buffer, size_t length);

void CopyHostArrayToHostImMatrix (math::ComplexMatrix* matrix, const floatt* buffer, size_t length);

void SetMatrix (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt column, uintt row);
void SetReMatrix (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt column, uintt row);
void SetImMatrix (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt column, uintt row);

template<typename Matrices>
oap::ThreadsMapper createThreadsMapper (const std::vector<Matrices>& matricesVec, oap::threads::ThreadsMapperAlgo algo)
{
  return oap::threads::createThreadsMapper (matricesVec, oap::host::GetRefHostMatrix, malloc, memcpy, free, algo);
}

oap::ThreadsMapper CreateThreadsMapper (const std::vector<std::vector<math::ComplexMatrix*>>& matrices, oap::threads::ThreadsMapperAlgo algo);

math::ComplexMatrix* NewMatrixFromMemory (uintt columns, uintt rows, oap::Memory& rememory, const oap::MemoryLoc& reloc, oap::Memory& immemory, const oap::MemoryLoc& imloc);
math::ComplexMatrix* NewReMatrixFromMemory (uintt columns, uintt rows, oap::Memory& memory, const oap::MemoryLoc& loc);
math::ComplexMatrix* NewImMatrixFromMemory (uintt columns, uintt rows, oap::Memory& memory, const oap::MemoryLoc& loc);

floatt GetReDiagonal (const math::ComplexMatrix* matrix, uintt index);

floatt GetImDiagonal (const math::ComplexMatrix* matrix, uintt index);

void CopyReMatrixToHostBuffer (floatt* buffer, uintt length, const math::ComplexMatrix* matrix);
void CopyHostBufferToReMatrix (math::ComplexMatrix* matrix, const floatt* buffer, uintt length);

std::string to_carraystr(const math::ComplexMatrix* matrix);
std::string to_carraystr(const std::vector<math::ComplexMatrix*>& matrices);

std::vector<math::ComplexMatrix*> NewMatrices (const std::vector<math::MatrixInfo>& minfos);
std::vector<math::ComplexMatrix*> NewMatrices (const math::MatrixInfo& minfo, uintt count);
std::vector<math::ComplexMatrix*> NewMatricesCopyOfArray (const std::vector<math::MatrixInfo>& minfos, const std::vector<std::vector<floatt>>& arrays);
std::vector<math::ComplexMatrix*> NewMatricesCopyOfArray(const math::MatrixInfo& minfo, const std::vector<std::vector<floatt>>& arrays);

/**
 * @param loc - location in matrix (@see param matrix)
 * @param dim - dimension of new matrix
 * @param matrix - original matrix whose memory will be shared with constructed matrix
 */
math::ComplexMatrix* NewSharedSubMatrix (const math::MatrixLoc& loc, const math::MatrixDim& dim, const math::ComplexMatrix* matrix);
math::ComplexMatrix* NewSharedSubMatrix (const math::MatrixDim& dim, const math::ComplexMatrix* matrix);

}
}

#endif /* OAP_HOST_MATRIX_UTILS_H */
