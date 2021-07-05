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

#ifndef OAP_CUDA_MATRIX_UTILS_H
#define OAP_CUDA_MATRIX_UTILS_H

#include "ByteBuffer.hpp"

#include "Matrix.hpp"
#include "MatrixEx.hpp"

#include "oapThreadsMapperApi.hpp"
#include "oapHostComplexMatrixApi.hpp"
#include "ThreadUtils.hpp"
#include "CudaUtils.hpp"

#define PRINT_CUMATRIX(m) logInfo ("%s %p\n%s %s", #m, m, oap::cuda::to_string(m).c_str(), oap::cuda::GetMatrixInfo(m).toString().c_str());
#define PRINT_CUMATRIX_CARRAY(m) logInfo ("%s", oap::cuda::to_carraystr(m).c_str());

namespace oap
{
namespace cuda
{

math::ComplexMatrix* NewDeviceMatrix (uintt columns, uintt rows);
math::ComplexMatrix* NewDeviceMatrixWithValue (uintt columns, uintt rows, floatt v);
math::ComplexMatrix* NewDeviceReMatrixWithValue (uintt columns, uintt rows, floatt v);
math::ComplexMatrix* NewDeviceImMatrixWithValue (uintt columns, uintt rows, floatt v);

math::ComplexMatrix* NewDeviceMatrixWithValue (bool isre, bool isim, uintt columns, uintt rows, floatt v);
math::ComplexMatrix* NewDeviceMatrixWithValue (const math::MatrixInfo& minfo, floatt v);

#if 0
math::ComplexMatrix* NewShareDeviceMatrix(uintt columns, uintt rows, math::ComplexMatrix* src);
#endif

math::ComplexMatrix* NewDeviceMatrixHostRef(const math::ComplexMatrix* hostMatrix);

math::ComplexMatrix* NewDeviceMatrixDeviceRef(const math::ComplexMatrix* deviceMatrix);

math::ComplexMatrix* NewDeviceMatrix(const std::string& matrixStr);

math::ComplexMatrix* NewDeviceMatrix(const math::MatrixInfo& minfo);

inline math::ComplexMatrix* NewDeviceMatrixFromMatrixInfo (const math::MatrixInfo& minfo)
{
  return NewDeviceMatrix (minfo);
}

math::ComplexMatrix* NewDeviceMatrixFromMemory (uintt columns, uintt rows, const oap::Memory& rememory, const oap::MemoryLoc& reloc, oap::Memory& immemory, const oap::MemoryLoc& imloc);
math::ComplexMatrix* NewDeviceReMatrixFromMemory (uintt columns, uintt rows, const oap::Memory& memory, const oap::MemoryLoc& loc);
math::ComplexMatrix* NewDeviceImMatrixFromMemory (uintt columns, uintt rows, const oap::Memory& memory, const oap::MemoryLoc& loc);

uintt GetColumns(const math::ComplexMatrix* dMatrix);

uintt GetRows(const math::ComplexMatrix* dMatrix);

math::ComplexMatrix GetRefHostMatrix (const math::ComplexMatrix* dMatrix);
oap::MemoryRegion GetReMemoryRegion (const math::ComplexMatrix* dMatrix);
oap::Memory GetReMemory (const math::ComplexMatrix* dMatrix);
oap::MemoryRegion GetImMemoryRegion (const math::ComplexMatrix* dMatrix);
oap::Memory GetImMemory (const math::ComplexMatrix* dMatrix);

floatt* GetReValuesPtr (const math::ComplexMatrix* dMatrix);
floatt* GetImValuesPtr (const math::ComplexMatrix* dMatrix);

math::ComplexMatrix* NewDeviceMatrixCopyOfHostMatrix(const math::ComplexMatrix* hostMatrix);

math::ComplexMatrix* NewDeviceMatrix(const math::ComplexMatrix* hostMatrix, uintt columns, uintt rows);

math::ComplexMatrix* NewDeviceMatrix(bool allocRe, bool allocIm, uintt columns, uintt rows);

inline math::ComplexMatrix* NewKernelMatrix (bool allocRe, bool allocIm, uintt columns, uintt rows)
{
  return NewDeviceMatrix (allocRe, allocIm, columns, rows);
}

math::ComplexMatrix* NewDeviceReMatrix(uintt columns, uintt rows);

math::ComplexMatrix* NewDeviceImMatrix(uintt columns, uintt rows);

math::ComplexMatrix* NewHostMatrixCopyOfDeviceMatrix(const math::ComplexMatrix* matrix);

void DeleteDeviceMatrix(const math::ComplexMatrix* deviceMatrix);

void DeleteDeviceComplexMatrix(const math::ComplexMatrix* deviceMatrix);

void DeleteDeviceMatrix(const math::Matrix* matrix);

template<typename Matrices>
void deleteDeviceMatrices(const Matrices& matrices)
{
  for (uintt idx = 0; idx < matrices.size(); ++idx)
  {
    const math::ComplexMatrix* matrix = matrices[idx];
    DeleteDeviceMatrix (matrix);
  }
}

/**
 * @brief CopyDeviceMatrixToHostMatrix - copy device to host - matrices must have the same columns and rows
 * @param dst - host matrix
 * @param src - device matrix
 */
void CopyDeviceMatrixToHostMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src);

/**
 * @brief copies host matrix to device matrix - copy host to device - matrices must have the same columns and rows
 * @param dst - host matrix
 * @param src - device matrix
 */
void CopyHostMatrixToDeviceMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src);

/**
 * @brief copies device matrix to device matrix - copy device to device - matrices must have the same columns and rows
 * @param dst - host matrix
 * @param src - device matrix
 */
void CopyDeviceMatrixToDeviceMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src);

/**
 * @brief copies davice matrix to host matrix - copy device to host - matrices must have the same product of columns and rows
 *                                              (dst->columns * dst->rows == src->columns * src->rows)
 * @param dst - host matrix
 * @param src - device matrix
 */
void CopyDeviceToHost(math::ComplexMatrix* dst, const math::ComplexMatrix* src);

/**
 * @brief copies host matrix to device matrix - copy device to host - matrices must have the same product of columns and rows
 *                                              (dst->columns * dst->rows == src->columns * src->rows)
 * @param dst - host matrix
 * @param src - device matrix
 */
void CopyHostToDevice(math::ComplexMatrix* dst, const math::ComplexMatrix* src);

/**
 * @brief copies device matri to device matrix - copy device to host - matrices must have the same product of columns and rows
 *                                               (dst->columns * dst->rows == src->columns * src->rows)
 * @param dst - host matrix
 * @param src - device matrix
 */
void CopyDeviceToDevice(math::ComplexMatrix* dst, const math::ComplexMatrix* src);

void CopyDeviceMatrixToHostMatrixEx (math::ComplexMatrix* dst, const oap::MemoryLoc& loc, const math::ComplexMatrix* src, const oap::MemoryRegion& reg);

void CopyHostMatrixToDeviceMatrixEx (math::ComplexMatrix* dst, const oap::MemoryLoc& loc, const math::ComplexMatrix* src, const oap::MemoryRegion& reg);

void CopyDeviceMatrixToDeviceMatrixEx (math::ComplexMatrix* dst, const oap::MemoryLoc& loc, const math::ComplexMatrix* src, const oap::MemoryRegion& reg);

void SetMatrix(math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt column, uintt row);
void SetReMatrix(math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt column, uintt row);
void SetImMatrix(math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt column, uintt row);

std::pair<floatt, floatt> GetDiagonal (const math::ComplexMatrix* matrix, uintt index);
floatt GetReDiagonal (const math::ComplexMatrix* matrix, uintt index);
floatt GetImDiagonal (const math::ComplexMatrix* matrix, uintt index);

void SetZeroRow (const math::ComplexMatrix* matrix, uintt index, bool re = true, bool im = true);
void SetReZeroRow (const math::ComplexMatrix* matrix, uintt index);
void SetImZeroRow (const math::ComplexMatrix* matrix, uintt index);

void SetValueToMatrix (math::ComplexMatrix* matrix, floatt re, floatt im);
void SetValueToReMatrix (math::ComplexMatrix* matrix, floatt v);
void SetValueToImMatrix (math::ComplexMatrix* matrix, floatt v);

void SetZeroMatrix (math::ComplexMatrix* matrix);
void SetZeroReMatrix (math::ComplexMatrix* matrix);
void SetZeroImMatrix (math::ComplexMatrix* matrix);

MatrixEx** NewDeviceMatrixEx(uintt count);

void CopyHostArrayToDeviceMatrix (math::ComplexMatrix* matrix, const floatt* rebuffer, const floatt* imbuffer, size_t length);
void CopyHostArrayToDeviceReMatrix (math::ComplexMatrix* matrix, const floatt* buffer, size_t length);
void CopyHostArrayToDeviceImMatrix (math::ComplexMatrix* matrix, const floatt* buffer, size_t length);

void CopyHostArrayToDeviceMatrixBuffer (math::ComplexMatrix* matrix, const floatt* rebuffer, const floatt* imbuffer, size_t length);
void CopyHostArrayToDeviceReMatrixBuffer (math::ComplexMatrix* matrix, const floatt* buffer, size_t length);
void CopyHostArrayToDeviceImMatrixBuffer (math::ComplexMatrix* matrix, const floatt* buffer, size_t length);

void DeleteDeviceMatrixEx(MatrixEx** matrixEx);

void SetMatrixEx(MatrixEx** deviceMatrixEx, const uintt* buffer, uintt count);

void SetMatrixEx(MatrixEx* deviceMatrixEx, const MatrixEx* hostMatrixEx);

void GetMatrixEx(MatrixEx* hostMatrixEx, const MatrixEx* deviceMatrixEx);

MatrixEx* NewDeviceMatrixEx();
MatrixEx* NewDeviceMatrixExCopy(const MatrixEx& hostMatrixEx);

void DeleteDeviceMatrixEx(MatrixEx* matrixEx);

void PrintMatrix(const std::string& text, const math::ComplexMatrix* matrix,
                 floatt zeroLimit = 0);

void PrintMatrix(const math::ComplexMatrix* matrix);

inline void ToHost (void* dst, const void* src, size_t size)
{
  return CudaUtils::ToHost (dst, src, size);
}

inline void TransferToHost (void* dst, const void* src, uintt size)
{
  memcpy (dst, src, size);
}

void SetReValue(math::ComplexMatrix* matrix, uintt column, uintt row, floatt value);
void SetReValue(math::ComplexMatrix* matrix, uintt index, floatt value);

void SetImValue(math::ComplexMatrix* matrix, uintt column, uintt row, floatt value);
void SetImValue(math::ComplexMatrix* matrix, uintt index, floatt value);

void SetValue(math::ComplexMatrix* matrix, uintt column, uintt row, floatt revalue, floatt imvalue);

void SetValue(math::ComplexMatrix* matrix, uintt index, floatt revalue, floatt imvalue);

math::MatrixInfo GetMatrixInfo(const math::ComplexMatrix* devMatrix);

bool IsCudaMatrix(const math::ComplexMatrix* devMatrix);

inline bool IsDeviceMatrix(const math::ComplexMatrix* devMatrix)
{
	return IsCudaMatrix (devMatrix);
}

void ToString (std::string& str, const math::ComplexMatrix* devMatrix);

inline std::string GetMatrixStr (const math::ComplexMatrix* devMatrix)
{
  std::string str;
  ToString (str, devMatrix);
  return str;
}

inline std::string to_string (const math::ComplexMatrix* devMatrix)
{
  return GetMatrixStr (devMatrix);
}

void PrintMatrixInfo(const std::string& msg, const math::ComplexMatrix* devMatrix);

bool WriteMatrix(const std::string& path, const math::ComplexMatrix* devMatrix);

math::ComplexMatrix* ReadMatrix(const std::string& path);

void SaveMatrix (const math::ComplexMatrix* matrix, utils::ByteBuffer& buffer);
void SaveMatrixInfo (const math::MatrixInfo& minfo, utils::ByteBuffer& buffer);

math::ComplexMatrix* LoadMatrix (const utils::ByteBuffer& buffer);
math::MatrixInfo LoadMatrixInfo (const utils::ByteBuffer& buffer);

template<typename MatricesLine>
oap::ThreadsMapper createThreadsMapper (const std::vector<MatricesLine>& matrices, oap::threads::ThreadsMapperAlgo algo)
{
  return oap::threads::createThreadsMapper (matrices, oap::cuda::GetRefHostMatrix, CudaUtils::Malloc, CudaUtils::CopyHostToDevice, CudaUtils::Free, algo);
}

oap::ThreadsMapper CreateThreadsMapper (const std::vector<std::vector<math::ComplexMatrix*>>& matrices, oap::threads::ThreadsMapperAlgo algo);

void CopyDeviceReMatrixToHostBuffer (floatt* buffer, uintt length, const math::ComplexMatrix* matrix);
void CopyHostBufferToDeviceReMatrix (math::ComplexMatrix* matrix, const floatt* buffer, uintt length);
void CopyDeviceBufferToDeviceReMatrix (math::ComplexMatrix* matrix, const floatt* buffer, uintt length);

std::string to_carraystr(const math::ComplexMatrix* matrix);
std::string to_carraystr(const std::vector<math::ComplexMatrix*>& matrices);

math::ComplexMatrix* NewDeviceReMatrixCopyOfArray(uintt columns, uintt rows, floatt* array);

std::vector<math::ComplexMatrix*> NewDeviceMatrices (const std::vector<math::MatrixInfo>& minfos);
std::vector<math::ComplexMatrix*> NewDeviceMatrices (const math::MatrixInfo& minfo, uintt count);
std::vector<math::ComplexMatrix*> NewDeviceMatricesCopyOfArray(const std::vector<math::MatrixInfo>& minfos, const std::vector<std::vector<floatt>>& arrays);
std::vector<math::ComplexMatrix*> NewDeviceMatricesCopyOfArray(const math::MatrixInfo& minfo, const std::vector<std::vector<floatt>>& arrays);

math::ComplexMatrix* NewDeviceSharedSubMatrix (const math::MatrixLoc& loc, const math::MatrixDim& dim, const math::ComplexMatrix* matrix);
math::ComplexMatrix* NewDeviceSharedSubMatrix (const math::MatrixDim& dim, const math::ComplexMatrix* matrix);

}
}

#endif /* MATRIXMEM_H */
