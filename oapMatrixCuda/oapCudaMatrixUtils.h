/*
 * Copyright 2016 - 2019 Marcin Matula
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

#include "ByteBuffer.h"

#include "Matrix.h"
#include "MatrixEx.h"

#include "ThreadUtils.h"
#include "CudaUtils.h"

#define PRINT_CUMATRIX(m) logInfo ("%s %p %s %s", #m, m, oap::cuda::to_string(m).c_str(), oap::cuda::GetMatrixInfo(m).toString().c_str());

namespace oap
{
namespace cuda
{

math::Matrix* NewDeviceMatrix(uintt columns, uintt rows, floatt revalue = 0.f, floatt imvalue = 0.f);

math::Matrix* NewDeviceMatrixHostRef(const math::Matrix* hostMatrix);

math::Matrix* NewDeviceMatrixDeviceRef(const math::Matrix* deviceMatrix);

math::Matrix* NewDeviceMatrix(const std::string& matrixStr);

math::Matrix* NewDeviceMatrix(const math::MatrixInfo& minfo);

uintt GetColumns(const math::Matrix* dMatrix);

uintt GetRows(const math::Matrix* dMatrix);

math::Matrix* NewDeviceMatrixCopyOfHostMatrix(const math::Matrix* hostMatrix);

math::Matrix* NewDeviceMatrix(const math::Matrix* hostMatrix, uintt columns, uintt rows);

math::Matrix* NewDeviceMatrix(bool allocRe, bool allocIm, uintt columns, uintt rows);

math::Matrix* NewDeviceReMatrix(uintt columns, uintt rows);

math::Matrix* NewDeviceImMatrix(uintt columns, uintt rows);

math::Matrix* NewHostMatrixCopyOfDeviceMatrix(const math::Matrix* matrix);

void DeleteDeviceMatrix(const math::Matrix* deviceMatrix);

/**
 * @brief CopyDeviceMatrixToHostMatrix - copy device to host - matrices must have the same columns and rows
 * @param dst - host matrix
 * @param src - device matrix
 */
void CopyDeviceMatrixToHostMatrix(math::Matrix* dst, const math::Matrix* src);

/**
 * @brief copies davice matrix to host matrix - copy device to host - matrices must have the same product of columns and rows
 *                                              (dst->columns * dst->rows == src->columns * src->rows)
 * @param dst - host matrix
 * @param src - device matrix
 */
void CopyDeviceToHost(math::Matrix* dst, const math::Matrix* src);

/**
 * @brief copies host matrix to device matrix - copy host to device - matrices must have the same columns and rows
 * @param dst - host matrix
 * @param src - device matrix
 */
void CopyHostMatrixToDeviceMatrix(math::Matrix* dst, const math::Matrix* src);

/**
 * @brief copies host matrix to device matrix - copy device to host - matrices must have the same product of columns and rows
 *                                              (dst->columns * dst->rows == src->columns * src->rows)
 * @param dst - host matrix
 * @param src - device matrix
 */
void CopyHostToDevice(math::Matrix* dst, const math::Matrix* src);

/**
 * @brief copies device matrix to device matrix - copy device to device - matrices must have the same columns and rows
 * @param dst - host matrix
 * @param src - device matrix
 */
void CopyDeviceMatrixToDeviceMatrix(math::Matrix* dst, const math::Matrix* src);

/**
 * @brief copies device matri to device matrix - copy device to host - matrices must have the same product of columns and rows
 *                                               (dst->columns * dst->rows == src->columns * src->rows)
 * @param dst - host matrix
 * @param src - device matrix
 */
void CopyDeviceToDevice(math::Matrix* dst, const math::Matrix* src);

/**
 *
 */
void SetMatrix(math::Matrix* matrix, math::Matrix* matrix1, uintt column, uintt row);

void SetReMatrix(math::Matrix* matrix, math::Matrix* matrix1, uintt column, uintt row);

void SetImMatrix(math::Matrix* matrix, math::Matrix* matrix1, uintt column, uintt row);

MatrixEx** NewDeviceMatrixEx(uintt count);

void CopyHostArrayToDeviceMatrix (math::Matrix* matrix, floatt* rebuffer, floatt* imbuffer, size_t length);
void CopyHostArrayToDeviceReMatrix (math::Matrix* matrix, floatt* buffer, size_t length);
void CopyHostArrayToDeviceImMatrix (math::Matrix* matrix, floatt* buffer, size_t length);

void DeleteDeviceMatrixEx(MatrixEx** matrixEx);

void SetMatrixEx(MatrixEx** deviceMatrixEx, const uintt* buffer, uintt count);

void SetMatrixEx(MatrixEx* deviceMatrixEx, const MatrixEx* hostMatrixEx);

void GetMatrixEx(MatrixEx* hostMatrixEx, const MatrixEx* deviceMatrixEx);

MatrixEx* NewDeviceMatrixEx();
MatrixEx* NewDeviceMatrixExCopy(const MatrixEx& hostMatrixEx);

void DeleteDeviceMatrixEx(MatrixEx* matrixEx);

void PrintMatrix(const std::string& text, const math::Matrix* matrix,
                 floatt zeroLimit = 0);

void PrintMatrix(const math::Matrix* matrix);

void SetReValue(math::Matrix* matrix, floatt value, uintt column, uintt row);
void SetReValue(math::Matrix* matrix, floatt value, uintt index);

void SetImValue(math::Matrix* matrix, floatt value, uintt column, uintt row);
void SetImValue(math::Matrix* matrix, floatt value, uintt index);

void SetValue(math::Matrix* matrix, floatt revalue, floatt imvalue,
              uintt column, uintt row);

void SetValue(math::Matrix* matrix, floatt revalue, floatt imvalue,
              uintt index);

math::MatrixInfo GetMatrixInfo(const math::Matrix* devMatrix);

bool IsCudaMatrix(const math::Matrix* devMatrix);

inline bool IsDeviceMatrix(const math::Matrix* devMatrix)
{
	return IsCudaMatrix (devMatrix);
}

void ToString (std::string& str, const math::Matrix* devMatrix);

inline std::string GetMatrixStr (const math::Matrix* devMatrix)
{
  std::string str;
  ToString (str, devMatrix);
  return str;
}

inline std::string to_string (const math::Matrix* devMatrix)
{
  return GetMatrixStr (devMatrix);
}

void PrintMatrixInfo(const std::string& msg, const math::Matrix* devMatrix);

bool WriteMatrix(const std::string& path, const math::Matrix* devMatrix);

math::Matrix* ReadMatrix(const std::string& path);

void SaveMatrix (const math::Matrix* matrix, utils::ByteBuffer& buffer);
void SaveMatrixInfo (const math::MatrixInfo& minfo, utils::ByteBuffer& buffer);

math::Matrix* LoadMatrix (const utils::ByteBuffer& buffer);
math::MatrixInfo LoadMatrixInfo (const utils::ByteBuffer& buffer);

}
}

#endif /* MATRIXMEM_H */
