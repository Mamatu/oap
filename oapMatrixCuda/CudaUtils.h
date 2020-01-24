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

#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cuda.h>
#include <stdio.h>
#include <string>
#include "Matrix.h"
#include "MatrixInfo.h"
#include "MatrixEx.h"
#include "MatrixPrinter.h"

namespace CudaUtils
{

template <typename T>
T* AllocDeviceObj(const T& v = 0);

template <typename T>
void FreeDeviceObj(T* valuePtr);

void* AllocDeviceMem(uintt size);

void* AllocDeviceMem(uintt size, const void* src);

void FreeDeviceMem(const void* devicePtr);

void FreeDeviceMem(CUdeviceptr ptr);

math::MatrixInfo GetMatrixInfo(const math::Matrix* devMatrix);

void CopyHostToDevice(void* dst, const void* src, uintt size);

void CopyDeviceToHost(void* dst, const void* src, uintt size);

void CopyDeviceToDevice(void* dst, const void* src, uintt size);

CUdeviceptr GetReValuesAddress(const math::Matrix* matrix);
CUdeviceptr GetImValuesAddress(const math::Matrix* matrix);
CUdeviceptr GetColumnsAddress(const math::Matrix* matrix);
CUdeviceptr GetRowsAddress(const math::Matrix* matrix);

floatt* GetValue (floatt* const* ptr);

floatt* GetReValues(const math::Matrix* matrix);
floatt* GetImValues(const math::Matrix* matrix);
uintt GetColumns(const math::Matrix* matrix);
uintt GetRows(const math::Matrix* matrix);
CUdeviceptr GetReValuesAddress(CUdeviceptr matrix);
CUdeviceptr GetImValuesAddress(CUdeviceptr matrix);
CUdeviceptr GetColumnsAddress(CUdeviceptr matrix);
CUdeviceptr GetRealColumnsAddress(CUdeviceptr matrix);
CUdeviceptr GetRowsAddress(CUdeviceptr matrix);
CUdeviceptr GetRealRowsAddress(CUdeviceptr matrix);

CUdeviceptr GetBColumnAddress(const MatrixEx* matrixEx);
CUdeviceptr GetColumnsAddress(const MatrixEx* matrixEx);

CUdeviceptr GetBRowAddress(const MatrixEx* matrixEx);
CUdeviceptr GetRowAddress(const MatrixEx* matrixEx);

floatt* GetReValues(CUdeviceptr matrix);

floatt* GetImValues(CUdeviceptr matrix);

uintt GetColumns(CUdeviceptr matrix);

uintt GetRows(CUdeviceptr matrix);

uintt GetColumns(const MatrixEx* matrix);
uintt GetRows(const MatrixEx* matrix);

struct CuDevicePtrs
{
  CUdeviceptr matrixPtr;
  CUdeviceptr reValuesPtr;
  CUdeviceptr imValuesPtr;
};

CUdeviceptr AllocMatrix_AllocMemory (bool allocRe, bool allocIm, uintt columns, uintt rows, floatt revalue = 0, floatt imvalue = 0, CuDevicePtrs* cuDevicePtrs = nullptr);
CUdeviceptr AllocReMatrix (CUdeviceptr devicePtrMatrix, uintt columns, uintt rows, floatt value);
CUdeviceptr AllocImMatrix (CUdeviceptr devicePtrMatrix, uintt columns, uintt rows, floatt value);

CUdeviceptr AllocMatrix_ReuseMemory (uintt columns, uintt rows, floatt* revalues, floatt* imvalues, CuDevicePtrs* cuDevicePtrs = nullptr);

CUdeviceptr SetReMatrixToNull(CUdeviceptr devicePtrMatrix);
CUdeviceptr SetImMatrixToNull(CUdeviceptr devicePtrMatrix);

void SetVariables(CUdeviceptr devicePtrMatrix, uintt columns, uintt rows);

void SetReValue (math::Matrix* m, uintt index, floatt value);
floatt GetReValue (const math::Matrix* m, uintt index);

void SetImValue (math::Matrix* m, uintt index, floatt value);
floatt GetImValue (const math::Matrix* m, uintt index);

floatt GetReDiagonal(math::Matrix* m, uintt index);
floatt GetImDiagonal(math::Matrix* m, uintt index);

void SetZeroMatrix(math::Matrix* matrix, bool re = true, bool im = true);
void SetZeroRow(math::Matrix* matrix, uintt index, bool re = true, bool im = true);

void GetMatrixStr(std::string& output, const math::Matrix* matrix, floatt zeroLimit = 0, bool repeats = false, const std::string& sectionSeparator = "|\n");

void PrintMatrix(FILE* stream, const math::Matrix* matrix, floatt zeroLimit = 0, bool repeats = false, const std::string& sectionSeparator = "|\n");

void PrintMatrix(const math::Matrix* matrix, floatt zeroLimit = 0, bool repeats = false, const std::string& sectionSeparator = "|\n");

void PrintMatrix(const std::string& output, const math::Matrix* matrix, floatt zeroLimit = 0, bool repeats = false, const std::string& sectionSeparator = "|\n");

}

template <typename T>
T* CudaUtils::AllocDeviceObj(const T& v) {
  T* valuePtr = NULL;
  void* ptr = CudaUtils::AllocDeviceMem(sizeof(T));
  valuePtr = reinterpret_cast<T*>(ptr);
  CudaUtils::CopyHostToDevice(valuePtr, &v, sizeof(T));
  return valuePtr;
}

template <typename T>
void CudaUtils::FreeDeviceObj(T* valuePtr) {
  CudaUtils::FreeDeviceMem(valuePtr);
}

#endif /* DEVICEUTILS_H */
