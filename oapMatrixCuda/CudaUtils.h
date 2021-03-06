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

inline void* Malloc (uintt size)
{
  return AllocDeviceMem (size);
}

inline void Free (void* ptr)
{
  FreeDeviceMem (ptr);
}

#if 0
math::MatrixInfo GetMatrixInfo(const math::ComplexMatrix* devMatrix);
#endif

void CopyHostToDevice(void* dst, const void* src, uintt size);

void CopyDeviceToHost(void* dst, const void* src, uintt size);

void CopyDeviceToDevice(void* dst, const void* src, uintt size);
void MoveDeviceToDevice(void* dst, const void* src, uintt size);

void ToHost (void* dst, const void* src, size_t length);

oap::MemoryRegion* GetReMemoryRegionPtr (const math::ComplexMatrix* matrix);
oap::MemoryRegion* GetImMemoryRegionPtr (const math::ComplexMatrix* matrix);

oap::Memory GetMemory (const math::ComplexMatrix* matrix, oap::Memory* cuptr);

oap::Memory GetReMemory (const math::ComplexMatrix* matrix);
oap::Memory GetImMemory (const math::ComplexMatrix* matrix);

oap::MemoryRegion GetMemoryRegion (const math::ComplexMatrix* matrix, oap::MemoryRegion* cuptr);

oap::MemoryRegion GetReMemoryRegion (const math::ComplexMatrix* matrix);
oap::MemoryRegion GetImMemoryRegion (const math::ComplexMatrix* matrix);

//uintt GetColumns(const math::ComplexMatrix* matrix);
//uintt GetRows(const math::ComplexMatrix* matrix);

CUdeviceptr GetBColumnAddress(const MatrixEx* matrixEx);
CUdeviceptr GetColumnsAddress(const MatrixEx* matrixEx);

CUdeviceptr GetBRowAddress(const MatrixEx* matrixEx);
CUdeviceptr GetRowAddress(const MatrixEx* matrixEx);

floatt* GetReValues(CUdeviceptr matrix);
floatt* GetImValues(CUdeviceptr matrix);

uintt GetColumns(const MatrixEx* matrix);
uintt GetRows(const MatrixEx* matrix);

CUdeviceptr SetReMatrixToNull(CUdeviceptr devicePtrMatrix);
CUdeviceptr SetImMatrixToNull(CUdeviceptr devicePtrMatrix);

void SetVariables(CUdeviceptr devicePtrMatrix, uintt columns, uintt rows);

void SetReValue (math::ComplexMatrix* m, uintt index, floatt value);
floatt GetReValue (const math::ComplexMatrix* m, uintt index);

void SetImValue (math::ComplexMatrix* m, uintt index, floatt value);
floatt GetImValue (const math::ComplexMatrix* m, uintt index);

#if 0
floatt GetReDiagonal(math::ComplexMatrix* m, uintt index);
floatt GetImDiagonal(math::ComplexMatrix* m, uintt index);

void SetZeroMatrix(math::ComplexMatrix* matrix, bool re = true, bool im = true);
void SetZeroRow(math::ComplexMatrix* matrix, uintt index, bool re = true, bool im = true);
#endif
void GetMatrixStr(std::string& output, const math::ComplexMatrix* matrix, floatt zeroLimit = 0, bool repeats = false, const std::string& sectionSeparator = "|\n");

void PrintMatrix(FILE* stream, const math::ComplexMatrix* matrix, floatt zeroLimit = 0, bool repeats = false, const std::string& sectionSeparator = "|\n");

void PrintMatrix(const math::ComplexMatrix* matrix, floatt zeroLimit = 0, bool repeats = false, const std::string& sectionSeparator = "|\n");

void PrintMatrix(const std::string& output, const math::ComplexMatrix* matrix, floatt zeroLimit = 0, bool repeats = false, const std::string& sectionSeparator = "|\n");

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
