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

#include <algorithm>
#include <map>
#include <string.h>
#include <vector>

#include "oapCudaMatrixUtils.h"
#include "oapCudaMemoryApi.h"
#include "oapHostMatrixUtils.h"

#include "KernelExecutor.h"
#include "MatrixUtils.h"
#include "oapMemoryManager.h"
#include "oapMemory_GenericApi.h"
#include "oapMemory_CommonApi.h"
#include "oapMemoryPrimitivesApi.h"

namespace CudaUtils {

namespace
{

inline CUdeviceptr allocDeviceMem (size_t size)
{
  CUdeviceptr devPtr;
  printCuError (cuMemAlloc(&devPtr, size));
  return devPtr;
}

inline void freeDeviceMem (CUdeviceptr ptr)
{
  if (ptr != 0)
  {
    printCuError (cuMemFree(ptr));
  }
}
}

void AllocDeviceMem (CUdeviceptr* devPtr, size_t size)
{
  CUdeviceptr ptr = allocDeviceMem (size);
  *devPtr = ptr;
}

void FreeDeviceMem (CUdeviceptr ptr)
{
  freeDeviceMem (ptr);
}

void* AllocDeviceMem (uintt size)
{
  CUdeviceptr devicePtr;
  AllocDeviceMem (&devicePtr, size);
  return reinterpret_cast<void*>(devicePtr);
}

void* AllocDeviceMem (uintt size, const void* src)
{
  void* devPtr = AllocDeviceMem(size);
  CopyHostToDevice (devPtr, src, size);
  return devPtr;
}

void FreeDeviceMem (const void* devicePtr)
{
  if (devicePtr)
  {
    CUdeviceptr cuDPtr = reinterpret_cast<CUdeviceptr>(devicePtr);
    FreeDeviceMem (cuDPtr);
  }
}

void ToHost (void* dst, const void* src, size_t size)
{
  cuMemcpyDtoH (&dst, reinterpret_cast<CUdeviceptr>(src), size);
}

oap::Memory* GetReMemoryPtr (const math::Matrix* matrix)
{
  oap::Memory* re = nullptr;
  CudaUtils::CopyDeviceToHost (&re, &(matrix->re), sizeof(re));
  return re;
}

oap::Memory* GetImMemoryPtr (const math::Matrix* matrix)
{
  oap::Memory* im = nullptr;
  CudaUtils::CopyDeviceToHost (&im, &(matrix->im), sizeof(im));
  return im;
}

oap::MemoryRegion* GetReMemoryRegionPtr (const math::Matrix* matrix)
{
  oap::MemoryRegion* reReg = nullptr;
  CudaUtils::CopyDeviceToHost (&reReg, &(matrix->reReg), sizeof(reReg));
  return reReg;
}

oap::MemoryRegion* GetImMemoryRegionPtr (const math::Matrix* matrix)
{
  oap::MemoryRegion* imReg = nullptr;
  CudaUtils::CopyDeviceToHost (&imReg, &(matrix->imReg), sizeof(imReg));
  return imReg;
}

oap::Memory GetMemory (const math::Matrix* matrix, oap::Memory* cuptr)
{
  oap::Memory mem = oap::common::OAP_NONE_MEMORY();
  if (cuptr)
  {
    CudaUtils::CopyDeviceToHost (&mem, cuptr, sizeof(mem));
  }
  return mem;
}

oap::Memory GetReMemory (const math::Matrix* matrix)
{
  oap::Memory* cuptr = GetReMemoryPtr (matrix);
  return GetMemory (matrix, cuptr);
}

oap::Memory GetImMemory (const math::Matrix* matrix)
{
  oap::Memory* cuptr = GetImMemoryPtr (matrix);
  return GetMemory (matrix, cuptr);
}

oap::MemoryRegion GetMemoryRegion (const math::Matrix* matrix, oap::MemoryRegion* cuptr)
{
  oap::MemoryRegion reg = oap::common::OAP_NONE_REGION();
  if (cuptr)
  {
    CudaUtils::CopyDeviceToHost (&reg, cuptr, sizeof(reg));
  }
  return reg;
}

oap::MemoryRegion GetReMemoryRegion (const math::Matrix* matrix)
{
  oap::MemoryRegion* cuptr = GetReMemoryRegionPtr (matrix);
  return GetMemoryRegion (matrix, cuptr);
}

oap::MemoryRegion GetImMemoryRegion (const math::Matrix* matrix)
{
  oap::MemoryRegion* cuptr = GetImMemoryRegionPtr (matrix);
  return GetMemoryRegion (matrix, cuptr);
}

CUdeviceptr GetBColumnAddress(const MatrixEx* matrixEx)
{
  return reinterpret_cast<CUdeviceptr>(&matrixEx->column);
}

CUdeviceptr GetColumnsAddress(const MatrixEx* matrixEx)
{
  return reinterpret_cast<CUdeviceptr>(&matrixEx->columns);
}

CUdeviceptr GetBRowAddress(const MatrixEx* matrixEx)
{
  return reinterpret_cast<CUdeviceptr>(&matrixEx->row);
}

CUdeviceptr GetRowsAddress(const MatrixEx* matrixEx)
{
  return reinterpret_cast<CUdeviceptr>(&matrixEx->rows);
}

uintt GetColumns(const MatrixEx* matrixEx)
{
  uintt columns = 0;
  cuMemcpyDtoH(&columns, GetColumnsAddress(matrixEx), sizeof(uintt));
  return columns;
}

uintt GetRows(const MatrixEx* matrixEx)
{
  uintt rows = 0;
  cuMemcpyDtoH(&rows, GetRowsAddress(matrixEx), sizeof(uintt));
  return rows;
}

void CopyHtoD(CUdeviceptr devPtr, void* hostPtr, size_t size) {
  printCuError(cuMemcpyHtoD(devPtr, hostPtr, size));
}

#if 0
math::MatrixInfo GetMatrixInfo(const math::Matrix* devMatrix)
{
  uintt columns = CudaUtils::GetColumns(devMatrix);
  uintt rows = CudaUtils::GetRows(devMatrix);

  bool isRe = CudaUtils::GetReMemoryPtr (devMatrix) != NULL;
  bool isIm = CudaUtils::GetImMemoryPtr (devMatrix) != NULL;

  return math::MatrixInfo(isRe, isIm, columns, rows);
}
#endif

void CopyHostToDevice(void* dst, const void* src, uintt size) {
  CUdeviceptr dstPtr = reinterpret_cast<CUdeviceptr>(dst);
  cuMemcpyHtoD(dstPtr, src, size);
}

void CopyDeviceToHost(void* dst, const void* src, uintt size) {
  CUdeviceptr srcPtr = reinterpret_cast<CUdeviceptr>(src);
  cuMemcpyDtoH(dst, srcPtr, size);
}

void CopyDeviceToDevice(void* dst, const void* src, uintt size) {
  CUdeviceptr dstPtr = reinterpret_cast<CUdeviceptr>(dst);
  CUdeviceptr srcPtr = reinterpret_cast<CUdeviceptr>(src);
  cuMemcpyDtoD(dstPtr, srcPtr, size);
}

void SetReValue(math::Matrix* m, uintt index, floatt value)
{
  using namespace oap::utils;
  oap::MemoryRegion regMem = CudaUtils::GetReMemoryRegion (m);
  oap::Memory mem = CudaUtils::GetReMemory (m);

  oap::MemoryLoc loc = {0, 0};

  loc = oap::common::ConvertIdxToMemoryLocRef (index, mem, regMem);

  oap::generic::copy (mem.ptr, mem.dims, loc, &value, {1, 1}, {{0, 0}, {1, 1}}, CudaUtils::CopyHostToDevice);
}

floatt GetReValue(const math::Matrix* m, uintt index)
{
  using namespace oap::utils;

  floatt v;
  oap::MemoryRegion regMem = CudaUtils::GetReMemoryRegion (m);
  oap::Memory mem = CudaUtils::GetReMemory (m);

  oap::MemoryLoc loc = {0, 0};

  loc = oap::common::ConvertIdxToMemoryLocRef (index, mem, regMem);

  oap::generic::copy (&v, {1, 1}, {0, 0}, mem.ptr, {1, 1}, {{0, 0}, {1, 1}}, CudaUtils::CopyHostToDevice);

  return v;
}

void SetImValue(math::Matrix* m, uintt index, floatt value)
{
  using namespace oap::utils;
  oap::MemoryRegion regMem = CudaUtils::GetImMemoryRegion (m);
  oap::Memory mem = CudaUtils::GetImMemory (m);

  oap::MemoryLoc loc = {0, 0};

  loc = oap::common::ConvertIdxToMemoryLocRef (index, mem, regMem);

  oap::generic::copy (mem.ptr, mem.dims, loc, &value, {1, 1}, {{0, 0}, {1, 1}}, CudaUtils::CopyHostToDevice);
}

floatt GetImValue(const math::Matrix* m, uintt index)
{
  using namespace oap::utils;

  floatt v;
  oap::MemoryRegion regMem = CudaUtils::GetImMemoryRegion (m);
  oap::Memory mem = CudaUtils::GetImMemory (m);

  oap::MemoryLoc loc = {0, 0};

  loc = oap::common::ConvertIdxToMemoryLocRef (index, mem, regMem);

  oap::generic::copy (&v, {1, 1}, {0, 0}, mem.ptr, {1, 1}, {{0, 0}, {1, 1}}, CudaUtils::CopyHostToDevice);

  return v;
}

#if 0
floatt GetReDiagonal(math::Matrix* m, uintt index)
{
  uintt columns = GetColumns(m);
  return GetReValue(m, index * columns + index);
}

floatt GetImDiagonal(math::Matrix* m, uintt index) {
  uintt columns = GetColumns(m);
  return GetImValue(m, index * columns + index);
}

void SetZeroMatrix(math::Matrix* matrix, bool re, bool im) {
  floatt* rearray = GetReValues(matrix);
  floatt* imarray = GetImValues(matrix);
  uintt columns = GetColumns(matrix);
  uintt rows = GetRows(matrix);
  size_t elementsCount = columns * rows * sizeof(floatt) / 4;
  if (NULL != rearray && re) {
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(rearray), 0, elementsCount);
  }
  if (NULL != imarray && im) {
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(imarray), 0, elementsCount);
  }
}

void SetZeroRow(math::Matrix* matrix, uintt index, bool re, bool im) {
  floatt* rearray = GetReValues(matrix);
  floatt* imarray = GetImValues(matrix);
  uintt columns = GetColumns(matrix);
  size_t elementsCount = columns * sizeof(floatt) / 4;
  if (NULL != rearray && re) {
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(&rearray[columns * index]), 0,
                elementsCount);
  }
  if (NULL != imarray && im) {
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(&imarray[columns * index]), 0,
                elementsCount);
  }
}
#endif
void GetMatrixStr(std::string& output, math::Matrix* matrix, floatt zeroLimit, bool repeats, const std::string& sectionSeparator)
{
  matrixUtils::PrintArgs args;

  args.zrr = zeroLimit;
  args.repeats = repeats;
  args.section.separator = sectionSeparator;

  oap::generic::printMatrix (output, matrix, args,
               oap::cuda::GetMatrixInfo, oap::host::NewHostMatrixFromMatrixInfo, oap::host::DeleteMatrix, oap::cuda::CopyDeviceMatrixToHostMatrix);
}

void PrintMatrix(FILE* stream, const math::Matrix* matrix, floatt zeroLimit, bool repeats, const std::string& sectionSeparator)
{
  matrixUtils::PrintArgs args;

  args.zrr = zeroLimit;
  args.repeats = repeats;
  args.section.separator = sectionSeparator;

  std::string output;

  oap::generic::printMatrix (output, matrix, args,
               oap::cuda::GetMatrixInfo, oap::host::NewHostMatrixFromMatrixInfo, oap::host::DeleteMatrix, oap::cuda::CopyDeviceMatrixToHostMatrix);
  fprintf(stream, "%s CUDA \n", output.c_str());
}

void PrintMatrix(const math::Matrix* matrix, floatt zeroLimit, bool repeats, const std::string& sectionSeparator)
{
  PrintMatrix("", matrix, zeroLimit, repeats, sectionSeparator);
}

void PrintMatrix(const std::string& text, const math::Matrix* matrix, floatt zeroLimit, bool repeats, const std::string& sectionSeparator)
{
  matrixUtils::PrintArgs args;

  args.zrr = zeroLimit;
  args.repeats = repeats;
  args.section.separator = sectionSeparator;

  std::string output;

  oap::generic::printMatrix (output, matrix, args,
               oap::cuda::GetMatrixInfo, oap::host::NewHostMatrixFromMatrixInfo, oap::host::DeleteMatrix, oap::cuda::CopyDeviceMatrixToHostMatrix);
  printf("%s %s CUDA \n", text.c_str(), output.c_str());
}
}
