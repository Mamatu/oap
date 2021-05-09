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

#include <algorithm>
#include <map>
#include <string.h>
#include <vector>

#include "oapCudaMatrixUtils.h"
#include "oapCudaMemoryApi.h"
#include "oapHostMatrixUtils.h"

#include "KernelExecutor.h"
#include "MatrixUtils.h"
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

  logTrace ("CUDeviceptr = %u", devPtr);

  return devPtr;
}

inline void freeDeviceMem (CUdeviceptr ptr)
{
  if (ptr != 0)
  {
    logTrace ("~CUDeviceptr = %u", ptr);
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
  void* ptr = reinterpret_cast<void*>(devicePtr);
  logTrace ("ptr = %p", ptr);
  return ptr;
}

void* AllocDeviceMem (uintt size, const void* src)
{
  void* devPtr = AllocDeviceMem(size);
  logTrace ("ptr = %p", devPtr);
  CopyHostToDevice (devPtr, src, size);
  return devPtr;
}

void FreeDeviceMem (const void* devicePtr)
{
  if (devicePtr)
  {
    logTrace ("~ptr = %p", devicePtr);
    CUdeviceptr cuDPtr = reinterpret_cast<CUdeviceptr>(devicePtr);
    FreeDeviceMem (cuDPtr);
  }
}

void ToHost (void* dst, const void* src, size_t size)
{
  cuMemcpyDtoH (&dst, reinterpret_cast<CUdeviceptr>(src), size);
}

oap::Memory GetMemory (const oap::Memory* cumem)
{
  oap::Memory mem = oap::common::OAP_NONE_MEMORY();
  CudaUtils::CopyDeviceToHost (&mem, cumem, sizeof(mem));
  return mem;
}

oap::Memory GetReMemory (const math::ComplexMatrix* matrix)
{
  return GetMemory (&(matrix->re.mem));
}

oap::Memory GetImMemory (const math::ComplexMatrix* matrix)
{
  return GetMemory (&(matrix->im.mem));
}

oap::MemoryRegion GetMemoryRegion (const oap::MemoryRegion* cureg)
{
  oap::MemoryRegion reg = oap::common::OAP_NONE_REGION();
  CudaUtils::CopyDeviceToHost (&reg, cureg, sizeof(reg));
  return reg;
}

oap::MemoryRegion GetReMemoryRegion (const math::ComplexMatrix* matrix)
{
  return GetMemoryRegion (&(matrix->re.reg));
}

oap::MemoryRegion GetImMemoryRegion (const math::ComplexMatrix* matrix)
{
  return GetMemoryRegion (&(matrix->im.reg));
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
math::MatrixInfo GetMatrixInfo(const math::ComplexMatrix* devMatrix)
{
  uintt columns = CudaUtils::GetColumns(devMatrix);
  uintt rows = CudaUtils::GetRows(devMatrix);

  bool isRe = CudaUtils::GetReMemoryPtr (devMatrix) != NULL;
  bool isIm = CudaUtils::GetImMemoryPtr (devMatrix) != NULL;

  return math::MatrixInfo(isRe, isIm, columns, rows);
}
#endif

void CopyHostToDevice(void* dst, const void* src, uintt size) {
  logTrace ("%s %p -> %p size = %u", __FUNCTION__, src, dst, size);
  CUdeviceptr dstPtr = reinterpret_cast<CUdeviceptr>(dst);
  cuMemcpyHtoD(dstPtr, src, size);
}

void CopyDeviceToHost(void* dst, const void* src, uintt size) {
  logTrace ("%s %p -> %p size = %u", __FUNCTION__, src, dst, size);
  CUdeviceptr srcPtr = reinterpret_cast<CUdeviceptr>(src);
  cuMemcpyDtoH(dst, srcPtr, size);
}

void CopyDeviceToDevice(void* dst, const void* src, uintt size) {
  logTrace ("%s %p -> %p size = %u", __FUNCTION__, src, dst, size);
  CUdeviceptr dstPtr = reinterpret_cast<CUdeviceptr>(dst);
  CUdeviceptr srcPtr = reinterpret_cast<CUdeviceptr>(src);
  cuMemcpyDtoD(dstPtr, srcPtr, size);
}

void MoveDeviceToDevice (void* dst, const void* src, uintt size)
{
  logTrace ("%s %p -> %p size = %u", __FUNCTION__, src, dst, size);
  CUdeviceptr tempPtr = allocDeviceMem (size);
  CUdeviceptr dstPtr = reinterpret_cast<CUdeviceptr>(dst);
  CUdeviceptr srcPtr = reinterpret_cast<CUdeviceptr>(src);
  cuMemcpyDtoD(tempPtr, srcPtr, size);
  cuMemcpyDtoD(dstPtr, tempPtr, size);
  freeDeviceMem (tempPtr);
}

void SetReValue(math::ComplexMatrix* m, uintt index, floatt value)
{
  using namespace oap::utils;
  oap::MemoryRegion regMem = CudaUtils::GetReMemoryRegion (m);
  oap::Memory mem = CudaUtils::GetReMemory (m);

  oap::MemoryLoc loc = {0, 0};

  loc = oap::common::ConvertIdxToMemoryLocRef (index, mem, regMem);

  oap::generic::copy (mem.ptr, mem.dims, loc, &value, {1, 1}, {{0, 0}, {1, 1}}, CudaUtils::CopyHostToDevice, CudaUtils::CopyHostToDevice);
}

floatt GetReValue(const math::ComplexMatrix* m, uintt index)
{
  using namespace oap::utils;

  floatt v;
  oap::MemoryRegion regMem = CudaUtils::GetReMemoryRegion (m);
  oap::Memory mem = CudaUtils::GetReMemory (m);

  oap::MemoryLoc loc = {0, 0};

  loc = oap::common::ConvertIdxToMemoryLocRef (index, mem, regMem);

  oap::generic::copy (&v, {1, 1}, {0, 0}, mem.ptr, mem.dims, {loc, {1, 1}}, CudaUtils::CopyDeviceToHost, CudaUtils::CopyDeviceToHost);

  return v;
}

void SetImValue(math::ComplexMatrix* m, uintt index, floatt value)
{
  using namespace oap::utils;
  oap::MemoryRegion regMem = CudaUtils::GetImMemoryRegion (m);
  oap::Memory mem = CudaUtils::GetImMemory (m);

  oap::MemoryLoc loc = {0, 0};

  loc = oap::common::ConvertIdxToMemoryLocRef (index, mem, regMem);

  oap::generic::copy (mem.ptr, mem.dims, loc, &value, {1, 1}, {{0, 0}, {1, 1}}, CudaUtils::CopyHostToDevice, CudaUtils::CopyHostToDevice);
}

floatt GetImValue(const math::ComplexMatrix* m, uintt index)
{
  using namespace oap::utils;

  floatt v;
  oap::MemoryRegion regMem = CudaUtils::GetImMemoryRegion (m);
  oap::Memory mem = CudaUtils::GetImMemory (m);

  oap::MemoryLoc loc = {0, 0};

  loc = oap::common::ConvertIdxToMemoryLocRef (index, mem, regMem);

  oap::generic::copy (&v, {1, 1}, {0, 0}, mem.ptr, mem.dims, {loc, {1, 1}}, CudaUtils::CopyDeviceToHost, CudaUtils::CopyDeviceToHost);

  return v;
}

#if 0
floatt GetReDiagonal(math::ComplexMatrix* m, uintt index)
{
  uintt columns = GetColumns(m);
  return GetReValue(m, index * columns + index);
}

floatt GetImDiagonal(math::ComplexMatrix* m, uintt index) {
  uintt columns = GetColumns(m);
  return GetImValue(m, index * columns + index);
}

void SetZeroMatrix(math::ComplexMatrix* matrix, bool re, bool im) {
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

void SetZeroRow(math::ComplexMatrix* matrix, uintt index, bool re, bool im) {
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
void GetMatrixStr(std::string& output, math::ComplexMatrix* matrix, floatt zeroLimit, bool repeats, const std::string& sectionSeparator)
{
  matrixUtils::PrintArgs args;

  args.zrr = zeroLimit;
  args.repeats = repeats;
  args.section.separator = sectionSeparator;

  oap::generic::printMatrix (output, matrix, args,
               oap::cuda::GetMatrixInfo, oap::host::NewHostMatrixFromMatrixInfo, oap::host::DeleteComplexMatrix, oap::cuda::CopyDeviceMatrixToHostMatrix);
}

void PrintMatrix(FILE* stream, const math::ComplexMatrix* matrix, floatt zeroLimit, bool repeats, const std::string& sectionSeparator)
{
  matrixUtils::PrintArgs args;

  args.zrr = zeroLimit;
  args.repeats = repeats;
  args.section.separator = sectionSeparator;

  std::string output;

  oap::generic::printMatrix (output, matrix, args,
               oap::cuda::GetMatrixInfo, oap::host::NewHostMatrixFromMatrixInfo, oap::host::DeleteComplexMatrix, oap::cuda::CopyDeviceMatrixToHostMatrix);
  fprintf(stream, "%s CUDA \n", output.c_str());
}

void PrintMatrix(const math::ComplexMatrix* matrix, floatt zeroLimit, bool repeats, const std::string& sectionSeparator)
{
  PrintMatrix("", matrix, zeroLimit, repeats, sectionSeparator);
}

void PrintMatrix(const std::string& text, const math::ComplexMatrix* matrix, floatt zeroLimit, bool repeats, const std::string& sectionSeparator)
{
  matrixUtils::PrintArgs args;

  args.zrr = zeroLimit;
  args.repeats = repeats;
  args.section.separator = sectionSeparator;

  std::string output;

  oap::generic::printMatrix (output, matrix, args,
               oap::cuda::GetMatrixInfo, oap::host::NewHostMatrixFromMatrixInfo, oap::host::DeleteComplexMatrix, oap::cuda::CopyDeviceMatrixToHostMatrix);
  printf("%s %s CUDA \n", text.c_str(), output.c_str());
}
}
