/*
 * Copyright 2016 Marcin Matula
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

#include <string.h>
#include <vector>
#include <algorithm>
#include <map>
#include "DeviceMatrixModules.h"
#include "KernelExecutor.h"
#include "MatrixUtils.h"

namespace CudaUtils {

void AllocDeviceMem(CUdeviceptr* devPtr, size_t size) {
  printCuError(cuMemAlloc(devPtr, size));
}

void FreeDeviceMem(CUdeviceptr ptr) {
  if (ptr != 0) {
    printCuError(cuMemFree(ptr));
  }
}

CUdeviceptr GetReValuesAddress(const math::Matrix* matrix) {
  return reinterpret_cast<CUdeviceptr>(&matrix->reValues);
}

CUdeviceptr GetImValuesAddress(const math::Matrix* matrix) {
  return reinterpret_cast<CUdeviceptr>(&matrix->imValues);
}

CUdeviceptr GetColumnsAddress(const math::Matrix* matrix) {
  return reinterpret_cast<CUdeviceptr>(&matrix->columns);
}

CUdeviceptr GetRowsAddress(const math::Matrix* matrix) {
  return reinterpret_cast<CUdeviceptr>(&matrix->rows);
}

floatt* GetReValues(const math::Matrix* matrix) {
  floatt* reValues = NULL;
  cuMemcpyDtoH(&reValues, GetReValuesAddress(matrix), sizeof(floatt*));
  return reValues;
}

floatt* GetImValues(const math::Matrix* matrix) {
  floatt* imValues = NULL;
  cuMemcpyDtoH(&imValues, GetImValuesAddress(matrix), sizeof(floatt*));
  return imValues;
}

uintt GetColumns(const math::Matrix* matrix) {
  uintt columns = 0;
  cuMemcpyDtoH(&columns, GetColumnsAddress(matrix), sizeof(uintt));
  return columns;
}

uintt GetRows(const math::Matrix* matrix) {
  uintt rows = 0;
  cuMemcpyDtoH(&rows, GetRowsAddress(matrix), sizeof(uintt));
  return rows;
}

CUdeviceptr GetReValuesAddress(CUdeviceptr matrixptr) {
  math::Matrix* matrix = reinterpret_cast<math::Matrix*>(matrixptr);
  return reinterpret_cast<CUdeviceptr>(&matrix->reValues);
}

CUdeviceptr GetImValuesAddress(CUdeviceptr matrixptr) {
  math::Matrix* matrix = reinterpret_cast<math::Matrix*>(matrixptr);
  return reinterpret_cast<CUdeviceptr>(&matrix->imValues);
}

CUdeviceptr GetColumnsAddress(CUdeviceptr matrixptr) {
  math::Matrix* matrix = reinterpret_cast<math::Matrix*>(matrixptr);
  return reinterpret_cast<CUdeviceptr>(&matrix->columns);
}

CUdeviceptr GetRealColumnsAddress(CUdeviceptr matrixptr) {
  math::Matrix* matrix = reinterpret_cast<math::Matrix*>(matrixptr);
  return reinterpret_cast<CUdeviceptr>(&matrix->realColumns);
}

CUdeviceptr GetRowsAddress(CUdeviceptr matrixptr) {
  math::Matrix* matrix = reinterpret_cast<math::Matrix*>(matrixptr);
  return reinterpret_cast<CUdeviceptr>(&matrix->rows);
}

CUdeviceptr GetRealRowsAddress(CUdeviceptr matrixptr) {
  math::Matrix* matrix = reinterpret_cast<math::Matrix*>(matrixptr);
  return reinterpret_cast<CUdeviceptr>(&matrix->realRows);
}

CUdeviceptr GetBColumnAddress(const MatrixEx* matrixEx) {
  return reinterpret_cast<CUdeviceptr>(&matrixEx->beginColumn);
}

CUdeviceptr GetColumnsAddress(const MatrixEx* matrixEx) {
  return reinterpret_cast<CUdeviceptr>(&matrixEx->columnsLength);
}

CUdeviceptr GetBRowAddress(const MatrixEx* matrixEx) {
  return reinterpret_cast<CUdeviceptr>(&matrixEx->beginRow);
}

CUdeviceptr GetRowsAddress(const MatrixEx* matrixEx) {
  return reinterpret_cast<CUdeviceptr>(&matrixEx->rowsLength);
}

floatt* GetReValues(CUdeviceptr matrix) {
  floatt* reValues = NULL;
  cuMemcpyDtoH(&reValues, GetReValuesAddress(matrix), sizeof(floatt*));
  return reValues;
}

floatt* GetImValues(CUdeviceptr matrix) {
  floatt* imValues = NULL;
  cuMemcpyDtoH(&imValues, GetImValuesAddress(matrix), sizeof(floatt*));
  return imValues;
}

uintt GetColumns(CUdeviceptr matrix) {
  uintt columns = 0;
  cuMemcpyDtoH(&columns, GetColumnsAddress(matrix), sizeof(uintt));
  return columns;
}

uintt GetRows(CUdeviceptr matrix) {
  uintt rows = 0;
  cuMemcpyDtoH(&rows, GetRowsAddress(matrix), sizeof(uintt));
  return rows;
}

uintt GetColumns(const MatrixEx* matrixEx) {
  uintt columns = 0;
  cuMemcpyDtoH(&columns, GetColumnsAddress(matrixEx), sizeof(uintt));
  return columns;
}

uintt GetRows(const MatrixEx* matrixEx) {
  uintt rows = 0;
  cuMemcpyDtoH(&rows, GetRowsAddress(matrixEx), sizeof(uintt));
  return rows;
}

CUdeviceptr AllocMatrix(bool allocRe, bool allocIm, uintt columns, uintt rows,
                        floatt revalue, floatt imvalue) {
  CUdeviceptr matrix = 0;
  AllocDeviceMem(&matrix, sizeof(math::Matrix));

  CUdeviceptr matrixRe = 0;
  CUdeviceptr matrixIm = 0;

  if (allocRe) {
    matrixRe = AllocReMatrix(matrix, columns, rows, revalue);
  } else {
    matrixRe = SetReMatrixToNull(matrix);
  }

  if (allocIm) {
    matrixIm = AllocImMatrix(matrix, columns, rows, imvalue);
  } else {
    matrixIm = SetImMatrixToNull(matrix);
  }

  SetVariables(matrix, columns, rows);
  return matrix;
}

void CopyHtoD(CUdeviceptr devPtr, void* hostPtr, size_t size) {
  printCuError(cuMemcpyHtoD(devPtr, hostPtr, size));
}

CUdeviceptr AllocReMatrix(CUdeviceptr devicePtrMatrix, uintt columns,
                          uintt rows, floatt value) {
  CUdeviceptr devicePtrReValues = 0;
  AllocDeviceMem(&devicePtrReValues, columns * rows * sizeof(floatt));
  CopyHtoD(GetReValuesAddress(devicePtrMatrix), &devicePtrReValues,
           sizeof(CUdeviceptr));
  unsigned int dvalue = *reinterpret_cast<unsigned int*>(&value);
  cuMemsetD32(devicePtrReValues, dvalue, columns * rows * sizeof(floatt) / 4);
  return devicePtrReValues;
}

CUdeviceptr AllocImMatrix(CUdeviceptr devicePtrMatrix, uintt columns,
                          uintt rows, floatt value) {
  CUdeviceptr devicePtrImValues = 0;
  AllocDeviceMem(&devicePtrImValues, columns * rows * sizeof(floatt));
  CopyHtoD(GetImValuesAddress(devicePtrMatrix), &devicePtrImValues,
           sizeof(CUdeviceptr));
  unsigned int dvalue = *reinterpret_cast<unsigned int*>(&value);
  cuMemsetD32(devicePtrImValues, dvalue, columns * rows * sizeof(floatt) / 4);
  return devicePtrImValues;
}

CUdeviceptr SetReMatrixToNull(CUdeviceptr devicePtrMatrix) {
  CUdeviceptr buffer = 0;
  printCuError(cuMemcpyHtoD(GetReValuesAddress(devicePtrMatrix), &buffer,
                            sizeof(CUdeviceptr)));
  return 0;
}

CUdeviceptr SetImMatrixToNull(CUdeviceptr devicePtrMatrix) {
  CUdeviceptr buffer = 0;
  printCuError(cuMemcpyHtoD(GetImValuesAddress(devicePtrMatrix), &buffer,
                            sizeof(CUdeviceptr)));
  return 0;
}

void SetVariables(CUdeviceptr devicePtrMatrix, uintt columns, uintt rows) {
  printCuError(cuMemcpyHtoD(GetColumnsAddress(devicePtrMatrix), &columns,
                            sizeof(uintt)));
  printCuError(cuMemcpyHtoD(GetRealColumnsAddress(devicePtrMatrix), &columns,
                            sizeof(uintt)));
  printCuError(
      cuMemcpyHtoD(GetRowsAddress(devicePtrMatrix), &rows, sizeof(uintt)));
  printCuError(
      cuMemcpyHtoD(GetRealRowsAddress(devicePtrMatrix), &rows, sizeof(uintt)));
}

void* AllocDeviceMem(uintt size) {
  CUdeviceptr devicePtr;
  AllocDeviceMem(&devicePtr, size);
  debug("%d %d", sizeof(void*), sizeof(CUdeviceptr));
  return reinterpret_cast<void*>(devicePtr);
}

void* AllocDeviceMem(uintt size, const void* src) {
  void* devPtr = AllocDeviceMem(size);
  CopyHostToDevice(devPtr, src, size);
  return devPtr;
}

void FreeDeviceMem(void* devicePtr) {
  if (devicePtr) {
    CUdeviceptr deviecPtr = reinterpret_cast<CUdeviceptr>(devicePtr);
    FreeDeviceMem(deviecPtr);
  }
}

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

void SetReValue(math::Matrix* m, uintt index, floatt value) {
  floatt* array = GetReValues(m);
  cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(&array[index]), &value,
               sizeof(floatt));
}

floatt GetReValue(math::Matrix* m, uintt index) {
  floatt* array = GetReValues(m);
  floatt value = 0;
  cuMemcpyDtoH(&value, reinterpret_cast<CUdeviceptr>(&array[index]),
               sizeof(floatt));
  return value;
}

void SetImValue(math::Matrix* m, uintt index, floatt value) {
  floatt* array = GetImValues(m);
  cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(&array[index]), &value,
               sizeof(floatt));
}

floatt GetImValue(math::Matrix* m, uintt index) {
  floatt* array = GetImValues(m);
  floatt value = 0;
  cuMemcpyDtoH(&value, reinterpret_cast<CUdeviceptr>(&array[index]),
               sizeof(floatt));
  return value;
}

floatt GetReDiagonal(math::Matrix* m, uintt index) {
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

void printHostMatrix(std::string& output, const math::Matrix* dmatrix,
                     floatt zeroLimit, bool repeats, bool pipe, bool endl) {
  uintt columns = CudaUtils::GetColumns(dmatrix);
  uintt rows = CudaUtils::GetRows(dmatrix);
  bool isre = CudaUtils::GetReValues(dmatrix) != NULL;
  bool isim = CudaUtils::GetImValues(dmatrix) != NULL;
  math::Matrix* hmatrix = host::NewMatrix(isre, isim, columns, rows);
  device::CopyDeviceMatrixToHostMatrix(hmatrix, dmatrix);
  matrixUtils::PrintMatrix(output, hmatrix, zeroLimit, repeats, pipe, endl);
  host::DeleteMatrix(hmatrix);
}

void GetMatrixStr(std::string& output, math::Matrix* matrix, floatt zeroLimit,
                  bool repeats, bool pipe, bool endl) {
  printHostMatrix(output, matrix, zeroLimit, repeats, pipe, endl);
}

void PrintMatrix(FILE* stream, const math::Matrix* matrix, floatt zeroLimit,
                 bool repeats, bool pipe, bool endl) {
  std::string output;
  printHostMatrix(output, matrix, zeroLimit, repeats, pipe, endl);
  fprintf(stream, "%s CUDA \n", output.c_str());
}

void PrintMatrix(const math::Matrix* matrix, floatt zeroLimit, bool repeats,
                 bool pipe, bool endl) {
  PrintMatrix("", matrix, zeroLimit, repeats, pipe, endl);
}

void PrintMatrix(const std::string& text, const math::Matrix* matrix,
                 floatt zeroLimit, bool repeats, bool pipe, bool endl) {
  std::string output;
  printHostMatrix(output, matrix, zeroLimit, repeats, pipe, endl);
  printf("%s %s CUDA \n", text.c_str(), output.c_str());
}
}
