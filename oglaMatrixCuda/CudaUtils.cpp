#include <string.h>
#include <vector>
#include <algorithm>
#include <map>
#include "DeviceMatrixModules.h"
#include "KernelExecutor.h"
#include "MatrixUtils.h"

namespace CudaUtils {

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
  return reinterpret_cast<CUdeviceptr>(&matrixEx->bcolumn);
}

CUdeviceptr GetEColumnAddress(const MatrixEx* matrixEx) {
  return reinterpret_cast<CUdeviceptr>(&matrixEx->ecolumn);
}

CUdeviceptr GetBRowAddress(const MatrixEx* matrixEx) {
  return reinterpret_cast<CUdeviceptr>(&matrixEx->brow);
}

CUdeviceptr GetERowAddress(const MatrixEx* matrixEx) {
  return reinterpret_cast<CUdeviceptr>(&matrixEx->erow);
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
  uintt bcolumns = 0;
  uintt ecolumns = 0;
  cuMemcpyDtoH(&bcolumns, GetBColumnAddress(matrixEx), sizeof(uintt));
  cuMemcpyDtoH(&ecolumns, GetEColumnAddress(matrixEx), sizeof(uintt));
  return ecolumns - bcolumns;
}

uintt GetRows(const MatrixEx* matrixEx) {
  uintt brows = 0;
  uintt erows = 0;
  cuMemcpyDtoH(&brows, GetBRowAddress(matrixEx), sizeof(uintt));
  cuMemcpyDtoH(&erows, GetERowAddress(matrixEx), sizeof(uintt));
  return erows - brows;
}

CUdeviceptr AllocMatrix(bool allocRe, bool allocIm, uintt columns, uintt rows,
                        floatt revalue, floatt imvalue) {
  class InternalAllocator {
   public:
    static CUdeviceptr allocMatrix() {
      CUdeviceptr devicePtrMatrix = 0;
      printCuError(cuMemAlloc(&devicePtrMatrix, sizeof(math::Matrix)));
      return devicePtrMatrix;
    }
  };
  CUdeviceptr matrix = InternalAllocator::allocMatrix();
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

CUdeviceptr AllocReMatrix(CUdeviceptr devicePtrMatrix, uintt columns,
                          uintt rows, floatt value) {
  CUdeviceptr devicePtrReValues = 0;
  printCuError(cuMemAlloc(&devicePtrReValues, columns * rows * sizeof(floatt)));
  printCuError(cuMemcpyHtoD(GetReValuesAddress(devicePtrMatrix),
                            &devicePtrReValues, sizeof(CUdeviceptr)));
  unsigned int dvalue = *reinterpret_cast<unsigned int*>(&value);
  cuMemsetD32(devicePtrReValues, dvalue, columns * rows * sizeof(floatt) / 4);
  return devicePtrReValues;
}

CUdeviceptr AllocImMatrix(CUdeviceptr devicePtrMatrix, uintt columns,
                          uintt rows, floatt value) {
  CUdeviceptr devicePtrImValues = 0;
  printCuError(cuMemAlloc(&devicePtrImValues, columns * rows * sizeof(floatt)));
  printCuError(cuMemcpyHtoD(GetImValuesAddress(devicePtrMatrix),
                            &devicePtrImValues, sizeof(CUdeviceptr)));
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
  cuMemAlloc(&devicePtr, size);
  cuMemsetD32(devicePtr, 0, size);
  return reinterpret_cast<void*>(devicePtr);
}

void* AllocDeviceMem(uintt size, const void* src) {
  static unsigned int count = 0;
  void* devPtr = AllocDeviceMem(size);
  CopyHostToDevice(devPtr, src, size);
  fprintf(stderr, "count = %u \n", count++);
  return devPtr;
}

void FreeDeviceMem(void* devicePtr) {
  if (devicePtr) {
    CUdeviceptr deviecPtr = reinterpret_cast<CUdeviceptr>(devicePtr);
    FreeDeviceMem(deviecPtr);
  }
}

void FreeDeviceMem(CUdeviceptr ptr) {
  if (ptr != 0) {
    cuMemFree(ptr);
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
                     bool repeats, bool pipe, bool endl) {
  uintt columns = CudaUtils::GetColumns(dmatrix);
  uintt rows = CudaUtils::GetRows(dmatrix);
  bool isre = CudaUtils::GetReValues(dmatrix) != NULL;
  bool isim = CudaUtils::GetImValues(dmatrix) != NULL;
  math::Matrix* hmatrix = host::NewMatrix(isre, isim, columns, rows);
  device::CopyDeviceMatrixToHostMatrix(hmatrix, dmatrix);
  matrixUtils::PrintMatrix(output, hmatrix, repeats, pipe, endl);
  host::DeleteMatrix(hmatrix);
}

void GetMatrixStr(std::string& output, math::Matrix* matrix, bool repeats,
                  bool pipe, bool endl) {
  printHostMatrix(output, matrix, repeats, pipe, endl);
}

void PrintMatrix(FILE* stream, const math::Matrix* matrix, bool repeats,
                 bool pipe, bool endl) {
  std::string output;
  printHostMatrix(output, matrix, repeats, pipe, endl);
  fprintf(stream, "%s CUDA \n", output.c_str());
}

void PrintMatrix(const math::Matrix* matrix, bool repeats, bool pipe,
                 bool endl) {
  PrintMatrix("", matrix, repeats, pipe, endl);
}

void PrintMatrix(const std::string& text, const math::Matrix* matrix,
                 bool repeats, bool pipe, bool endl) {
  std::string output;
  printHostMatrix(output, matrix, repeats, pipe, endl);
  printf("%s %s CUDA \n", text.c_str(), output.c_str());
}
}
