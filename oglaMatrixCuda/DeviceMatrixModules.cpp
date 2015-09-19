#include "DeviceMatrixModules.h"
#include "KernelExecutor.h"
#include <string.h>
#include <vector>
#include <algorithm>
#include <netdb.h>
#include <map>

namespace cuda {

math::Matrix* NewHostMatrixCopyOfDeviceMatrix(const math::Matrix* matrix) {
  CUdeviceptr matrixRePtr = CudaUtils::GetReValuesAddress(matrix);
  CUdeviceptr matrixImPtr = CudaUtils::GetImValuesAddress(matrix);
  uintt columns = CudaUtils::GetColumns(matrix);
  uintt rows = CudaUtils::GetRows(matrix);
  math::Matrix* matrix1 = NULL;
  if (matrixRePtr != 0 && matrixImPtr != 0) {
    matrix1 = HostMatrixModules::GetInstance()->getMatrixAllocator()->newMatrix(
        columns, rows);
  } else if (matrixRePtr != 0) {
    matrix1 =
        HostMatrixModules::GetInstance()->getMatrixAllocator()->newReMatrix(
            columns, rows);
  } else if (matrixImPtr != 0) {
    matrix1 =
        HostMatrixModules::GetInstance()->getMatrixAllocator()->newImMatrix(
            columns, rows);
  }
  cuda::CopyDeviceMatrixToHostMatrix(matrix1, matrix);
  return matrix1;
}

math::Matrix* NewDeviceMatrix(const math::Matrix* hostMatrix) {
  return NewDeviceMatrix(hostMatrix, hostMatrix->columns, hostMatrix->rows);
}

math::Matrix* NewDeviceMatrixCopy(const math::Matrix* hostMatrix) {
  math::Matrix* dmatrix = cuda::NewDeviceMatrix(hostMatrix);
  cuda::CopyHostMatrixToDeviceMatrix(dmatrix, hostMatrix);
  return dmatrix;
}

typedef std::pair<uintt, uintt> MatrixDim;
typedef std::pair<bool, bool> MatrixStr;
typedef std::pair<MatrixDim, MatrixStr> MatrixInfo;
typedef std::map<const math::Matrix*, MatrixInfo> MatrixInfos;

MatrixInfos globalMatrixInfos;

math::Matrix* allocMatrix(bool allocRe, bool allocIm, uintt columns, uintt rows,
                          floatt revalue = 0.f, floatt imvalue = 0.f) {
  CUdeviceptr ptr = CudaUtils::AllocMatrix(allocRe, allocIm, columns, rows);
  math::Matrix* mptr = reinterpret_cast<math::Matrix*>(ptr);
  globalMatrixInfos[mptr] =
      MatrixInfo(MatrixDim(columns, rows), MatrixStr(allocRe, allocIm));
  return mptr;
}

math::Matrix* NewDeviceMatrix(const math::Matrix* hostMatrix, uintt columns,
                              uintt rows) {
  bool allocRe = hostMatrix->reValues != NULL;
  bool allocIm = hostMatrix->imValues != NULL;
  return allocMatrix(allocRe, allocIm, columns, rows);
}

math::Matrix* NewDeviceMatrix(bool isRe, bool isIm, uintt columns, uintt rows) {
  debugAssert(isRe != false || isIm != false);
  return allocMatrix(isRe, isIm, columns, rows);
}

math::Matrix* NewDeviceMatrix(uintt columns, uintt rows, floatt revalue,
                              floatt imvalue) {
  debugFuncBegin();
  math::Matrix* dmatrix =
      allocMatrix(true, true, columns, rows, revalue, imvalue);
  debugFuncEnd();
  return dmatrix;
}

void DeleteDeviceMatrix(math::Matrix* dMatrix) {
  if (dMatrix != NULL) {
    MatrixInfos::iterator it = globalMatrixInfos.find(dMatrix);
    if (globalMatrixInfos.end() != it) {
      globalMatrixInfos.erase(it);
    }
    CUdeviceptr rePtr = CudaUtils::GetReValuesAddress(dMatrix);
    CUdeviceptr imPtr = CudaUtils::GetImValuesAddress(dMatrix);
    CUdeviceptr matrixPtr = reinterpret_cast<CUdeviceptr>(dMatrix);
    CudaUtils::FreeDeviceMem(matrixPtr);
    CudaUtils::FreeDeviceMem(imPtr);
    CudaUtils::FreeDeviceMem(rePtr);
  }
}

uintt GetColumns(const math::Matrix* dMatrix) {
  return globalMatrixInfos[dMatrix].first.first;
}

uintt GetRows(const math::Matrix* dMatrix) {
  return globalMatrixInfos[dMatrix].first.second;
}

void CopyDeviceMatrixToHostMatrix(math::Matrix* dst, const math::Matrix* src) {
  uintt length1 = dst->columns * dst->rows;
  uintt length2 = CudaUtils::GetColumns(src) * CudaUtils::GetRows(src);
  length1 = length1 < length2 ? length1 : length2;
  debugAssert(length1 == length2);
  CUdeviceptr srcRePtr =
      reinterpret_cast<CUdeviceptr>(CudaUtils::GetReValues(src));
  CUdeviceptr srcImPtr =
      reinterpret_cast<CUdeviceptr>(CudaUtils::GetImValues(src));
  if (srcRePtr != 0 && dst->reValues != NULL) {
    cuMemcpyDtoH(dst->reValues, srcRePtr, length1 * sizeof(floatt));
  }
  if (srcImPtr != 0 && dst->imValues != NULL) {
    cuMemcpyDtoH(dst->imValues, srcImPtr, length1 * sizeof(floatt));
  }
}

void CopyHostMatrixToDeviceMatrix(math::Matrix* dst, const math::Matrix* src) {
  uintt length1 = CudaUtils::GetColumns(dst) * CudaUtils::GetRows(dst);
  uintt length2 = src->columns * src->rows;
  length1 = length1 < length2 ? length1 : length2;
  debugAssert(length1 == length2);
  CUdeviceptr dstRePtr =
      reinterpret_cast<CUdeviceptr>(CudaUtils::GetReValues(dst));
  CUdeviceptr dstImPtr =
      reinterpret_cast<CUdeviceptr>(CudaUtils::GetImValues(dst));
  if (dstRePtr != 0 && src->reValues != NULL) {
    cuMemcpyHtoD(dstRePtr, src->reValues, length1 * sizeof(floatt));
  }
  if (dstImPtr != 0 && src->imValues != NULL) {
    cuMemcpyHtoD(dstImPtr, src->imValues, length1 * sizeof(floatt));
  }
}

void CopyDeviceMatrixToDeviceMatrix(math::Matrix* dst,
                                    const math::Matrix* src) {
  uintt length1 = CudaUtils::GetColumns(dst) * CudaUtils::GetColumns(dst);
  uintt length2 = CudaUtils::GetColumns(src) * CudaUtils::GetRows(src);
  length1 = length1 < length2 ? length1 : length2;
  debugAssert(length1 == length2);
  CUdeviceptr dstRePtr =
      reinterpret_cast<CUdeviceptr>(CudaUtils::GetReValues(dst));
  CUdeviceptr dstImPtr =
      reinterpret_cast<CUdeviceptr>(CudaUtils::GetImValues(dst));
  CUdeviceptr srcRePtr =
      reinterpret_cast<CUdeviceptr>(CudaUtils::GetReValues(src));
  CUdeviceptr srcImPtr =
      reinterpret_cast<CUdeviceptr>(CudaUtils::GetImValues(src));
  if (srcRePtr != 0 && dstRePtr != 0) {
    cuMemcpyDtoD(dstRePtr, srcRePtr, length1 * sizeof(floatt));
  }
  if (srcImPtr != 0 && dstImPtr != 0) {
    cuMemcpyDtoD(dstImPtr, srcImPtr, length1 * sizeof(floatt));
  }
}

void CopyHostArraysToDeviceMatrix(math::Matrix* dst, const floatt* rearray,
                                  const floatt* imarray) {
  uintt columns = CudaUtils::GetColumns(dst);
  uintt rows = CudaUtils::GetRows(dst);
  uintt length1 = columns * rows;
  math::Matrix matrix = {columns, rows, const_cast<floatt*>(rearray),
                         const_cast<floatt*>(imarray), columns, rows};
  CopyHostMatrixToDeviceMatrix(dst, &matrix);
}

MatrixEx* NewDeviceMatrixEx() {
  MatrixEx host = {0, 0, 0, 0, 0, 0};
  return CudaUtils::AllocDeviceObj<MatrixEx>(host);
}

MatrixEx** NewDeviceMatrixEx(uintt count) {
  debugAssert(count != 0);
  MatrixEx** array = new MatrixEx* [count];
  MatrixEx* data = static_cast<MatrixEx*>(
      CudaUtils::AllocDeviceMem(count * sizeof(MatrixEx)));
  for (uintt fa = 0; fa < count; ++fa) {
    array[fa] = &data[fa];
  }
  return array;
}

void DeleteDeviceMatrixEx(MatrixEx** matrixEx) {
  CudaUtils::FreeDeviceMem(matrixEx[0]);
  delete[] matrixEx;
}

void DeleteDeviceMatrixEx(MatrixEx* matrixEx) {
  CudaUtils::FreeDeviceObj<MatrixEx>(matrixEx);
}

void SetMatrixEx(MatrixEx* deviceMatrixEx, const MatrixEx* hostMatrixEx) {
  CudaUtils::CopyHostToDevice(deviceMatrixEx, hostMatrixEx, sizeof(MatrixEx));
}

void SetMatrixEx(MatrixEx** deviceMatrixEx, const uintt* buffer, uintt count) {
  debugAssert(count != 0);
  CudaUtils::CopyHostToDevice(deviceMatrixEx[0], buffer,
                              count * sizeof(MatrixEx));
}

void PrintMatrix(const std::string& text, const math::Matrix* matrix) {
  CudaUtils::PrintMatrix(text, matrix);
}

void PrintMatrix(const math::Matrix* matrix) { CudaUtils::PrintMatrix(matrix); }
}
