/*
 * Copyright 2016 - 2018 Marcin Matula
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

#include "oapCudaMatrixUtils.h"
#include "oapHostMatrixUtils.h"
#include "KernelExecutor.h"
#include <csignal>
#include <string.h>
#include <vector>
#include <algorithm>
#include <netdb.h>
#include <execinfo.h>
#include <map>

namespace oap
{
namespace cuda
{

math::Matrix* NewHostMatrixCopyOfDeviceMatrix(const math::Matrix* matrix) {
  CUdeviceptr matrixRePtr = CudaUtils::GetReValuesAddress(matrix);
  CUdeviceptr matrixImPtr = CudaUtils::GetImValuesAddress(matrix);
  uintt columns = CudaUtils::GetColumns(matrix);
  uintt rows = CudaUtils::GetRows(matrix);
  math::Matrix* matrix1 = NULL;
  if (matrixRePtr != 0 && matrixImPtr != 0) {
    matrix1 = oap::host::NewMatrix(columns, rows);
  } else if (matrixRePtr != 0) {
    matrix1 = oap::host::NewReMatrix(columns, rows);
  } else if (matrixImPtr != 0) {
    matrix1 = oap::host::NewImMatrix(columns, rows);
  }
  oap::cuda::CopyDeviceMatrixToHostMatrix(matrix1, matrix);
  return matrix1;
}

math::Matrix* NewDeviceMatrixHostRef(const math::Matrix* hostMatrix) {
  return NewDeviceMatrix(hostMatrix, hostMatrix->columns, hostMatrix->rows);
}

math::Matrix* NewDeviceMatrixDeviceRef(const math::Matrix* deviceMatrix) {
  uintt columns = CudaUtils::GetColumns(deviceMatrix);
  uintt rows = CudaUtils::GetRows(deviceMatrix);
  bool hasRe = CudaUtils::GetReValues(deviceMatrix) != NULL;
  bool hasIm = CudaUtils::GetImValues(deviceMatrix) != NULL;
  return NewDeviceMatrix(hasRe, hasIm, columns, rows);
}

math::Matrix* NewDeviceMatrix(const std::string& matrixStr) {
  math::Matrix* host = oap::host::NewMatrix(matrixStr);
  math::Matrix* device = NewDeviceMatrixCopy(host);
  oap::host::DeleteMatrix(host);
  return device;
}

math::Matrix* NewDeviceMatrixCopy(const math::Matrix* hostMatrix) {
  math::Matrix* dmatrix = oap::cuda::NewDeviceMatrixHostRef(hostMatrix);
  oap::cuda::CopyHostMatrixToDeviceMatrix(dmatrix, hostMatrix);
  return dmatrix;
}

typedef std::map<const math::Matrix*, math::MatrixInfo> MatrixInfos;

namespace
{
MatrixInfos gMatrixInfos;
MatrixInfos gDeletedMatrixInfos;
}

void gRegister (math::Matrix* dMatrix, const math::MatrixInfo& minfo)
{
  gMatrixInfos[dMatrix] = minfo;

  MatrixInfos::iterator it = gDeletedMatrixInfos.find (dMatrix);
  if (it != gDeletedMatrixInfos.end ())
  {
    gDeletedMatrixInfos.erase (it);
  }

  debug("Allocate: dMatrix = %p %s", dMatrix, minfo.toString().c_str());
}

math::MatrixInfo gUnregister (const math::Matrix* dMatrix)
{
  math::MatrixInfo minfo;

  MatrixInfos::iterator it = gMatrixInfos.find(dMatrix);
  if (gMatrixInfos.end() != it)
  {
    gDeletedMatrixInfos[dMatrix] = it->second;
    minfo = it->second;

    gMatrixInfos.erase(it);
  }
  else
  {

    MatrixInfos::iterator it = gDeletedMatrixInfos.find(dMatrix);
    if (it != gDeletedMatrixInfos.end ())
    {
      debugError ("Double deallocation: dMatrix = %p %s", dMatrix, it->second.toString().c_str());
      //debugAssert (false);
    }
    else
    {
      debugError ("Not found: dMatrix = %p", dMatrix);
      //debugAssert (false);
    }
  }

  return minfo;
}


math::Matrix* allocMatrix(bool allocRe, bool allocIm, uintt columns, uintt rows,
                          floatt revalue = 0.f, floatt imvalue = 0.f) {
  CUdeviceptr ptr = CudaUtils::AllocMatrix(allocRe, allocIm, columns, rows);
  math::Matrix* mptr = reinterpret_cast<math::Matrix*>(ptr);

  math::MatrixInfo matrixInfo = math::MatrixInfo (allocRe, allocIm, columns, rows);

  gRegister (mptr, matrixInfo);

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

math::Matrix* NewDeviceReMatrix(uintt columns, uintt rows) {
  return NewDeviceMatrix(true, false, columns, rows);
}

math::Matrix* NewDeviceImMatrix(uintt columns, uintt rows) {
  return NewDeviceMatrix(false, true, columns, rows);
}

math::Matrix* NewDeviceMatrix(uintt columns, uintt rows, floatt revalue,
                              floatt imvalue) {
  debugFuncBegin();
  math::Matrix* dmatrix =
      allocMatrix(true, true, columns, rows, revalue, imvalue);
  debugFuncEnd();
  return dmatrix;
}

void DeleteDeviceMatrix(const math::Matrix* dMatrix) {
  if (dMatrix != NULL)
  {
    math::MatrixInfo minfo = gUnregister (dMatrix);

    CUdeviceptr rePtr = reinterpret_cast<CUdeviceptr>(CudaUtils::GetReValues(dMatrix));

    CUdeviceptr imPtr = reinterpret_cast<CUdeviceptr>(CudaUtils::GetImValues(dMatrix));

    CUdeviceptr matrixPtr = reinterpret_cast<CUdeviceptr>(dMatrix);

    CudaUtils::FreeDeviceMem(matrixPtr);
    CudaUtils::FreeDeviceMem(rePtr);
    CudaUtils::FreeDeviceMem(imPtr);

    if (minfo.isInitialized ())
    {
      debug ("Deallocate: dMatrix = %p %s", dMatrix, minfo.toString().c_str());
    }
  }
}

uintt GetColumns(const math::Matrix* dMatrix)
{
  return gMatrixInfos[dMatrix].m_matrixDim.columns;
}

uintt GetRows(const math::Matrix* dMatrix)
{
  return gMatrixInfos[dMatrix].m_matrixDim.rows;
}

math::MatrixInfo GetMatrixInfo(const math::Matrix* devMatrix)
{
  return gMatrixInfos[devMatrix];
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
  uintt length1 = CudaUtils::GetColumns(dst) * CudaUtils::GetRows(dst);
  uintt length2 = CudaUtils::GetColumns(src) * CudaUtils::GetRows(src);
  length1 = length1 < length2 ? length1 : length2;
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

void SetMatrixEx(MatrixEx** deviceMatrixEx, const uintt* buffer, uintt count) {
  debugAssert(count != 0);
  for (uintt fa = 0; fa < count; ++fa) {
    CudaUtils::CopyHostToDevice(
        deviceMatrixEx[fa], &buffer[fa * (sizeof(MatrixEx) / sizeof(uintt))],
        sizeof(MatrixEx));
  }
}

void SetMatrixEx(MatrixEx* deviceMatrixEx, const MatrixEx* hostMatrixEx) {
  CudaUtils::CopyHostToDevice(deviceMatrixEx, hostMatrixEx, sizeof(MatrixEx));
}

void GetMatrixEx(MatrixEx* hostMatrixEx, const MatrixEx* deviceMatrixEx) {
  CudaUtils::CopyDeviceToHost(hostMatrixEx, deviceMatrixEx, sizeof(MatrixEx));
}

void PrintMatrix(const std::string& text, const math::Matrix* matrix, floatt zeroLimit)
{
  CudaUtils::PrintMatrix(text, matrix, zeroLimit);
}

void PrintMatrix(const math::Matrix* matrix) { CudaUtils::PrintMatrix(matrix); }

void SetReValue(math::Matrix* matrix, floatt value, uintt column, uintt row) {
  uintt columns = CudaUtils::GetColumns(matrix);
  SetReValue(matrix, value, column + columns * row);
}

void SetReValue(math::Matrix* matrix, floatt value, uintt index) {
  CudaUtils::SetReValue(matrix, index, value);
}

void SetImValue(math::Matrix* matrix, floatt value, uintt column, uintt row) {
  uintt columns = CudaUtils::GetColumns(matrix);
  SetImValue(matrix, value, column + columns * row);
}

void SetImValue(math::Matrix* matrix, floatt value, uintt index) {
  CudaUtils::SetImValue(matrix, index, value);
}

void SetValue(math::Matrix* matrix, floatt revalue, floatt imvalue,
              uintt column, uintt row) {
  uintt columns = CudaUtils::GetColumns(matrix);
  SetValue(matrix, revalue, imvalue, column + columns * row);
}

void SetValue(math::Matrix* matrix, floatt revalue, floatt imvalue,
              uintt index) {
  CudaUtils::SetReValue(matrix, index, revalue);
  CudaUtils::SetImValue(matrix, index, imvalue);
}

void SetMatrix(math::Matrix* matrix, math::Matrix* matrix1, uintt column,
               uintt row)
{
  uintt columns = CudaUtils::GetColumns(matrix);
  uintt columns1 = CudaUtils::GetColumns(matrix1);
  uintt rows1 = CudaUtils::GetRows(matrix1);

  floatt* dstreptr = CudaUtils::GetReValues(matrix);
  floatt* dstimptr = CudaUtils::GetImValues(matrix);

  floatt* srcreptr = CudaUtils::GetReValues(matrix1);
  floatt* srcimptr = CudaUtils::GetImValues(matrix1);

  for (uintt fa = 0; fa < rows1; ++fa) {
    uintt index = column + columns * (row + fa);
    if (dstreptr != NULL && srcreptr != NULL) {
      CudaUtils::CopyDeviceToDevice(dstreptr + index, srcreptr + columns1 * fa,
           columns1 * sizeof(floatt));
    }
    if (dstimptr != NULL && srcimptr != NULL) {
      CudaUtils::CopyDeviceToDevice(dstimptr + index, srcimptr + columns1 * fa,
          columns1 * sizeof(floatt));
    }
  }
}

void SetReMatrix(math::Matrix* matrix, math::Matrix* matrix1,
                uintt column, uintt row)
{
  uintt columns = CudaUtils::GetColumns(matrix);
  uintt columns1 = CudaUtils::GetColumns(matrix1);
  uintt rows1 = CudaUtils::GetRows(matrix1);

  floatt* dstreptr = CudaUtils::GetReValues(matrix);

  floatt* srcreptr = CudaUtils::GetReValues(matrix1);

  for (uintt fa = 0; fa < rows1; ++fa) {
    uintt index = column + columns * (row + fa);
    CudaUtils::CopyDeviceToDevice(dstreptr + index, srcreptr + columns1 * fa,
                                  columns1 * sizeof(floatt));
  }
}

void SetImMatrix(math::Matrix* matrix, math::Matrix* matrix1, uintt column,
               uintt row)
{
  uintt columns = CudaUtils::GetColumns(matrix);
  uintt columns1 = CudaUtils::GetColumns(matrix1);
  uintt rows1 = CudaUtils::GetRows(matrix1);

  floatt* dstimptr = CudaUtils::GetImValues(matrix);

  floatt* srcimptr = CudaUtils::GetImValues(matrix1);

  for (uintt fa = 0; fa < rows1; ++fa) {
    uintt index = column + columns * (row + fa);
    CudaUtils::CopyDeviceToDevice(dstimptr + index, srcimptr + columns1 * fa,
                                  columns1 * sizeof(floatt));
  }
}

void PrintMatrixInfo(const std::string& msg, const math::Matrix* devMatrix)
{
  math::MatrixInfo minfo = GetMatrixInfo (devMatrix);
  printf ("%s (columns=%u rows=%u) (isRe=%d isIm=%d)\n",
          msg.c_str(), minfo.m_matrixDim.columns, minfo.m_matrixDim.rows, minfo.isRe, minfo.isIm);
}

math::Matrix* ReadMatrix(const std::string& path) {
  math::Matrix* hostMatrix = oap::host::ReadMatrix(path);
  math::Matrix* devMatrix = oap::cuda::NewDeviceMatrixCopy(hostMatrix);
  oap::host::DeleteMatrix(hostMatrix);
  return devMatrix;
}

bool WriteMatrix(const std::string& path, const math::Matrix* devMatrix) {
  math::MatrixInfo matrixInfo = oap::cuda::GetMatrixInfo(devMatrix);
  math::Matrix* hostMatrix = oap::host::NewMatrix(matrixInfo);
  oap::cuda::CopyDeviceMatrixToHostMatrix(hostMatrix, devMatrix);
  bool status = oap::host::WriteMatrix(path, hostMatrix);
  oap::host::DeleteMatrix(hostMatrix);
  return status;
}

}
}
