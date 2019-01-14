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

#include <algorithm>
#include <csignal>
#include <string.h>
#include <vector>
#include <netdb.h>
#include <execinfo.h>
#include <map>

#include "oapCudaMatrixUtils.h"

#include "oapHostMatrixUPtr.h"

#include "MatricesList.h"

#include "KernelExecutor.h"

namespace oap
{
namespace cuda
{

namespace
{
MatricesList gMatricesList ("CUDA");
}

math::Matrix* NewHostMatrixCopyOfDeviceMatrix(const math::Matrix* matrix)
{
  CUdeviceptr matrixRePtr = CudaUtils::GetReValuesAddress(matrix);
  CUdeviceptr matrixImPtr = CudaUtils::GetImValuesAddress(matrix);
  uintt columns = CudaUtils::GetColumns(matrix);
  uintt rows = CudaUtils::GetRows(matrix);
  math::Matrix* matrix1 = NULL;
  if (matrixRePtr != 0 && matrixImPtr != 0)
  {
    matrix1 = oap::host::NewMatrix(columns, rows);
  }
  else if (matrixRePtr != 0)
  {
    matrix1 = oap::host::NewReMatrix(columns, rows);
  }
  else if (matrixImPtr != 0)
  {
    matrix1 = oap::host::NewImMatrix(columns, rows);
  }
  oap::cuda::CopyDeviceMatrixToHostMatrix(matrix1, matrix);
  return matrix1;
}

math::Matrix* NewDeviceMatrixHostRef(const math::Matrix* hostMatrix)
{
  return NewDeviceMatrix(hostMatrix, hostMatrix->columns, hostMatrix->rows);
}

math::Matrix* NewDeviceMatrixDeviceRef(const math::Matrix* deviceMatrix)
{
  uintt columns = CudaUtils::GetColumns(deviceMatrix);
  uintt rows = CudaUtils::GetRows(deviceMatrix);
  bool hasRe = CudaUtils::GetReValues(deviceMatrix) != NULL;
  bool hasIm = CudaUtils::GetImValues(deviceMatrix) != NULL;
  return NewDeviceMatrix(hasRe, hasIm, columns, rows);
}

math::Matrix* NewDeviceMatrix(const std::string& matrixStr)
{
  math::Matrix* host = oap::host::NewMatrix(matrixStr);
  math::Matrix* device = NewDeviceMatrixCopy(host);
  oap::host::DeleteMatrix(host);
  return device;
}

math::Matrix* NewDeviceMatrix(const math::MatrixInfo& minfo)
{
  return NewDeviceMatrix (minfo.isRe, minfo.isIm, minfo.m_matrixDim.columns, minfo.m_matrixDim.rows);
}

math::Matrix* NewDeviceMatrixCopy(const math::Matrix* hostMatrix)
{
  math::Matrix* dmatrix = oap::cuda::NewDeviceMatrixHostRef(hostMatrix);
  oap::cuda::CopyHostMatrixToDeviceMatrix(dmatrix, hostMatrix);
  return dmatrix;
}

math::Matrix* allocMatrix(bool allocRe, bool allocIm, uintt columns, uintt rows,
                          floatt revalue = 0.f, floatt imvalue = 0.f)
{

  math::MatrixInfo matrixInfo = math::MatrixInfo (allocRe, allocIm, columns, rows);
  debug ("Try to allocate: %s", matrixInfo.toString().c_str());

  CUdeviceptr ptr = CudaUtils::AllocMatrix(allocRe, allocIm, columns, rows);
  math::Matrix* mptr = reinterpret_cast<math::Matrix*>(ptr);

  gMatricesList.add (mptr, matrixInfo);

  return mptr;
}

math::Matrix* NewDeviceMatrix(const math::Matrix* hostMatrix, uintt columns,
                              uintt rows)
{
  bool allocRe = hostMatrix->reValues != NULL;
  bool allocIm = hostMatrix->imValues != NULL;
  return allocMatrix(allocRe, allocIm, columns, rows);
}

math::Matrix* NewDeviceMatrix(bool isRe, bool isIm, uintt columns, uintt rows)
{
  debugAssert(isRe != false || isIm != false);
  return allocMatrix(isRe, isIm, columns, rows);
}

math::Matrix* NewDeviceReMatrix(uintt columns, uintt rows)
{
  return NewDeviceMatrix(true, false, columns, rows);
}

math::Matrix* NewDeviceImMatrix(uintt columns, uintt rows)
{
  return NewDeviceMatrix(false, true, columns, rows);
}

math::Matrix* NewDeviceMatrix(uintt columns, uintt rows, floatt revalue,
                              floatt imvalue)
{
  debugFuncBegin();
  math::Matrix* dmatrix =
    allocMatrix(true, true, columns, rows, revalue, imvalue);
  debugFuncEnd();
  return dmatrix;
}

void DeleteDeviceMatrix(const math::Matrix* dMatrix)
{
  if (dMatrix != NULL)
  {
    math::MatrixInfo minfo = gMatricesList.remove (dMatrix);

    CUdeviceptr rePtr = reinterpret_cast<CUdeviceptr>(CudaUtils::GetReValues(dMatrix));

    CUdeviceptr imPtr = reinterpret_cast<CUdeviceptr>(CudaUtils::GetImValues(dMatrix));

    CUdeviceptr matrixPtr = reinterpret_cast<CUdeviceptr>(dMatrix);

    CudaUtils::FreeDeviceMem(matrixPtr);
    CudaUtils::FreeDeviceMem(rePtr);
    CudaUtils::FreeDeviceMem(imPtr);

    if (minfo.isInitialized ())
    {
      debugInfo ("Deallocate: cuda matrix = %p %s", dMatrix, minfo.toString().c_str());
    }
  }
}

uintt GetColumns(const math::Matrix* dMatrix)
{
  debugAssert (dMatrix != nullptr);
  return gMatricesList.getMatrixInfo (dMatrix).columns ();
}

uintt GetRows(const math::Matrix* dMatrix)
{
  debugAssert (dMatrix != nullptr);
  return gMatricesList.getMatrixInfo (dMatrix).rows ();
}

math::MatrixInfo GetMatrixInfo(const math::Matrix* dMatrix)
{
  debugAssert (dMatrix != nullptr);
  return gMatricesList.getMatrixInfo (dMatrix);
}

void copyDeviceMatrixToHostMatrix(math::Matrix* dst, const math::Matrix* src, uintt columns, uintt rows)
{
  uintt length = columns * rows;

  floatt* srcRePtr = CudaUtils::GetReValues(src);
  floatt* srcImPtr = CudaUtils::GetImValues(src);

  if (srcRePtr != nullptr && dst->reValues != nullptr)
  {
    CudaUtils::CopyDeviceToHost (dst->reValues, srcRePtr, length * sizeof(floatt));
  }
  if (srcImPtr != nullptr && dst->imValues != nullptr)
  {
    CudaUtils::CopyDeviceToHost (dst->imValues, srcImPtr, length * sizeof(floatt));
  }
}

void CopyDeviceMatrixToHostMatrix(math::Matrix* dst, const math::Matrix* src)
{
  uintt hcolumns = dst->columns;
  uintt hrows = dst->rows;

  uintt dcolumns = CudaUtils::GetColumns (src);
  uintt drows = CudaUtils::GetRows (src);

  debugAssert(hcolumns == dcolumns);
  debugAssert(hrows == drows);

  copyDeviceMatrixToHostMatrix (dst, src, hcolumns, hrows);
}

void CopyDeviceToHost(math::Matrix* dst, const math::Matrix* src)
{
  uintt hcolumns = dst->columns;
  uintt hrows = dst->rows;

  uintt dcolumns = CudaUtils::GetColumns (src);
  uintt drows = CudaUtils::GetRows (src);

  debugAssert(hrows * hcolumns == drows * dcolumns);

  copyDeviceMatrixToHostMatrix (dst, src, hcolumns, hrows);
}

void copyHostMatrixToDeviceMatrix(math::Matrix* dst, const math::Matrix* src, uintt columns, uintt rows)
{
  uintt length = columns * rows;

  floatt* dstRePtr = CudaUtils::GetReValues(dst);
  floatt* dstImPtr = CudaUtils::GetImValues(dst);

  if (dstRePtr != nullptr && src->reValues != nullptr)
  {
    CudaUtils::CopyHostToDevice(dstRePtr, src->reValues, length * sizeof(floatt));
  }
  if (dstImPtr != nullptr && src->imValues != nullptr)
  {
    CudaUtils::CopyHostToDevice(dstImPtr, src->imValues, length * sizeof(floatt));
  }
}

void CopyHostMatrixToDeviceMatrix(math::Matrix* dst, const math::Matrix* src)
{
  uintt hcolumns = src->columns;
  uintt hrows = src->rows;

  uintt dcolumns = CudaUtils::GetColumns (dst);
  uintt drows = CudaUtils::GetRows (dst);

  debugAssert(hcolumns == dcolumns);
  debugAssert(hrows == drows);

  copyHostMatrixToDeviceMatrix (dst, src, hcolumns, hrows);
}

void CopyHostToDevice(math::Matrix* dst, const math::Matrix* src)
{
  uintt hcolumns = src->columns;
  uintt hrows = src->rows;

  uintt dcolumns = CudaUtils::GetColumns (dst);
  uintt drows = CudaUtils::GetRows (dst);

  debugAssert(hrows * hcolumns == drows * dcolumns);

  copyHostMatrixToDeviceMatrix (dst, src, hcolumns, hrows);
}

void copyDeviceMatrixToDeviceMatrix(math::Matrix* dst, const math::Matrix* src, uintt columns, uintt rows)
{
  uintt length = columns * rows;

  floatt* dstRePtr = CudaUtils::GetReValues(dst);
  floatt* dstImPtr = CudaUtils::GetImValues(dst);
  floatt* srcRePtr = CudaUtils::GetReValues(src);
  floatt* srcImPtr = CudaUtils::GetImValues(src);

  if (srcRePtr != nullptr && dstRePtr != nullptr)
  {
    CudaUtils::CopyDeviceToDevice (dstRePtr, srcRePtr, length * sizeof(floatt));
  }
  if (srcImPtr != nullptr && dstImPtr != nullptr)
  {
    CudaUtils::CopyDeviceToDevice (dstImPtr, srcImPtr, length * sizeof(floatt));
  }
}

void CopyDeviceMatrixToDeviceMatrix(math::Matrix* dst, const math::Matrix* src)
{
  uintt dcolumns1 = CudaUtils::GetColumns (src);
  uintt drows1 = CudaUtils::GetRows (src);

  uintt dcolumns2 = CudaUtils::GetColumns (dst);
  uintt drows2 = CudaUtils::GetRows (dst);

  debugAssert(dcolumns1 == dcolumns2);
  debugAssert(drows1 == drows2);

  copyDeviceMatrixToDeviceMatrix (dst, src, dcolumns1, drows1);
}

void CopyDeviceToDevice (math::Matrix* dst, const math::Matrix* src)
{
  uintt dcolumns1 = CudaUtils::GetColumns (src);
  uintt drows1 = CudaUtils::GetRows (src);

  uintt dcolumns2 = CudaUtils::GetColumns (dst);
  uintt drows2 = CudaUtils::GetRows (dst);

  debugAssert(drows1 * dcolumns1 == drows2 * dcolumns2);

  copyDeviceMatrixToDeviceMatrix (dst, src, dcolumns1, drows1);
}

void SetMatrix(math::Matrix* matrix, math::Matrix* matrix1, uintt column, uintt row)
{
  uintt columns = oap::cuda::GetColumns(matrix);
  uintt rows = oap::cuda::GetRows(matrix);

  uintt columns1 = oap::cuda::GetColumns(matrix1);
  uintt rows1 = oap::cuda::GetRows(matrix1);

  debugAssert (columns1 + column <= columns);
  debugAssert (rows1 + row <= rows);

  floatt* dstreptr = CudaUtils::GetReValues(matrix);
  floatt* dstimptr = CudaUtils::GetImValues(matrix);

  floatt* srcreptr = CudaUtils::GetReValues(matrix1);
  floatt* srcimptr = CudaUtils::GetImValues(matrix1);

  debugAssert (!(dstreptr == nullptr && srcreptr != nullptr));
  debugAssert (!(dstreptr != nullptr && srcreptr == nullptr));

  debugAssert (!(dstimptr == nullptr && srcimptr != nullptr));
  debugAssert (!(dstimptr != nullptr && srcimptr == nullptr));

  for (uintt rowIdx = 0; rowIdx < rows1; ++rowIdx)
  {
    uintt index = column + columns * (row + rowIdx);
    if (dstreptr != NULL && srcreptr != NULL)
    {
      CudaUtils::CopyDeviceToDevice(dstreptr + index, srcreptr + columns1 * rowIdx, columns1 * sizeof(floatt));
    }
    if (dstimptr != NULL && srcimptr != NULL)
    {
      CudaUtils::CopyDeviceToDevice(dstimptr + index, srcimptr + columns1 * rowIdx, columns1 * sizeof(floatt));
    }
  }
}

void CopyHostArraysToDeviceMatrix(math::Matrix* dst, const floatt* rearray,
                                  const floatt* imarray)
{
  uintt columns = CudaUtils::GetColumns(dst);
  uintt rows = CudaUtils::GetRows(dst);
  uintt length1 = columns * rows;
  math::Matrix matrix = {columns, rows, const_cast<floatt*>(rearray),
                         const_cast<floatt*>(imarray), columns, rows
                        };
  CopyHostMatrixToDeviceMatrix(dst, &matrix);
}

MatrixEx* NewDeviceMatrixEx()
{
  MatrixEx host = {0, 0, 0, 0, 0, 0};
  return CudaUtils::AllocDeviceObj<MatrixEx>(host);
}

MatrixEx** NewDeviceMatrixEx(uintt count)
{
  debugAssert(count != 0);
  MatrixEx** array = new MatrixEx* [count];
  MatrixEx* data = static_cast<MatrixEx*>(
                     CudaUtils::AllocDeviceMem(count * sizeof(MatrixEx)));
  for (uintt fa = 0; fa < count; ++fa)
  {
    array[fa] = &data[fa];
  }
  return array;
}

void DeleteDeviceMatrixEx(MatrixEx** matrixEx)
{
  CudaUtils::FreeDeviceMem(matrixEx[0]);
  delete[] matrixEx;
}

void DeleteDeviceMatrixEx(MatrixEx* matrixEx)
{
  CudaUtils::FreeDeviceObj<MatrixEx>(matrixEx);
}

void SetMatrixEx(MatrixEx** deviceMatrixEx, const uintt* buffer, uintt count)
{
  debugAssert(count != 0);
  for (uintt fa = 0; fa < count; ++fa)
  {
    CudaUtils::CopyHostToDevice(
      deviceMatrixEx[fa], &buffer[fa * (sizeof(MatrixEx) / sizeof(uintt))],
      sizeof(MatrixEx));
  }
}

void SetMatrixEx(MatrixEx* deviceMatrixEx, const MatrixEx* hostMatrixEx)
{
  CudaUtils::CopyHostToDevice(deviceMatrixEx, hostMatrixEx, sizeof(MatrixEx));
}

void GetMatrixEx(MatrixEx* hostMatrixEx, const MatrixEx* deviceMatrixEx)
{
  CudaUtils::CopyDeviceToHost(hostMatrixEx, deviceMatrixEx, sizeof(MatrixEx));
}

void PrintMatrix(const std::string& text, const math::Matrix* matrix, floatt zeroLimit)
{
  CudaUtils::PrintMatrix(text, matrix, zeroLimit);
}

void PrintMatrix(const math::Matrix* matrix)
{
  CudaUtils::PrintMatrix(matrix);
}

void SetReValue(math::Matrix* matrix, floatt value, uintt column, uintt row)
{
  uintt columns = CudaUtils::GetColumns(matrix);
  SetReValue(matrix, value, column + columns * row);
}

void SetReValue(math::Matrix* matrix, floatt value, uintt index)
{
  CudaUtils::SetReValue(matrix, index, value);
}

void SetImValue(math::Matrix* matrix, floatt value, uintt column, uintt row)
{
  uintt columns = CudaUtils::GetColumns(matrix);
  SetImValue(matrix, value, column + columns * row);
}

void SetImValue(math::Matrix* matrix, floatt value, uintt index)
{
  CudaUtils::SetImValue(matrix, index, value);
}

void SetValue(math::Matrix* matrix, floatt revalue, floatt imvalue,
              uintt column, uintt row)
{
  uintt columns = CudaUtils::GetColumns(matrix);
  SetValue(matrix, revalue, imvalue, column + columns * row);
}

void SetValue(math::Matrix* matrix, floatt revalue, floatt imvalue,
              uintt index)
{
  CudaUtils::SetReValue(matrix, index, revalue);
  CudaUtils::SetImValue(matrix, index, imvalue);
}

void SetReMatrix(math::Matrix* matrix, math::Matrix* matrix1,
                 uintt column, uintt row)
{
  uintt columns = CudaUtils::GetColumns(matrix);
  uintt columns1 = CudaUtils::GetColumns(matrix1);
  uintt rows1 = CudaUtils::GetRows(matrix1);

  floatt* dstreptr = CudaUtils::GetReValues(matrix);

  floatt* srcreptr = CudaUtils::GetReValues(matrix1);

  for (uintt fa = 0; fa < rows1; ++fa)
  {
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

  for (uintt fa = 0; fa < rows1; ++fa)
  {
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

math::Matrix* ReadMatrix(const std::string& path)
{
  math::Matrix* hostMatrix = oap::host::ReadMatrix(path);
  math::Matrix* devMatrix = oap::cuda::NewDeviceMatrixCopy(hostMatrix);
  oap::host::DeleteMatrix(hostMatrix);
  return devMatrix;
}

bool WriteMatrix(const std::string& path, const math::Matrix* devMatrix)
{
  math::MatrixInfo matrixInfo = oap::cuda::GetMatrixInfo(devMatrix);
  math::Matrix* hostMatrix = oap::host::NewMatrix(matrixInfo);
  oap::cuda::CopyDeviceMatrixToHostMatrix(hostMatrix, devMatrix);
  bool status = oap::host::WriteMatrix(path, hostMatrix);
  oap::host::DeleteMatrix(hostMatrix);
  return status;
}

void SaveMatrixInfo (const math::MatrixInfo& minfo, utils::ByteBuffer& buffer)
{
  oap::host::SaveMatrixInfo (minfo, buffer);
}

void SaveMatrix (const math::Matrix* matrix, utils::ByteBuffer& buffer)
{
  bool isMatrix = (matrix != nullptr);

  buffer.push_back (isMatrix);

  if (!isMatrix)
  {
    return;
  }

  auto minfo = oap::cuda::GetMatrixInfo (matrix);
  SaveMatrixInfo (minfo, buffer);

  oap::HostMatrixUPtr hmatrix = oap::host::NewMatrix (minfo);

  oap::cuda::CopyDeviceMatrixToHostMatrix (hmatrix, matrix);

  if (minfo.isRe)
  {
    buffer.push_back (hmatrix->reValues, minfo.length ());
  }

  if (minfo.isIm)
  {
    buffer.push_back (hmatrix->imValues, minfo.length ());
  }
}

math::Matrix* LoadMatrix (const utils::ByteBuffer& buffer)
{
  oap::HostMatrixUPtr hmatrix = oap::host::LoadMatrix (buffer);

  if (!hmatrix)
  {
    return nullptr;
  }

  math::Matrix* matrix = oap::cuda::NewDeviceMatrixCopy (hmatrix);

  return matrix;
}

math::MatrixInfo LoadMatrixInfo (const utils::ByteBuffer& buffer)
{
  return oap::host::LoadMatrixInfo (buffer);
}

}
}
