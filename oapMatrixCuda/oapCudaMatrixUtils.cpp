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

class MatricesMgr
{
  public:
    typedef std::map<const math::Matrix*, math::MatrixInfo> MatrixInfos;
  private:
    MatrixInfos m_matrixInfos;
    MatrixInfos m_deletedMatrixInfos;

    void checkOnDelete()
    {
      if (m_matrixInfos.size() > 0)
      {
        debugError ("Memleak: not deallocated matrices");
        for (MatrixInfos::iterator it = m_matrixInfos.begin(); it != m_matrixInfos.end(); ++it)
        {
          debug("Memleak: dMatrix = %p %s not deallocated", it->first, it->second.toString().c_str());
        }
        debugAssert (false);
      }
    }
  public:

    MatricesMgr ()
    {}

    ~MatricesMgr ()
    {
      checkOnDelete ();
    }

    const MatrixInfos& getAllocated() const
    {
      return m_matrixInfos;
    }

    void add (math::Matrix* dMatrix, const math::MatrixInfo& minfo)
    {
      m_matrixInfos[dMatrix] = minfo;

      MatrixInfos::iterator it = m_deletedMatrixInfos.find (dMatrix);
      if (it != m_deletedMatrixInfos.end ())
      {
        m_deletedMatrixInfos.erase (it);
      }
      auto toInt = [](bool b) -> int
      {
        return b ? 1 : 0;
      };
      size_t size = (toInt (minfo.isRe) + toInt (minfo.isIm)) * minfo.m_matrixDim.columns * minfo.m_matrixDim.rows * sizeof(floatt);
      std::string units = "bytes";
      if (size / 1024 > 0)
      {
        size = size / 1024; units = "KB";
      }
      if (size / 1024 > 0)
      {
        size = size / 1024; units = "MB";
      }
      if (size / 1024 > 0)
      {
        size = size / 1024; units = "GB";
      }
      debug("Allocate: dMatrix = %p %s size: %lu in %s", dMatrix, minfo.toString().c_str(), size, units.c_str());
    }

    math::MatrixInfo remove (const math::Matrix* dMatrix)
    {
      math::MatrixInfo minfo;

      MatrixInfos::iterator it = m_matrixInfos.find(dMatrix);
      if (m_matrixInfos.end() != it)
      {
        m_deletedMatrixInfos[dMatrix] = it->second;
        minfo = it->second;

        m_matrixInfos.erase(it);
      }
      else
      {

        MatrixInfos::iterator it = m_deletedMatrixInfos.find(dMatrix);
        if (it != m_deletedMatrixInfos.end ())
        {
          debugError ("Double deallocation: dMatrix = %p %s", dMatrix, it->second.toString().c_str());
          debugAssert (false);
        }
        else
        {
          debugError ("Not found: dMatrix = %p", dMatrix);
          debugAssert (false);
        }
      }
      return minfo;
    }
};

namespace
{
  MatricesMgr gMatricesMgr;
}

math::Matrix* allocMatrix(bool allocRe, bool allocIm, uintt columns, uintt rows,
                          floatt revalue = 0.f, floatt imvalue = 0.f) {
  CUdeviceptr ptr = CudaUtils::AllocMatrix(allocRe, allocIm, columns, rows);
  math::Matrix* mptr = reinterpret_cast<math::Matrix*>(ptr);

  math::MatrixInfo matrixInfo = math::MatrixInfo (allocRe, allocIm, columns, rows);

  gMatricesMgr.add (mptr, matrixInfo);

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
    math::MatrixInfo minfo = gMatricesMgr.remove (dMatrix);

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
  return gMatricesMgr.getAllocated().at(dMatrix).m_matrixDim.columns;
}

uintt GetRows(const math::Matrix* dMatrix)
{
  return gMatricesMgr.getAllocated().at(dMatrix).m_matrixDim.rows;
}

math::MatrixInfo GetMatrixInfo(const math::Matrix* devMatrix)
{
  return gMatricesMgr.getAllocated().at(devMatrix);
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
