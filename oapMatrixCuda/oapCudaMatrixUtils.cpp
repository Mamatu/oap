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
#include <csignal>
#include <string.h>
#include <vector>
#include <netdb.h>
#include <execinfo.h>
#include <map>

#include "oapCudaMatrixUtils.h"
#include "oapCudaMemoryApi.h"

#include "oapHostMatrixUPtr.h"
#include "GenericCoreApi.h"

#include "MatricesList.h"
#include "MatrixUtils.h"
#include "MatrixAPI.h"

#include "KernelExecutor.h"

namespace oap
{
namespace cuda
{

namespace
{
MatricesListExt<math::Matrix> g_matricesList ("CUDA");

void registerMatrix (math::Matrix* matrix, const math::Matrix& hostRefMatrix, const math::MatrixInfo& matrixInfo)
{
  logTrace ("Allocated: %s", matrixInfo.toString().c_str());

  logTrace ("Matrix allocation: %p", mptr);
  g_matricesList.add (matrix, std::make_pair (matrixInfo, hostRefMatrix));
}

oap::Memory allocPart (oap::Memory* memoryPtr, oap::MemoryRegion* regionPtr, bool alloc, uintt columns, uintt rows)
{
  oap::Memory memory = {nullptr, {0, 0}};
  oap::MemoryRegion region = {{0, 0}, {0, 0}};

  if (alloc)
  {
    memory = oap::cuda::NewMemory ({columns, rows});
    region = {{0, 0}, {columns, rows}};
    CudaUtils::CopyHostToDevice (&memoryPtr, &memory, sizeof(memory));
    CudaUtils::CopyHostToDevice (&regionPtr, &region, sizeof(region));
  }

  return memory;
}

math::Matrix* allocMatrix_AllocMemory (bool allocRe, bool allocIm, uintt columns, uintt rows)
{
  math::Matrix* matrix = static_cast<math::Matrix*>(CudaUtils::AllocDeviceMem (sizeof(math::Matrix)));
  math::Matrix hostRefMatrix;

  oap::Memory reMem = allocPart (&(matrix->re), &(matrix->reReg), allocRe, columns, rows);
  oap::Memory imMem = allocPart (&(matrix->im), &(matrix->imReg), allocIm, columns, rows);

  hostRefMatrix.re = reMem;
  hostRefMatrix.reReg = {{0, 0}, {columns, rows}};
  hostRefMatrix.im = imMem;
  hostRefMatrix.imReg = {{0, 0}, {columns, rows}};

  registerMatrix (matrix, hostRefMatrix, math::MatrixInfo(allocRe, allocIm, columns, rows));

  return matrix;
}

#if 0
math::Matrix* allocMatrix_ReuseMemory (uintt columns, uintt rows, floatt* revalues, floatt* imvalues)
{
  CudaUtils::CuDevicePtrs devicePtrs;

  CUdeviceptr ptr = CudaUtils::AllocMatrix_ReuseMemory (columns, rows, revalues, imvalues, &devicePtrs);
  math::Matrix* mptr = reinterpret_cast<math::Matrix*>(ptr);

  registerMatrix (mptr, devicePtrs, math::MatrixInfo (revalues != nullptr, imvalues != nullptr, columns, rows));

  return mptr;
}
#endif
}

math::Matrix* NewHostMatrixCopyOfDeviceMatrix (const math::Matrix* matrix)
{
  auto minfo = GetMatrixInfo (matrix);

  math::Matrix* hostMatrix = oap::host::NewMatrix (minfo);
  oap::cuda::CopyDeviceMatrixToHostMatrix(hostMatrix, matrix);

  return hostMatrix;
}

math::Matrix* NewDeviceMatrixHostRef(const math::Matrix* hostMatrix)
{
  return NewDeviceMatrix(hostMatrix, gColumns (hostMatrix), gRows (hostMatrix));
}

math::Matrix* NewDeviceMatrixDeviceRef(const math::Matrix* deviceMatrix)
{
  auto minfo = oap::cuda::GetMatrixInfo (deviceMatrix);
  return NewDeviceMatrix (minfo.isRe, minfo.isIm, minfo.columns (), minfo.rows ());
}

math::Matrix* NewDeviceMatrix(const std::string& matrixStr)
{
  math::Matrix* host = oap::host::NewMatrix(matrixStr);
  math::Matrix* device = NewDeviceMatrixCopyOfHostMatrix (host);
  oap::host::DeleteMatrix(host);
  return device;
}

#if 0
math::Matrix* NewShareDeviceMatrix(uintt columns, uintt rows, math::Matrix* src)
{
  return oap::generic::newShareMatrix (columns, rows, src, oap::cuda::GetMatrixInfo, allocMatrix_ReuseMemory, CudaUtils::GetValue);
}
#endif

math::Matrix* NewDeviceMatrix(const math::MatrixInfo& minfo)
{
  return NewDeviceMatrix (minfo.isRe, minfo.isIm, minfo.m_matrixDim.columns, minfo.m_matrixDim.rows);
}

math::Matrix* NewDeviceMatrixCopyOfHostMatrix(const math::Matrix* hostMatrix)
{
  math::Matrix* dmatrix = oap::cuda::NewDeviceMatrixHostRef(hostMatrix);
  oap::cuda::CopyHostMatrixToDeviceMatrix(dmatrix, hostMatrix);
  return dmatrix;
}

math::Matrix* NewDeviceMatrix (const math::Matrix* hostMatrix, uintt columns, uintt rows)
{
  bool allocRe = hostMatrix->re.ptr != NULL;
  bool allocIm = hostMatrix->im.ptr != NULL;
  return allocMatrix_AllocMemory (allocRe, allocIm, columns, rows);
}

math::Matrix* NewDeviceMatrix(bool isRe, bool isIm, uintt columns, uintt rows)
{
  debugAssert(isRe != false || isIm != false);
  return allocMatrix_AllocMemory (isRe, isIm, columns, rows);
}

math::Matrix* NewDeviceReMatrix(uintt columns, uintt rows)
{
  return NewDeviceMatrix(true, false, columns, rows);
}

math::Matrix* NewDeviceImMatrix(uintt columns, uintt rows)
{
  return NewDeviceMatrix(false, true, columns, rows);
}

math::Matrix* NewDeviceMatrix (uintt columns, uintt rows)
{
  math::Matrix* dmatrix = allocMatrix_AllocMemory (true, true, columns, rows);
  return dmatrix;
}

void DeleteDeviceMatrix(const math::Matrix* dMatrix)
{
  if (dMatrix != NULL)
  {
    auto minfo = g_matricesList.remove (dMatrix);

    math::Matrix hm1lvl = minfo.second;

    oap::cuda::DeleteMemory (hm1lvl.re);
    oap::cuda::DeleteMemory (hm1lvl.im);
    CudaUtils::FreeDeviceMem (dMatrix);

    logTrace ("Matrix deallocation: %p", matrixPtr);

    if (minfo.first.isInitialized ())
    {
      logTrace ("Deallocate: cuda matrix = %p %s", dMatrix, minfo.toString().c_str());
    }
  }
}

uintt GetColumns(const math::Matrix* dMatrix)
{
  debugAssert (dMatrix != nullptr);
  return g_matricesList.getUserData (dMatrix).first.columns ();
}

uintt GetRows(const math::Matrix* dMatrix)
{
  debugAssert (dMatrix != nullptr);
  return g_matricesList.getUserData (dMatrix).first.rows ();
}

math::Matrix GetRefHostMatrix (const math::Matrix* dMatrix)
{
  debugAssert (dMatrix != nullptr);

  auto userData = g_matricesList.getUserData (dMatrix);

  return userData.second;
}

floatt* GetReValuesPtr (const math::Matrix* dMatrix)
{
  return GetRefHostMatrix (dMatrix).re.ptr;
}

floatt* GetImValuesPtr (const math::Matrix* dMatrix)
{
  return GetRefHostMatrix (dMatrix).im.ptr;
}

math::MatrixInfo GetMatrixInfo(const math::Matrix* dMatrix)
{
  debugAssert (dMatrix != nullptr);
  return g_matricesList.getUserData (dMatrix).first;
}

bool IsCudaMatrix(const math::Matrix* devMatrix)
{
	return g_matricesList.contains (devMatrix);
}

inline void copyDeviceMatrixToHostMatrix (math::Matrix* dst, const oap::MemoryLoc& loc, const math::Matrix* src, const oap::MemoryRegion& reg)
{
  auto srcRef = GetRefHostMatrix (src);
  if (dst->re.ptr && srcRef.re.ptr) { oap::cuda::CopyDeviceToHost (dst->re, loc, srcRef.re, reg); }
  if (dst->im.ptr && srcRef.im.ptr) { oap::cuda::CopyDeviceToHost (dst->im, loc, srcRef.im, reg); }
}

inline void copyDeviceMatrixToHostMatrix (math::Matrix* dst, const math::Matrix* src)
{
#if 0
  oap::generic::MatrixMemoryApi<decltype (oap::host::GetMatrixInfo), decltype (oap::host::ToHost)> dstApi (oap::host::GetMatrixInfo, oap::host::ToHost);
  oap::generic::MatrixMemoryApi<decltype (oap::cuda::GetMatrixInfo), decltype (CudaUtils::ToHost)> srcApi (oap::cuda::GetMatrixInfo, CudaUtils::ToHost);
  oap::generic::copyMatrixToMatrix (dst, src, CudaUtils::CopyDeviceToHost, dstApi, srcApi, check);
#endif

  auto srcRef = GetRefHostMatrix (src);
  if (dst->re.ptr && srcRef.re.ptr) { oap::cuda::CopyDeviceToHost (dst->re, srcRef.re); }
  if (dst->im.ptr && srcRef.im.ptr) { oap::cuda::CopyDeviceToHost (dst->im, srcRef.im); }
}

inline void copyHostMatrixToDeviceMatrix (math::Matrix* dst, const oap::MemoryLoc& loc, const math::Matrix* src, const oap::MemoryRegion& reg)
{
  auto dstRef = GetRefHostMatrix (dst);
  if (src->re.ptr && dstRef.re.ptr) { oap::cuda::CopyHostToDevice (dstRef.re, loc, src->re, reg); }
  if (src->im.ptr && dstRef.im.ptr) { oap::cuda::CopyHostToDevice (dstRef.im, loc, src->im, reg); }
}

inline void copyHostMatrixToDeviceMatrix (math::Matrix* dst, const math::Matrix* src)
{
#if 0
  oap::generic::MatrixMemoryApi<decltype (oap::cuda::GetMatrixInfo), decltype (CudaUtils::ToHost)> dstApi (oap::cuda::GetMatrixInfo, CudaUtils::ToHost);
  oap::generic::MatrixMemoryApi<decltype (oap::host::GetMatrixInfo), decltype (oap::host::ToHost)> srcApi (oap::host::GetMatrixInfo, oap::host::ToHost);
  oap::generic::copyMatrixToMatrix (dst, src, CudaUtils::CopyHostToDevice, dstApi, srcApi, check);
#endif

  auto dstRef = GetRefHostMatrix (dst);
  if (src->re.ptr && dstRef.re.ptr) { oap::cuda::CopyHostToDevice (dstRef.re, src->re); }
  if (src->im.ptr && dstRef.im.ptr) { oap::cuda::CopyHostToDevice (dstRef.im, src->im); }
}

inline void copyDeviceMatrixToDeviceMatrix (math::Matrix* dst, const oap::MemoryLoc& loc, const math::Matrix* src, const oap::MemoryRegion& reg)
{
  auto srcRef = GetRefHostMatrix (src);
  auto dstRef = GetRefHostMatrix (dst);
  if (srcRef.re.ptr && dstRef.re.ptr) { oap::cuda::CopyDeviceToDevice (dstRef.re, loc, srcRef.re, reg); }
  if (srcRef.im.ptr && dstRef.im.ptr) { oap::cuda::CopyDeviceToDevice (dstRef.im, loc, srcRef.im, reg); }
}

inline void copyDeviceMatrixToDeviceMatrix (math::Matrix* dst, const math::Matrix* src)
{
#if 0
  oap::generic::MatrixMemoryApi<decltype (oap::cuda::GetMatrixInfo), decltype (CudaUtils::ToHost)> dstApi (oap::cuda::GetMatrixInfo, CudaUtils::ToHost);
  oap::generic::MatrixMemoryApi<decltype (oap::cuda::GetMatrixInfo), decltype (CudaUtils::ToHost)> srcApi (oap::cuda::GetMatrixInfo, CudaUtils::ToHost);
  oap::generic::copyMatrixToMatrix (dst, src, CudaUtils::CopyDeviceToDevice, dstApi, srcApi, check);
#endif

  auto srcRef = GetRefHostMatrix (src);
  auto dstRef = GetRefHostMatrix (dst);
  if (srcRef.re.ptr && dstRef.re.ptr) { oap::cuda::CopyDeviceToDevice (dstRef.re, srcRef.re); }
  if (srcRef.im.ptr && dstRef.im.ptr) { oap::cuda::CopyDeviceToDevice (dstRef.im, srcRef.im); }
}

void CopyDeviceMatrixToHostMatrix (math::Matrix* dst, const math::Matrix* src)
{
  copyDeviceMatrixToHostMatrix (dst, src);
}

void CopyHostMatrixToDeviceMatrix (math::Matrix* dst, const math::Matrix* src)
{
  copyHostMatrixToDeviceMatrix (dst, src);
}

void CopyDeviceMatrixToDeviceMatrix (math::Matrix* dst, const math::Matrix* src)
{
  copyDeviceMatrixToDeviceMatrix (dst, src);
}

void CopyDeviceToHost(math::Matrix* dst, const math::Matrix* src)
{
  uintt hcolumns = gColumns (dst);
  uintt hrows = gRows (dst);

  uintt dcolumns = oap::cuda::GetColumns (src);
  uintt drows = oap::cuda::GetRows (src);

  debugAssert(hrows * hcolumns == drows * dcolumns);

  copyDeviceMatrixToHostMatrix (dst, src);
}

void CopyHostToDevice(math::Matrix* dst, const math::Matrix* src)
{
  uintt hcolumns = gColumns (src);
  uintt hrows = gRows (src);

  uintt dcolumns = oap::cuda::GetColumns (dst);
  uintt drows = oap::cuda::GetRows (dst);

  debugAssert(hrows * hcolumns == drows * dcolumns);

  copyHostMatrixToDeviceMatrix (dst, src);
}

void CopyDeviceToDevice (math::Matrix* dst, const math::Matrix* src)
{
  uintt dcolumns1 = oap::cuda::GetColumns (src);
  uintt drows1 = oap::cuda::GetRows (src);

  uintt dcolumns2 = oap::cuda::GetColumns (dst);
  uintt drows2 = oap::cuda::GetRows (dst);

  debugAssert(drows1 * dcolumns1 == drows2 * dcolumns2);

  copyDeviceMatrixToDeviceMatrix (dst, src);
}

void CopyDeviceMatrixToHostMatrixEx (math::Matrix* dst, const oap::MemoryLoc& loc, const math::Matrix* src, const oap::MemoryRegion& reg)
{
  copyDeviceMatrixToHostMatrix (dst, loc, src, reg);
}

void CopyHostMatrixToDeviceMatrixEx (math::Matrix* dst, const oap::MemoryLoc& loc, const math::Matrix* src, const oap::MemoryRegion& reg)
{
  copyHostMatrixToDeviceMatrix (dst, loc, src, reg);
}

void CopyDeviceMatrixToDeviceMatrixEx (math::Matrix* dst, const oap::MemoryLoc& loc, const math::Matrix* src, const oap::MemoryRegion& reg)
{
  copyDeviceMatrixToDeviceMatrix (dst, loc, src, reg);
}


void SetMatrix(math::Matrix* matrix, math::Matrix* matrix1, uintt column, uintt row)
{
  SetReMatrix (matrix, matrix1, column, row);
  SetImMatrix (matrix, matrix1, column, row);
}

void SetReMatrix (math::Matrix* matrix, math::Matrix* matrix1, uintt column, uintt row)
{
  math::Matrix hm = GetRefHostMatrix (matrix);
  math::Matrix hm1 = GetRefHostMatrix (matrix1);

  debugAssert ((hm.re.ptr && !hm1.re.ptr) || (hm1.re.ptr && !hm.re.ptr));

  if (!hm.re.ptr && !hm1.re.ptr)
  {
    return;
  }

  oap::generic::copy (hm.re.ptr, hm.re.dims, hm.reReg.loc, hm1.re.ptr, hm1.re.dims, GetRefMemoryRegion(hm1.re, hm1.reReg), CudaUtils::CopyDeviceToHost);
}

void SetImMatrix (math::Matrix* matrix, math::Matrix* matrix1, uintt column, uintt row)
{
  math::Matrix hm = GetRefHostMatrix (matrix);
  math::Matrix hm1 = GetRefHostMatrix (matrix1);

  debugAssert ((hm.im.ptr && !hm1.im.ptr) || (hm1.im.ptr && !hm.im.ptr));

  if (!hm.im.ptr && !hm1.im.ptr)
  {
    return;
  }

  oap::generic::copy (hm.im.ptr, hm.im.dims, hm.imReg.loc, hm1.im.ptr, hm1.im.dims, GetRefMemoryRegion(hm1.im, hm1.imReg), CudaUtils::CopyDeviceToHost);
}

std::pair<floatt, floatt> GetDiagonal (const math::Matrix* matrix, uintt index)
{
  return std::make_pair (GetReDiagonal (matrix, index), GetImDiagonal (matrix, index));
}

floatt GetReDiagonal (const math::Matrix* matrix, uintt index)
{
  math::Matrix hm = GetRefHostMatrix (matrix);
  floatt v = 0;

  if (hm.re.ptr)
  {
    oap::MemoryLoc loc = oap::common::ConvertIdxToMemoryLoc (index, hm.re, hm.reReg);
    oap::generic::copy (&v, {1, 1}, {0, 0}, hm.re.ptr, hm.re.dims, {loc, {1, 1}}, CudaUtils::CopyDeviceToHost);
  }

  return v;
}

floatt GetImDiagonal (const math::Matrix* matrix, uintt index)
{
  math::Matrix hm = GetRefHostMatrix (matrix);
  floatt v = 0;

  if (hm.im.ptr)
  {
    oap::MemoryLoc loc = oap::common::ConvertIdxToMemoryLoc (index, hm.im, hm.imReg);
    oap::generic::copy (&v, {1, 1}, {0, 0}, hm.re.ptr, hm.re.dims, {loc, {1, 1}}, CudaUtils::CopyDeviceToHost);
  }

  return v;
}

void SetZeroRow (const math::Matrix* matrix, uintt index, bool re, bool im)
{
  if (re)
  {
    SetReZeroRow (matrix, index);
  }
  if (im)
  {
    SetImZeroRow (matrix, index);
  }
}

void SetReZeroRow (const math::Matrix* matrix, uintt index)
{
  math::Matrix hm = GetRefHostMatrix (matrix);

  if (hm.re.ptr)
  {
    uintt columns = gColumns (&hm);
    std::vector<floatt> row(columns, 0.);
    oap::MemoryLoc loc = oap::common::ConvertRegionLocToMemoryLoc (hm.re, hm.reReg, {0, index});
    oap::generic::copy (hm.re.ptr, hm.re.dims, loc, row.data(), {columns, 1}, {{0, 0}, {columns, 1}}, CudaUtils::CopyDeviceToHost);
  }
}

void SetImZeroRow (const math::Matrix* matrix, uintt index)
{
  math::Matrix hm = GetRefHostMatrix (matrix);

  if (hm.im.ptr)
  {
    uintt columns = gColumns (&hm);
    std::vector<floatt> row(columns, 0.);
    oap::MemoryLoc loc = oap::common::ConvertRegionLocToMemoryLoc (hm.im, hm.imReg, {0, index});
    oap::generic::copy (hm.im.ptr, hm.im.dims, loc, row.data(), {columns, 1}, {{0, 0}, {columns, 1}}, CudaUtils::CopyDeviceToHost);
  }
}

void SetValueToMatrix (math::Matrix* matrix, floatt re, floatt im)
{
  SetValueToReMatrix (matrix, re);
  SetValueToImMatrix (matrix, im);
}

void SetValueToReMatrix (math::Matrix* matrix, floatt v)
{
  using namespace oap::utils;

  math::Matrix hm = GetRefHostMatrix (matrix);

  if (hm.re.ptr)
  {
    auto minfo = GetMatrixInfo (matrix);
    oap::HostMatrixUPtr uptr = oap::host::NewReMatrixWithValue (minfo.columns(), minfo.rows(), v);

    oap::MemoryLoc loc = GetReMatrixMemoryLoc (&hm);
    oap::MemoryRegion reg = GetReMatrixMemoryRegion (uptr);
    oap::generic::copy (hm.re.ptr, hm.re.dims, loc, uptr->re.ptr, uptr->re.dims, reg, CudaUtils::CopyHostToDevice);
  }
}

void SetValueToImMatrix (math::Matrix* matrix, floatt v)
{
  using namespace oap::utils;

  math::Matrix hm = GetRefHostMatrix (matrix);

  if (hm.im.ptr)
  {
    auto minfo = GetMatrixInfo (matrix);
    oap::HostMatrixUPtr uptr = oap::host::NewImMatrixWithValue (minfo.columns(), minfo.rows(), v);

    oap::MemoryLoc loc = GetImMatrixMemoryLoc (&hm);
    oap::MemoryRegion reg = GetImMatrixMemoryRegion (uptr);
    oap::generic::copy (hm.im.ptr, hm.im.dims, loc, uptr->im.ptr, uptr->im.dims, reg, CudaUtils::CopyHostToDevice);
  }
}

void SetZeroMatrix (math::Matrix* matrix)
{
  SetValueToMatrix (matrix, 0, 0);
}

void SetZeroReMatrix (math::Matrix* matrix)
{
  SetValueToReMatrix (matrix, 0);
}

void SetZeroImMatrix (math::Matrix* matrix)
{
  SetValueToImMatrix (matrix, 0);
}

MatrixEx* NewDeviceMatrixEx()
{
  MatrixEx host = {0, 0, 0, 0};
  return CudaUtils::AllocDeviceObj<MatrixEx>(host);
}

void CopyHostArrayToDeviceMatrixBuffer (math::Matrix* matrix, const floatt* rebuffer, const floatt* imbuffer, size_t length)
{
  const auto& minfo = oap::cuda::GetMatrixInfo (matrix);

  if (minfo.isRe)
  {
    CopyHostArrayToDeviceReMatrixBuffer (matrix, rebuffer, length);
  }

  if (minfo.isIm)
  {
    CopyHostArrayToDeviceImMatrixBuffer (matrix, imbuffer, length);
  }
}

void CopyHostArrayToDeviceReMatrixBuffer (math::Matrix* matrix, const floatt* buffer, size_t length)
{
  const auto& minfo = oap::cuda::GetMatrixInfo (matrix);
  size_t mlength = minfo.columns() * minfo.rows();

  floatt* values = oap::cuda::GetReValuesPtr (matrix);

  debugAssert (values != nullptr);

  CudaUtils::CopyHostToDevice (values, buffer, length * sizeof(floatt));
}

void CopyHostArrayToDeviceImMatrixBuffer (math::Matrix* matrix, const floatt* buffer, size_t length)
{
  const auto& minfo = oap::cuda::GetMatrixInfo (matrix);
  size_t mlength = minfo.columns() * minfo.rows();

  floatt* values = oap::cuda::GetImValuesPtr (matrix);

  debugAssert (values != nullptr);

  CudaUtils::CopyHostToDevice (values, buffer, length * sizeof(floatt));
}

void CopyHostArrayToDeviceMatrix (math::Matrix* matrix, const floatt* rebuffer, const floatt* imbuffer, size_t length)
{
  const auto& minfo = oap::cuda::GetMatrixInfo (matrix);

  if (minfo.isRe)
  {
    CopyHostArrayToDeviceReMatrix (matrix, rebuffer, length);
  }

  if (minfo.isIm)
  {
    CopyHostArrayToDeviceImMatrix (matrix, imbuffer, length);
  }
}

void CopyHostArrayToDeviceReMatrix (math::Matrix* matrix, const floatt* buffer, size_t length)
{
  const auto& minfo = oap::cuda::GetMatrixInfo (matrix);
  size_t mlength = minfo.columns() * minfo.rows();

  debugAssert (mlength == length);

  floatt* values = oap::cuda::GetReValuesPtr (matrix);

  debugAssert (values != nullptr);

  CudaUtils::CopyHostToDevice (values, buffer, length * sizeof(floatt));
}

void CopyHostArrayToDeviceImMatrix (math::Matrix* matrix, const floatt* buffer, size_t length)
{
  const auto& minfo = oap::cuda::GetMatrixInfo (matrix);
  size_t mlength = minfo.columns() * minfo.rows();

  debugAssert (mlength == length);

  floatt* values = oap::cuda::GetImValuesPtr (matrix);

  debugAssert (values != nullptr);

  CudaUtils::CopyHostToDevice (values, buffer, length * sizeof(floatt));
}

MatrixEx* NewDeviceMatrixExCopy(const MatrixEx& hostMatrixEx)
{
  return CudaUtils::AllocDeviceObj<MatrixEx>(hostMatrixEx);
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
  uintt columns = oap::cuda::GetColumns(matrix);
  SetReValue(matrix, value, column + columns * row);
}

void SetReValue(math::Matrix* matrix, floatt value, uintt index)
{
  CudaUtils::SetReValue(matrix, index, value);
}

void SetImValue(math::Matrix* matrix, floatt value, uintt column, uintt row)
{
  uintt columns = oap::cuda::GetColumns(matrix);
  SetImValue(matrix, value, column + columns * row);
}

void SetImValue(math::Matrix* matrix, floatt value, uintt index)
{
  CudaUtils::SetImValue(matrix, index, value);
}

void SetValue(math::Matrix* matrix, floatt revalue, floatt imvalue,
              uintt column, uintt row)
{
  uintt columns = oap::cuda::GetColumns(matrix);
  SetValue(matrix, revalue, imvalue, column + columns * row);
}

void SetValue(math::Matrix* matrix, floatt revalue, floatt imvalue,
              uintt index)
{
  CudaUtils::SetReValue(matrix, index, revalue);
  CudaUtils::SetImValue(matrix, index, imvalue);
}

void ToString (std::string& str, const math::Matrix* devMatrix)
{
  if (devMatrix == nullptr)
  {
    str = "nullptr";
    return;
  }
  oap::HostMatrixUPtr ptr = oap::cuda::NewHostMatrixCopyOfDeviceMatrix (devMatrix);
  oap::host::ToString (str, ptr);
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
  math::Matrix* devMatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostMatrix);
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
    //buffer.push_back (hmatrix->reValues, minfo.length ());
  }

  if (minfo.isIm)
  {
    //buffer.push_back (hmatrix->imValues, minfo.length ());
  }
}

math::Matrix* LoadMatrix (const utils::ByteBuffer& buffer)
{
  oap::HostMatrixUPtr hmatrix = oap::host::LoadMatrix (buffer);

  if (!hmatrix)
  {
    return nullptr;
  }

  math::Matrix* matrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hmatrix);

  return matrix;
}

math::MatrixInfo LoadMatrixInfo (const utils::ByteBuffer& buffer)
{
  return oap::host::LoadMatrixInfo (buffer);
}

}
}
