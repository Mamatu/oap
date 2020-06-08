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
MatricesListExt<math::Matrix> g_matricesList ("MATRICES_CUDA");
std::map<floatt*, uintt> g_usedMemoryCounter;

void registerMatrix (math::Matrix* matrix, const math::Matrix& hostRefMatrix, const math::MatrixInfo& matrixInfo)
{
  logTrace ("Matrix: %p %s", matrix, matrixInfo.toString().c_str());

  logTrace ("Matrix allocation: %p", matrix);
  g_matricesList.add (matrix, std::make_pair (matrixInfo, hostRefMatrix));
}

void registerMemory (const oap::Memory& memory)
{
  if (memory.ptr != nullptr)
  {
    auto it = g_usedMemoryCounter.find (memory.ptr);
    if (it == g_usedMemoryCounter.end())
    {
      g_usedMemoryCounter[memory.ptr] = 0;
    }
    g_usedMemoryCounter[memory.ptr]++;
  }
}

template<typename Callback>
void unregisterMemory (const oap::Memory& memory, Callback&& callback)
{
  if (memory.ptr != nullptr)
  {
    auto it = g_usedMemoryCounter.find(memory.ptr);
    it->second--;
    if (it->second == 0)
    {
      callback (memory);
      g_usedMemoryCounter.erase (it);
    }
  }
}

oap::Memory allocPart (bool alloc, uintt columns, uintt rows)
{
  oap::Memory memory = {nullptr, {0, 0}};
  oap::MemoryRegion region = {{0, 0}, {columns, rows}};

  if (alloc)
  {
    memory = oap::cuda::NewMemory ({columns, rows});
    region = {{0, 0}, {columns, rows}};
  }

  return memory;
}

void initWithZero (math::Matrix* matrix, bool allocRe, bool allocIm, uintt columns, uintt rows) 
{
  oap::HostMatrixUPtr hmatrix = oap::host::NewMatrixWithValue (allocRe, allocIm, columns, rows, 0);
  oap::cuda::CopyHostMatrixToDeviceMatrix (matrix, hmatrix);
}

math::Matrix* allocMatrix (const math::Matrix& hostRefMatrix)
{
  math::Matrix* matrix = reinterpret_cast<math::Matrix*>(CudaUtils::AllocDeviceMem (sizeof(math::Matrix)));

  CudaUtils::CopyHostToDevice (matrix, &hostRefMatrix, sizeof(math::Matrix));

  bool allocRe = hostRefMatrix.re.ptr != nullptr;
  bool allocIm = hostRefMatrix.im.ptr != nullptr;
  uintt columns = ::GetColumns(&hostRefMatrix);
  uintt rows = ::GetRows(&hostRefMatrix);

  math::MatrixInfo minfo (allocRe, allocIm, columns, rows);

  registerMatrix (matrix, hostRefMatrix, minfo);
  registerMemory (hostRefMatrix.re);
  registerMemory (hostRefMatrix.im);

  initWithZero (matrix, allocRe, allocIm, columns, rows);

  return matrix;
}

math::Matrix* allocMatrix (bool allocRe, bool allocIm, uintt columns, uintt rows)
{
  auto initReg = [](oap::MemoryRegion& reg, uintt columns, uintt rows)
  {
    if (reg.dims.width == 0 && reg.dims.height == 0)
    {
      reg = {{0, 0}, {columns, rows}};
    }
  };

  oap::Memory reMem = allocPart (allocRe, columns, rows);
  oap::Memory imMem = allocPart (allocIm, columns, rows);

  math::Matrix hostRefMatrix;
  hostRefMatrix.re = reMem;
  hostRefMatrix.reReg = {{0, 0}, {columns, rows}};
  hostRefMatrix.im = imMem;
  hostRefMatrix.imReg = {{0, 0}, {columns, rows}};

  return allocMatrix (hostRefMatrix);
}

inline math::Matrix* allocReMatrix_FromMemory (oap::Memory& mem, const oap::MemoryRegion& reg)
{
  math::Matrix hostRefMatrix;

  hostRefMatrix.re = mem;
  hostRefMatrix.reReg = reg;
  hostRefMatrix.im = {nullptr, {0, 0}};
  hostRefMatrix.imReg = {{0, 0}, {0, 0}};

  return allocMatrix (hostRefMatrix);
}

inline math::Matrix* allocImMatrix_FromMemory (oap::Memory& mem, const oap::MemoryRegion& reg)
{
  math::Matrix hostRefMatrix;

  hostRefMatrix.re = {nullptr, {0, 0}};
  hostRefMatrix.reReg = {{0, 0}, {0, 0}};
  hostRefMatrix.im = mem;
  hostRefMatrix.imReg = reg;

  return allocMatrix (hostRefMatrix);
}

inline math::Matrix* allocRealMatrix_FromMemory (oap::Memory& remem, const oap::MemoryRegion& rereg, oap::Memory& immem, const oap::MemoryRegion& imreg)
{
  math::Matrix hostRefMatrix;

  hostRefMatrix.re = remem;
  hostRefMatrix.reReg = rereg;
  hostRefMatrix.im = immem;
  hostRefMatrix.imReg = imreg;

  return allocMatrix (hostRefMatrix);
}

}

math::Matrix* NewDeviceMatrixFromMemory (uintt columns, uintt rows, oap::Memory& remem, const oap::MemoryLoc& reloc, oap::Memory& immem, const oap::MemoryLoc& imloc)
{
  return allocRealMatrix_FromMemory (remem, {reloc, {columns, rows}}, immem, {imloc, {columns, rows}});
}

math::Matrix* NewDeviceReMatrixFromMemory (uintt columns, uintt rows, oap::Memory& memory, const oap::MemoryLoc& loc)
{
  return allocReMatrix_FromMemory (memory, {loc, {columns, rows}});
}

math::Matrix* NewDeviceImMatrixFromMemory (uintt columns, uintt rows, oap::Memory& memory, const oap::MemoryLoc& loc)
{
  return allocImMatrix_FromMemory (memory, {loc, {columns, rows}});
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
  return NewDeviceMatrix (hostMatrix, gColumns (hostMatrix), gRows (hostMatrix));
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

math::Matrix* NewDeviceMatrixWithValue (uintt columns, uintt rows, floatt v)
{
  math::Matrix* matrix = oap::cuda::NewDeviceMatrix (columns, rows);
  oap::HostMatrixUPtr hmptr = oap::host::NewMatrixWithValue (columns, rows, v);
  oap::cuda::CopyHostMatrixToDeviceMatrix (matrix, hmptr.get());
  return matrix;
}

math::Matrix* NewDeviceReMatrixWithValue (uintt columns, uintt rows, floatt v)
{
  math::Matrix* matrix = oap::cuda::NewDeviceReMatrix (columns, rows);
  oap::HostMatrixUPtr hmptr = oap::host::NewReMatrixWithValue (columns, rows, v);
  oap::cuda::CopyHostMatrixToDeviceMatrix (matrix, hmptr.get());
  return matrix;
}

math::Matrix* NewDeviceImMatrixWithValue (uintt columns, uintt rows, floatt v)
{
  math::Matrix* matrix = oap::cuda::NewDeviceImMatrix (columns, rows);
  oap::HostMatrixUPtr hmptr = oap::host::NewImMatrixWithValue (columns, rows, v);
  oap::cuda::CopyHostMatrixToDeviceMatrix (matrix, hmptr.get());
  return matrix;
}

math::Matrix* NewDeviceMatrixWithValue (bool isre, bool isim, uintt columns, uintt rows, floatt v)
{
  if (isre && isim)
  {
    return oap::cuda::NewDeviceMatrixWithValue (columns, rows, v);
  }
  else if (isre)
  {
    return oap::cuda::NewDeviceReMatrixWithValue (columns, rows, v);
  }
  else if (isim)
  {
    return oap::cuda::NewDeviceImMatrixWithValue (columns, rows, v);
  }
  return nullptr;
}

math::Matrix* NewDeviceMatrixWithValue (const math::MatrixInfo& minfo, floatt v)
{
  return oap::cuda::NewDeviceMatrixWithValue (minfo.isRe, minfo.isIm, minfo.columns(), minfo.rows(), v);
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
  return allocMatrix (allocRe, allocIm, columns, rows);
}

math::Matrix* NewDeviceMatrix(bool isRe, bool isIm, uintt columns, uintt rows)
{
  debugAssert(isRe != false || isIm != false);
  return allocMatrix (isRe, isIm, columns, rows);
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
  math::Matrix* dmatrix = allocMatrix (true, true, columns, rows);
  return dmatrix;
}

void DeleteDeviceMatrix(const math::Matrix* dMatrix)
{
  if (dMatrix != NULL)
  {
    auto pair = g_matricesList.remove (dMatrix);

    math::Matrix hm1lvl = pair.second;

    unregisterMemory(hm1lvl.re, oap::cuda::DeleteMemory);
    unregisterMemory(hm1lvl.im, oap::cuda::DeleteMemory);

    CudaUtils::FreeDeviceMem (dMatrix);

    logTrace ("Matrix deallocation: %p", dMatrix);

    if (pair.first.isInitialized ())
    {
      logTrace ("Deallocate: cuda matrix = %p %s", dMatrix, std::to_string (pair.first).c_str());
    }
  }
}

uintt GetColumns(const math::Matrix* dMatrix)
{
  debugAssert (dMatrix != nullptr);
  auto minfo = g_matricesList.getUserData (dMatrix).first;
  logTrace ("Matrix: %p %s", dMatrix, std::to_string(minfo).c_str());
  return minfo.columns ();
}

uintt GetRows(const math::Matrix* dMatrix)
{
  debugAssert (dMatrix != nullptr);
  auto minfo = g_matricesList.getUserData (dMatrix).first;
  logTrace ("Matrix: %p %s", dMatrix, std::to_string(minfo).c_str());
  return minfo.rows ();
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
  if (dst->re.ptr && srcRef.re.ptr)
  {
    debugAssert (dst->reReg.dims == srcRef.reReg.dims);
    oap::cuda::CopyDeviceToHost (dst->re, dst->reReg.loc, srcRef.re, srcRef.reReg);
  }
  if (dst->im.ptr && srcRef.im.ptr)
  {
    debugAssert (dst->imReg.dims == srcRef.imReg.dims);
    oap::cuda::CopyDeviceToHost (dst->im, dst->imReg.loc, srcRef.im, srcRef.imReg);
  }
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
  if (src->re.ptr && dstRef.re.ptr)
  {
    debugAssert (dstRef.reReg.dims == src->reReg.dims);
    oap::cuda::CopyHostToDevice (dstRef.re, dstRef.reReg.loc, src->re, src->reReg);
  }
  if (src->im.ptr && dstRef.im.ptr)
  {
    debugAssert (dstRef.imReg.dims == src->imReg.dims);
    oap::cuda::CopyHostToDevice (dstRef.im, dstRef.imReg.loc, src->im, src->imReg);
  }
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
  auto srcRef = GetRefHostMatrix (src);
  auto dstRef = GetRefHostMatrix (dst);

  oap::cuda::CopyDeviceToHostLinear (dstRef.re, dstRef.reReg.loc, srcRef.re, srcRef.reReg);
}

void CopyHostToDevice(math::Matrix* dst, const math::Matrix* src)
{
  uintt hcolumns = gColumns (src);
  uintt hrows = gRows (src);

  uintt dcolumns = oap::cuda::GetColumns (dst);
  uintt drows = oap::cuda::GetRows (dst);

  debugAssert(hrows * hcolumns == drows * dcolumns);
  auto srcRef = GetRefHostMatrix (src);
  auto dstRef = GetRefHostMatrix (dst);

  oap::cuda::CopyHostToDeviceLinear (dstRef.re, dstRef.reReg.loc, srcRef.re, srcRef.reReg);
}

void CopyDeviceToDevice (math::Matrix* dst, const math::Matrix* src)
{
  uintt dcolumns1 = oap::cuda::GetColumns (src);
  uintt drows1 = oap::cuda::GetRows (src);

  uintt dcolumns2 = oap::cuda::GetColumns (dst);
  uintt drows2 = oap::cuda::GetRows (dst);

  debugAssert(drows1 * dcolumns1 == drows2 * dcolumns2);
  auto srcRef = GetRefHostMatrix (src);
  auto dstRef = GetRefHostMatrix (dst);

  oap::cuda::CopyDeviceToDeviceLinear (dstRef.re, dstRef.reReg.loc, srcRef.re, srcRef.reReg);
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
  math::Matrix hmatrix = oap::cuda::GetRefHostMatrix (matrix);
  math::Matrix hmatrix1 = oap::cuda::GetRefHostMatrix (matrix1);

  auto getMemory = [&](math::Matrix* arg)
  {
    return arg == matrix ? hmatrix.re : hmatrix1.re;
  };

  auto getRegion = [&](math::Matrix* arg)
  {
    return arg == matrix ? hmatrix.reReg : hmatrix1.reReg;
  };

  oap::generic::setMatrix (matrix, matrix1, column, row, getMemory, getRegion, CudaUtils::CopyDeviceToDevice);
}

void SetImMatrix (math::Matrix* matrix, math::Matrix* matrix1, uintt column, uintt row)
{
  math::Matrix hmatrix = oap::cuda::GetRefHostMatrix (matrix);
  math::Matrix hmatrix1 = oap::cuda::GetRefHostMatrix (matrix1);

  auto getMemory = [&](math::Matrix* arg)
  {
    return arg == matrix ? hmatrix.im : hmatrix1.im;
  };

  auto getRegion = [&](math::Matrix* arg)
  {
    return arg == matrix ? hmatrix.imReg : hmatrix1.imReg;
  };

  oap::generic::setMatrix (matrix, matrix1, column, row, getMemory, getRegion, CudaUtils::CopyDeviceToDevice);
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

void SetReValue(math::Matrix* matrix, uintt column, uintt row, floatt value)
{
  uintt columns = oap::cuda::GetColumns(matrix);
  SetReValue(matrix, column + columns * row, value);
}

void SetReValue(math::Matrix* matrix, uintt index, floatt value)
{
  using namespace oap::utils;
  auto mhost = oap::cuda::GetRefHostMatrix (matrix);
  oap::MemoryRegion regMem = mhost.reReg;
  oap::Memory mem = mhost.re;

  oap::MemoryLoc loc = {0, 0};

  loc = oap::common::ConvertIdxToMemoryLocRef (index, mem, regMem);

  oap::generic::copy (mem.ptr, mem.dims, loc, &value, {1, 1}, {{0, 0}, {1, 1}}, CudaUtils::CopyHostToDevice);
}

void SetImValue (math::Matrix* matrix, uintt column, uintt row, floatt value)
{
  uintt columns = oap::cuda::GetColumns(matrix);
  SetImValue(matrix, column + columns * row, value);
}

void SetImValue (math::Matrix* matrix, uintt index, floatt value)
{
  using namespace oap::utils;
  auto mhost = oap::cuda::GetRefHostMatrix (matrix);
  oap::MemoryRegion regMem = mhost.imReg;
  oap::Memory mem = mhost.im;

  oap::MemoryLoc loc = {0, 0};

  loc = oap::common::ConvertIdxToMemoryLocRef (index, mem, regMem);

  oap::generic::copy (mem.ptr, mem.dims, loc, &value, {1, 1}, {{0, 0}, {1, 1}}, CudaUtils::CopyHostToDevice);
}

void SetValue (math::Matrix* matrix, uintt column, uintt row, floatt revalue, floatt imvalue)
{
  uintt columns = oap::cuda::GetColumns(matrix);
  SetValue(matrix, column + columns * row, revalue, imvalue);
}

void SetValue (math::Matrix* matrix, uintt index, floatt revalue, floatt imvalue)
{
  oap::cuda::SetReValue(matrix, index, revalue);
  oap::cuda::SetImValue(matrix, index, imvalue);
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
    buffer.push_back (hmatrix->re.ptr, gMemoryLength (hmatrix));
  }

  if (minfo.isIm)
  {
    buffer.push_back (hmatrix->im.ptr, gMemoryLength (hmatrix));
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

oap::threads::ThreadsMapper CreateThreadsMapper (const std::vector<math::Matrix*>& matrices)
{
  return createThreadsMapper (matrices);
}

}
}
