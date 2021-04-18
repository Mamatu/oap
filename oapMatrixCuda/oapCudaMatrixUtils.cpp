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
#include <csignal>
#include <string.h>
#include <vector>
#include <netdb.h>
#include <execinfo.h>
#include <map>

#include "oapCudaMatrixUtils.h"
#include "oapGenericMatrixApi.h"
#include "oapCudaMemoryApi.h"

#include "oapHostComplexMatrixUPtr.h"
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
MatricesListExt<math::ComplexMatrix> g_matricesList ("MATRICES_CUDA");

void registerMatrix (math::ComplexMatrix* matrix, const math::ComplexMatrix& hostRefMatrix, const math::MatrixInfo& matrixInfo)
{
  logTrace ("ComplexMatrix: %p %s", matrix, matrixInfo.toString().c_str());

  logTrace ("ComplexMatrix allocation: %p (%p, %p)", matrix, hostRefMatrix.re.mem.ptr, hostRefMatrix.im.mem.ptr);
  g_matricesList.add (matrix, std::make_pair (matrixInfo, hostRefMatrix));
}

std::tuple<oap::Memory, oap::MemoryRegion> allocPart (bool alloc, uintt columns, uintt rows)
{
  oap::Memory memory = {nullptr, {0, 0}};
  oap::MemoryRegion region = {{0, 0}, {0, 0}};

  if (alloc)
  {
    memory = oap::cuda::NewMemory ({columns, rows});
    region = {{0, 0}, {columns, rows}};
  }

  return std::make_tuple (memory, region);
}

void initWithZero (math::ComplexMatrix* matrix, bool allocRe, bool allocIm, uintt columns, uintt rows) 
{
  oap::HostComplexMatrixUPtr hmatrix = oap::host::NewMatrixWithValue (allocRe, allocIm, columns, rows, 0);
  oap::cuda::CopyHostMatrixToDeviceMatrix (matrix, hmatrix);
}

math::ComplexMatrix* allocMatrix (const math::ComplexMatrix& hostRefMatrix)
{
  math::ComplexMatrix* matrix = reinterpret_cast<math::ComplexMatrix*>(CudaUtils::AllocDeviceMem (sizeof(math::ComplexMatrix)));

  CudaUtils::CopyHostToDevice (matrix, &hostRefMatrix, sizeof(math::ComplexMatrix));

  bool allocRe = hostRefMatrix.re.mem.ptr != nullptr;
  bool allocIm = hostRefMatrix.im.mem.ptr != nullptr;
  uintt columns = ::GetColumns(&hostRefMatrix);
  uintt rows = ::GetRows(&hostRefMatrix);

  math::MatrixInfo minfo (allocRe, allocIm, columns, rows);

  registerMatrix (matrix, hostRefMatrix, minfo);

  initWithZero (matrix, allocRe, allocIm, columns, rows);

  return matrix;
}

math::ComplexMatrix* allocMatrix (bool allocRe, bool allocIm, uintt columns, uintt rows)
{
  auto retup = allocPart (allocRe, columns, rows);
  auto imtup = allocPart (allocIm, columns, rows);

  math::ComplexMatrix hostRefMatrix;
  hostRefMatrix.dim = {columns, rows};

  hostRefMatrix.re.mem = std::get<0>(retup);
  hostRefMatrix.re.reg = std::get<1>(retup);
  hostRefMatrix.im.mem = std::get<0>(imtup);
  hostRefMatrix.im.reg = std::get<1>(imtup);

  return allocMatrix (hostRefMatrix);
}

inline math::ComplexMatrix* allocReMatrix_FromMemory (const oap::Memory& mem, const oap::MemoryRegion& reg)
{
  math::ComplexMatrix hostRefMatrix;
  hostRefMatrix.dim = {reg.dims.width, reg.dims.height};

  hostRefMatrix.re.mem = oap::cuda::ReuseMemory (mem);
  hostRefMatrix.re.reg = reg;
  hostRefMatrix.im.mem = {nullptr, {0, 0}};
  hostRefMatrix.im.reg = {{0, 0}, {0, 0}};

  return allocMatrix (hostRefMatrix);
}

inline math::ComplexMatrix* allocImMatrix_FromMemory (const oap::Memory& mem, const oap::MemoryRegion& reg)
{
  math::ComplexMatrix hostRefMatrix;
  hostRefMatrix.dim = {reg.dims.width, reg.dims.height};

  hostRefMatrix.re.mem = {nullptr, {0, 0}};
  hostRefMatrix.re.reg = {{0, 0}, {0, 0}};
  hostRefMatrix.im.mem = oap::cuda::ReuseMemory (mem);
  hostRefMatrix.im.reg = reg;

  return allocMatrix (hostRefMatrix);
}

inline math::ComplexMatrix* allocRealMatrix_FromMemory (const oap::Memory& remem, const oap::MemoryRegion& rereg, const oap::Memory& immem, const oap::MemoryRegion& imreg)
{
  math::ComplexMatrix hostRefMatrix;

  hostRefMatrix.re.mem = oap::cuda::ReuseMemory (remem);
  hostRefMatrix.re.reg = rereg;
  hostRefMatrix.im.mem = oap::cuda::ReuseMemory (immem);
  hostRefMatrix.im.reg = imreg;

  return allocMatrix (hostRefMatrix);
}

}

math::ComplexMatrix* NewDeviceMatrixFromMemory (uintt columns, uintt rows, const oap::Memory& remem, const oap::MemoryLoc& reloc, oap::Memory& immem, const oap::MemoryLoc& imloc)
{
  return allocRealMatrix_FromMemory (remem, {reloc, {columns, rows}}, immem, {imloc, {columns, rows}});
}

math::ComplexMatrix* NewDeviceReMatrixFromMemory (uintt columns, uintt rows, const oap::Memory& memory, const oap::MemoryLoc& loc)
{
  return allocReMatrix_FromMemory (memory, {loc, {columns, rows}});
}

math::ComplexMatrix* NewDeviceImMatrixFromMemory (uintt columns, uintt rows, const oap::Memory& memory, const oap::MemoryLoc& loc)
{
  return allocImMatrix_FromMemory (memory, {loc, {columns, rows}});
}

math::ComplexMatrix* NewHostMatrixCopyOfDeviceMatrix (const math::ComplexMatrix* matrix)
{
  auto minfo = GetMatrixInfo (matrix);

  math::ComplexMatrix* hostMatrix = oap::host::NewMatrix (minfo);
  oap::cuda::CopyDeviceMatrixToHostMatrix(hostMatrix, matrix);

  return hostMatrix;
}

math::ComplexMatrix* NewDeviceMatrixHostRef(const math::ComplexMatrix* hostMatrix)
{
  return NewDeviceMatrix (hostMatrix, gColumns (hostMatrix), gRows (hostMatrix));
}

math::ComplexMatrix* NewDeviceMatrixDeviceRef(const math::ComplexMatrix* deviceMatrix)
{
  auto minfo = oap::cuda::GetMatrixInfo (deviceMatrix);
  return NewDeviceMatrix (minfo.isRe, minfo.isIm, minfo.columns (), minfo.rows ());
}

math::ComplexMatrix* NewDeviceMatrix(const std::string& matrixStr)
{
  math::ComplexMatrix* host = oap::host::NewMatrix(matrixStr);
  math::ComplexMatrix* device = NewDeviceMatrixCopyOfHostMatrix (host);
  oap::host::DeleteMatrix(host);
  return device;
}

math::ComplexMatrix* NewDeviceMatrixWithValue (uintt columns, uintt rows, floatt v)
{
  math::ComplexMatrix* matrix = oap::cuda::NewDeviceMatrix (columns, rows);
  oap::HostComplexMatrixUPtr hmptr = oap::host::NewMatrixWithValue (columns, rows, v);
  oap::cuda::CopyHostMatrixToDeviceMatrix (matrix, hmptr.get());
  return matrix;
}

math::ComplexMatrix* NewDeviceReMatrixWithValue (uintt columns, uintt rows, floatt v)
{
  math::ComplexMatrix* matrix = oap::cuda::NewDeviceReMatrix (columns, rows);
  oap::HostComplexMatrixUPtr hmptr = oap::host::NewReMatrixWithValue (columns, rows, v);
  oap::cuda::CopyHostMatrixToDeviceMatrix (matrix, hmptr.get());
  return matrix;
}

math::ComplexMatrix* NewDeviceImMatrixWithValue (uintt columns, uintt rows, floatt v)
{
  math::ComplexMatrix* matrix = oap::cuda::NewDeviceImMatrix (columns, rows);
  oap::HostComplexMatrixUPtr hmptr = oap::host::NewImMatrixWithValue (columns, rows, v);
  oap::cuda::CopyHostMatrixToDeviceMatrix (matrix, hmptr.get());
  return matrix;
}

math::ComplexMatrix* NewDeviceMatrixWithValue (bool isre, bool isim, uintt columns, uintt rows, floatt v)
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

math::ComplexMatrix* NewDeviceMatrixWithValue (const math::MatrixInfo& minfo, floatt v)
{
  return oap::cuda::NewDeviceMatrixWithValue (minfo.isRe, minfo.isIm, minfo.columns(), minfo.rows(), v);
}

math::ComplexMatrix* NewDeviceMatrix(const math::MatrixInfo& minfo)
{
  return NewDeviceMatrix (minfo.isRe, minfo.isIm, minfo.m_matrixDim.columns, minfo.m_matrixDim.rows);
}

math::ComplexMatrix* NewDeviceMatrixCopyOfHostMatrix(const math::ComplexMatrix* hostMatrix)
{
  math::ComplexMatrix* dmatrix = oap::cuda::NewDeviceMatrixHostRef(hostMatrix);
  oap::cuda::CopyHostMatrixToDeviceMatrix(dmatrix, hostMatrix);
  return dmatrix;
}

math::ComplexMatrix* NewDeviceMatrix (const math::ComplexMatrix* hostMatrix, uintt columns, uintt rows)
{
  bool allocRe = hostMatrix->re.mem.ptr != NULL;
  bool allocIm = hostMatrix->im.mem.ptr != NULL;
  return allocMatrix (allocRe, allocIm, columns, rows);
}

math::ComplexMatrix* NewDeviceMatrix(bool isRe, bool isIm, uintt columns, uintt rows)
{
  debugAssert(isRe != false || isIm != false);
  return allocMatrix (isRe, isIm, columns, rows);
}

math::ComplexMatrix* NewDeviceReMatrix(uintt columns, uintt rows)
{
  return NewDeviceMatrix(true, false, columns, rows);
}

math::ComplexMatrix* NewDeviceImMatrix(uintt columns, uintt rows)
{
  return NewDeviceMatrix(false, true, columns, rows);
}

math::ComplexMatrix* NewDeviceMatrix (uintt columns, uintt rows)
{
  math::ComplexMatrix* dmatrix = allocMatrix (true, true, columns, rows);
  return dmatrix;
}

void DeleteDeviceMatrix(const math::ComplexMatrix* dMatrix)
{
  if (dMatrix != NULL)
  {
    auto pair = g_matricesList.remove (dMatrix);

    math::ComplexMatrix hm1lvl = pair.second;

    oap::cuda::DeleteMemory (hm1lvl.re.mem);
    oap::cuda::DeleteMemory (hm1lvl.im.mem);

    CudaUtils::FreeDeviceMem (dMatrix);

    logTrace ("ComplexMatrix deallocation: %p", dMatrix);

    if (pair.first.isInitialized ())
    {
      logTrace ("Deallocate: cuda matrix = %p %s", dMatrix, std::to_string (pair.first).c_str());
    }
  }
}

void DeleteDeviceComplexMatrix(const math::ComplexMatrix* deviceMatrix)
{
  DeleteDeviceMatrix (deviceMatrix);
}

void DeleteDeviceMatrix(const math::Matrix* matrix)
{
  abort();
}

uintt GetColumns(const math::ComplexMatrix* dMatrix)
{
  debugAssert (dMatrix != nullptr);
  auto minfo = g_matricesList.getUserData (dMatrix).first;
  logTrace ("ComplexMatrix: %p %s", dMatrix, std::to_string(minfo).c_str());
  return minfo.columns ();
}

uintt GetRows(const math::ComplexMatrix* dMatrix)
{
  debugAssert (dMatrix != nullptr);
  auto minfo = g_matricesList.getUserData (dMatrix).first;
  logTrace ("ComplexMatrix: %p %s", dMatrix, std::to_string(minfo).c_str());
  return minfo.rows ();
}

math::ComplexMatrix GetRefHostMatrix (const math::ComplexMatrix* dMatrix)
{
  debugAssert (dMatrix != nullptr);

  auto userData = g_matricesList.getUserData (dMatrix);

  return userData.second;
}

oap::MemoryRegion GetReMemoryRegion (const math::ComplexMatrix* dmatrix)
{
  math::ComplexMatrix hmatrix = oap::cuda::GetRefHostMatrix (dmatrix);
  return hmatrix.re.reg;
}

oap::Memory GetReMemory (const math::ComplexMatrix* dmatrix)
{
  math::ComplexMatrix hmatrix = oap::cuda::GetRefHostMatrix (dmatrix);
  return hmatrix.re.mem;
}

oap::MemoryRegion GetImMemoryRegion (const math::ComplexMatrix* dmatrix)
{
  math::ComplexMatrix hmatrix = oap::cuda::GetRefHostMatrix (dmatrix);
  return hmatrix.im.reg;
}

oap::Memory GetImMemory (const math::ComplexMatrix* dmatrix)
{
  math::ComplexMatrix hmatrix = oap::cuda::GetRefHostMatrix (dmatrix);
  return hmatrix.im.mem;
}

floatt* GetReValuesPtr (const math::ComplexMatrix* dMatrix)
{
  return GetRefHostMatrix (dMatrix).re.mem.ptr;
}

floatt* GetImValuesPtr (const math::ComplexMatrix* dMatrix)
{
  return GetRefHostMatrix (dMatrix).im.mem.ptr;
}

math::MatrixInfo GetMatrixInfo(const math::ComplexMatrix* dMatrix)
{
  debugAssert (dMatrix != nullptr);
  return g_matricesList.getUserData (dMatrix).first;
}

bool IsCudaMatrix(const math::ComplexMatrix* devMatrix)
{
	return g_matricesList.contains (devMatrix);
}

inline void copyDeviceMatrixToHostMatrix (math::ComplexMatrix* dst, const oap::MemoryLoc& loc, const math::ComplexMatrix* src, const oap::MemoryRegion& reg)
{
  auto srcRef = GetRefHostMatrix (src);
  if (dst->re.mem.ptr && srcRef.re.mem.ptr) { oap::cuda::CopyDeviceToHost (dst->re.mem, loc, srcRef.re.mem, reg); }
  if (dst->im.mem.ptr && srcRef.im.mem.ptr) { oap::cuda::CopyDeviceToHost (dst->im.mem, loc, srcRef.im.mem, reg); }
}

inline void copyDeviceMatrixToHostMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
#if 0
  oap::generic::MatrixMemoryApi<decltype (oap::host::GetMatrixInfo), decltype (oap::host::ToHost)> dstApi (oap::host::GetMatrixInfo, oap::host::ToHost);
  oap::generic::MatrixMemoryApi<decltype (oap::cuda::GetMatrixInfo), decltype (CudaUtils::ToHost)> srcApi (oap::cuda::GetMatrixInfo, CudaUtils::ToHost);
  oap::generic::copyMatrixToMatrix (dst, src, CudaUtils::CopyDeviceToHost, dstApi, srcApi, check);
#endif

  auto srcRef = GetRefHostMatrix (src);
  if (dst->re.mem.ptr && srcRef.re.mem.ptr)
  {
    debugAssert (dst->re.reg.dims == srcRef.re.reg.dims);
    oap::cuda::CopyDeviceToHost (dst->re.mem, dst->re.reg.loc, srcRef.re.mem, srcRef.re.reg);
  }
  if (dst->im.mem.ptr && srcRef.im.mem.ptr)
  {
    debugAssert (dst->im.reg.dims == srcRef.im.reg.dims);
    oap::cuda::CopyDeviceToHost (dst->im.mem, dst->im.reg.loc, srcRef.im.mem, srcRef.im.reg);
  }
}

inline void copyHostMatrixToDeviceMatrix (math::ComplexMatrix* dst, const oap::MemoryLoc& loc, const math::ComplexMatrix* src, const oap::MemoryRegion& reg)
{
  auto dstRef = GetRefHostMatrix (dst);
  if (src->re.mem.ptr && dstRef.re.mem.ptr) { oap::cuda::CopyHostToDevice (dstRef.re.mem, loc, src->re.mem, reg); }
  if (src->im.mem.ptr && dstRef.im.mem.ptr) { oap::cuda::CopyHostToDevice (dstRef.im.mem, loc, src->im.mem, reg); }
}

inline void copyHostMatrixToDeviceMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
#if 0
  oap::generic::MatrixMemoryApi<decltype (oap::cuda::GetMatrixInfo), decltype (CudaUtils::ToHost)> dstApi (oap::cuda::GetMatrixInfo, CudaUtils::ToHost);
  oap::generic::MatrixMemoryApi<decltype (oap::host::GetMatrixInfo), decltype (oap::host::ToHost)> srcApi (oap::host::GetMatrixInfo, oap::host::ToHost);
  oap::generic::copyMatrixToMatrix (dst, src, CudaUtils::CopyHostToDevice, dstApi, srcApi, check);
#endif

  auto dstRef = GetRefHostMatrix (dst);
  if (src->re.mem.ptr && dstRef.re.mem.ptr)
  {
    debugAssert (dstRef.re.reg.dims == src->re.reg.dims);
    oap::cuda::CopyHostToDevice (dstRef.re.mem, dstRef.re.reg.loc, src->re.mem, src->re.reg);
  }
  if (src->im.mem.ptr && dstRef.im.mem.ptr)
  {
    debugAssert (dstRef.im.reg.dims == src->im.reg.dims);
    oap::cuda::CopyHostToDevice (dstRef.im.mem, dstRef.im.reg.loc, src->im.mem, src->im.reg);
  }
}

inline void copyDeviceMatrixToDeviceMatrix (math::ComplexMatrix* dst, const oap::MemoryLoc& loc, const math::ComplexMatrix* src, const oap::MemoryRegion& reg)
{
  auto srcRef = GetRefHostMatrix (src);
  auto dstRef = GetRefHostMatrix (dst);
  if (srcRef.re.mem.ptr && dstRef.re.mem.ptr) { oap::cuda::CopyDeviceToDevice (dstRef.re.mem, loc, srcRef.re.mem, reg); }
  if (srcRef.im.mem.ptr && dstRef.im.mem.ptr) { oap::cuda::CopyDeviceToDevice (dstRef.im.mem, loc, srcRef.im.mem, reg); }
}

inline void copyDeviceMatrixToDeviceMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
#if 0
  oap::generic::MatrixMemoryApi<decltype (oap::cuda::GetMatrixInfo), decltype (CudaUtils::ToHost)> dstApi (oap::cuda::GetMatrixInfo, CudaUtils::ToHost);
  oap::generic::MatrixMemoryApi<decltype (oap::cuda::GetMatrixInfo), decltype (CudaUtils::ToHost)> srcApi (oap::cuda::GetMatrixInfo, CudaUtils::ToHost);
  oap::generic::copyMatrixToMatrix (dst, src, CudaUtils::CopyDeviceToDevice, dstApi, srcApi, check);
#endif

  auto srcRef = GetRefHostMatrix (src);
  auto dstRef = GetRefHostMatrix (dst);
  if (srcRef.re.mem.ptr && dstRef.re.mem.ptr) { oap::cuda::CopyDeviceToDevice (dstRef.re.mem, dstRef.re.reg.loc, srcRef.re.mem, srcRef.re.reg); }
  if (srcRef.im.mem.ptr && dstRef.im.mem.ptr) { oap::cuda::CopyDeviceToDevice (dstRef.im.mem, dstRef.im.reg.loc, srcRef.im.mem, srcRef.im.reg); }
}

void CopyDeviceMatrixToHostMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
  copyDeviceMatrixToHostMatrix (dst, src);
}

void CopyHostMatrixToDeviceMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
  copyHostMatrixToDeviceMatrix (dst, src);
}

void CopyDeviceMatrixToDeviceMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
  copyDeviceMatrixToDeviceMatrix (dst, src);
}

void CopyDeviceToHost(math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
  uintt hcolumns = gColumns (dst);
  uintt hrows = gRows (dst);

  uintt dcolumns = oap::cuda::GetColumns (src);
  uintt drows = oap::cuda::GetRows (src);

  debugAssert(hrows * hcolumns == drows * dcolumns);
  auto srcRef = GetRefHostMatrix (src);
  auto dstRef = GetRefHostMatrix (dst);

  oap::cuda::CopyDeviceToHostLinear (dstRef.re.mem, dstRef.re.reg.loc, srcRef.re.mem, srcRef.re.reg);
}

void CopyHostToDevice(math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
  uintt hcolumns = gColumns (src);
  uintt hrows = gRows (src);

  uintt dcolumns = oap::cuda::GetColumns (dst);
  uintt drows = oap::cuda::GetRows (dst);

  debugAssert(hrows * hcolumns == drows * dcolumns);
  auto srcRef = GetRefHostMatrix (src);
  auto dstRef = GetRefHostMatrix (dst);

  oap::cuda::CopyHostToDeviceLinear (dstRef.re.mem, dstRef.re.reg.loc, srcRef.re.mem, srcRef.re.reg);
}

void CopyDeviceToDevice (math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
  uintt dcolumns1 = oap::cuda::GetColumns (src);
  uintt drows1 = oap::cuda::GetRows (src);

  uintt dcolumns2 = oap::cuda::GetColumns (dst);
  uintt drows2 = oap::cuda::GetRows (dst);

  debugAssert(drows1 * dcolumns1 == drows2 * dcolumns2);
  auto srcRef = GetRefHostMatrix (src);
  auto dstRef = GetRefHostMatrix (dst);

  oap::cuda::CopyDeviceToDeviceLinear (dstRef.re.mem, dstRef.re.reg.loc, srcRef.re.mem, srcRef.re.reg);
}

void CopyDeviceMatrixToHostMatrixEx (math::ComplexMatrix* dst, const oap::MemoryLoc& loc, const math::ComplexMatrix* src, const oap::MemoryRegion& reg)
{
  copyDeviceMatrixToHostMatrix (dst, loc, src, reg);
}

void CopyHostMatrixToDeviceMatrixEx (math::ComplexMatrix* dst, const oap::MemoryLoc& loc, const math::ComplexMatrix* src, const oap::MemoryRegion& reg)
{
  copyHostMatrixToDeviceMatrix (dst, loc, src, reg);
}

void CopyDeviceMatrixToDeviceMatrixEx (math::ComplexMatrix* dst, const oap::MemoryLoc& loc, const math::ComplexMatrix* src, const oap::MemoryRegion& reg)
{
  copyDeviceMatrixToDeviceMatrix (dst, loc, src, reg);
}


void SetMatrix(math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt column, uintt row)
{
  SetReMatrix (matrix, matrix1, column, row);
  SetImMatrix (matrix, matrix1, column, row);
}

void SetReMatrix (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt column, uintt row)
{
  math::ComplexMatrix hmatrix = oap::cuda::GetRefHostMatrix (matrix);
  math::ComplexMatrix hmatrix1 = oap::cuda::GetRefHostMatrix (matrix1);

  auto getMemory = [&](math::ComplexMatrix* arg)
  {
    return arg == matrix ? hmatrix.re.mem : hmatrix1.re.mem;
  };

  auto getRegion = [&](math::ComplexMatrix* arg)
  {
    return arg == matrix ? hmatrix.re.reg : hmatrix1.re.reg;
  };

  oap::generic::setMatrix (matrix, matrix1, column, row, getMemory, getRegion, CudaUtils::CopyDeviceToDevice, CudaUtils::MoveDeviceToDevice);
}

void SetImMatrix (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt column, uintt row)
{
  math::ComplexMatrix hmatrix = oap::cuda::GetRefHostMatrix (matrix);
  math::ComplexMatrix hmatrix1 = oap::cuda::GetRefHostMatrix (matrix1);

  auto getMemory = [&](math::ComplexMatrix* arg)
  {
    return arg == matrix ? hmatrix.im.mem : hmatrix1.im.mem;
  };

  auto getRegion = [&](math::ComplexMatrix* arg)
  {
    return arg == matrix ? hmatrix.im.reg : hmatrix1.im.reg;
  };

  oap::generic::setMatrix (matrix, matrix1, column, row, getMemory, getRegion, CudaUtils::CopyDeviceToDevice, CudaUtils::MoveDeviceToDevice);
}

std::pair<floatt, floatt> GetDiagonal (const math::ComplexMatrix* matrix, uintt index)
{
  return std::make_pair (GetReDiagonal (matrix, index), GetImDiagonal (matrix, index));
}

floatt GetReDiagonal (const math::ComplexMatrix* matrix, uintt index)
{
  return oap::generic::getDiagonal (matrix, index, oap::cuda::GetRefHostMatrix,
                                    [](const math::ComplexMatrix* matrix, const math::ComplexMatrix& ref){return ref.re.mem;},
                                    [](const math::ComplexMatrix* matrix, const math::ComplexMatrix& ref){return ref.re.reg;}, CudaUtils::CopyDeviceToHost, CudaUtils::CopyDeviceToHost);
}

floatt GetImDiagonal (const math::ComplexMatrix* matrix, uintt index)
{
  return oap::generic::getDiagonal (matrix, index, oap::cuda::GetRefHostMatrix,
                                    [](const math::ComplexMatrix* matrix, const math::ComplexMatrix& ref){return ref.im.mem;},
                                    [](const math::ComplexMatrix* matrix, const math::ComplexMatrix& ref){return ref.im.reg;}, CudaUtils::CopyDeviceToHost, CudaUtils::CopyDeviceToHost);
}

void SetZeroRow (const math::ComplexMatrix* matrix, uintt index, bool re, bool im)
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

void SetReZeroRow (const math::ComplexMatrix* matrix, uintt index)
{
  math::ComplexMatrix hm = GetRefHostMatrix (matrix);

  if (hm.re.mem.ptr)
  {
    uintt columns = gColumns (&hm);
    std::vector<floatt> row(columns, 0.);
    oap::MemoryLoc loc = oap::common::ConvertRegionLocToMemoryLoc (hm.re.mem, hm.re.reg, {index, 0});
    oap::generic::copy (hm.re.mem.ptr, hm.re.mem.dims, loc, row.data(), {1, columns}, {{0, 0}, {1, columns}}, CudaUtils::CopyHostToDevice, CudaUtils::CopyHostToDevice);
  }
}

void SetImZeroRow (const math::ComplexMatrix* matrix, uintt index)
{
  math::ComplexMatrix hm = GetRefHostMatrix (matrix);

  if (hm.im.mem.ptr)
  {
    uintt columns = gColumns (&hm);
    std::vector<floatt> row(columns, 0.);
    oap::MemoryLoc loc = oap::common::ConvertRegionLocToMemoryLoc (hm.im.mem, hm.im.reg, {index, 0});
    oap::generic::copy (hm.im.mem.ptr, hm.im.mem.dims, loc, row.data(), {1, columns}, {{0, 0}, {1, columns}}, CudaUtils::CopyHostToDevice, CudaUtils::CopyHostToDevice);
  }
}

void SetValueToMatrix (math::ComplexMatrix* matrix, floatt re, floatt im)
{
  SetValueToReMatrix (matrix, re);
  SetValueToImMatrix (matrix, im);
}

void SetValueToReMatrix (math::ComplexMatrix* matrix, floatt v)
{
  using namespace oap::utils;

  math::ComplexMatrix hm = GetRefHostMatrix (matrix);

  if (hm.re.mem.ptr)
  {
    auto minfo = GetMatrixInfo (matrix);
    oap::HostComplexMatrixUPtr uptr = oap::host::NewReMatrixWithValue (minfo.columns(), minfo.rows(), v);

    oap::MemoryLoc loc = GetReMatrixMemoryLoc (&hm);
    oap::MemoryRegion reg = GetReMatrixMemoryRegion (uptr);
    oap::generic::copy (hm.re.mem.ptr, hm.re.mem.dims, loc, uptr->re.mem.ptr, uptr->re.mem.dims, reg, CudaUtils::CopyHostToDevice, CudaUtils::CopyHostToDevice);
  }
}

void SetValueToImMatrix (math::ComplexMatrix* matrix, floatt v)
{
  using namespace oap::utils;

  math::ComplexMatrix hm = GetRefHostMatrix (matrix);

  if (hm.im.mem.ptr)
  {
    auto minfo = GetMatrixInfo (matrix);
    oap::HostComplexMatrixUPtr uptr = oap::host::NewImMatrixWithValue (minfo.columns(), minfo.rows(), v);

    oap::MemoryLoc loc = GetImMatrixMemoryLoc (&hm);
    oap::MemoryRegion reg = GetImMatrixMemoryRegion (uptr);
    oap::generic::copy (hm.im.mem.ptr, hm.im.mem.dims, loc, uptr->im.mem.ptr, uptr->im.mem.dims, reg, CudaUtils::CopyHostToDevice, CudaUtils::CopyHostToDevice);
  }
}

void SetZeroMatrix (math::ComplexMatrix* matrix)
{
  SetValueToMatrix (matrix, 0.f, 0.f);
}

void SetZeroReMatrix (math::ComplexMatrix* matrix)
{
  SetValueToReMatrix (matrix, 0.f);
}

void SetZeroImMatrix (math::ComplexMatrix* matrix)
{
  SetValueToImMatrix (matrix, 0.f);
}

MatrixEx* NewDeviceMatrixEx()
{
  MatrixEx host = {0, 0, 0, 0};
  return CudaUtils::AllocDeviceObj<MatrixEx>(host);
}

void CopyHostArrayToDeviceMatrixBuffer (math::ComplexMatrix* matrix, const floatt* rebuffer, const floatt* imbuffer, size_t length)
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

void CopyHostArrayToDeviceReMatrixBuffer (math::ComplexMatrix* matrix, const floatt* buffer, size_t length)
{
  const auto& minfo = oap::cuda::GetMatrixInfo (matrix);
  size_t mlength = minfo.columns() * minfo.rows();

  floatt* values = oap::cuda::GetReValuesPtr (matrix);

  debugAssert (values != nullptr);

  CudaUtils::CopyHostToDevice (values, buffer, length * sizeof(floatt));
}

void CopyHostArrayToDeviceImMatrixBuffer (math::ComplexMatrix* matrix, const floatt* buffer, size_t length)
{
  const auto& minfo = oap::cuda::GetMatrixInfo (matrix);
  size_t mlength = minfo.columns() * minfo.rows();

  floatt* values = oap::cuda::GetImValuesPtr (matrix);

  debugAssert (values != nullptr);

  CudaUtils::CopyHostToDevice (values, buffer, length * sizeof(floatt));
}

void CopyHostArrayToDeviceMatrix (math::ComplexMatrix* matrix, const floatt* rebuffer, const floatt* imbuffer, size_t length)
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

void CopyHostArrayToDeviceReMatrix (math::ComplexMatrix* matrix, const floatt* buffer, size_t length)
{
  const auto& minfo = oap::cuda::GetMatrixInfo (matrix);
  size_t mlength = minfo.columns() * minfo.rows();

  debugAssert (mlength == length);

  floatt* values = oap::cuda::GetReValuesPtr (matrix);

  debugAssert (values != nullptr);

  CudaUtils::CopyHostToDevice (values, buffer, length * sizeof(floatt));
}

void CopyHostArrayToDeviceImMatrix (math::ComplexMatrix* matrix, const floatt* buffer, size_t length)
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

void PrintMatrix(const std::string& text, const math::ComplexMatrix* matrix, floatt zeroLimit)
{
  CudaUtils::PrintMatrix(text, matrix, zeroLimit);
}

void PrintMatrix(const math::ComplexMatrix* matrix)
{
  CudaUtils::PrintMatrix(matrix);
}

void SetReValue(math::ComplexMatrix* matrix, uintt column, uintt row, floatt value)
{
  uintt columns = oap::cuda::GetColumns(matrix);
  SetReValue(matrix, column + columns * row, value);
}

void SetReValue(math::ComplexMatrix* matrix, uintt index, floatt value)
{
  using namespace oap::utils;
  auto mhost = oap::cuda::GetRefHostMatrix (matrix);
  oap::MemoryRegion regMem = mhost.re.reg;
  oap::Memory mem = mhost.re.mem;

  oap::MemoryLoc loc = {0, 0};

  loc = oap::common::ConvertIdxToMemoryLocRef (index, mem, regMem);

  oap::generic::copy (mem.ptr, mem.dims, loc, &value, {1, 1}, {{0, 0}, {1, 1}}, CudaUtils::CopyHostToDevice, CudaUtils::CopyHostToDevice);
}

void SetImValue (math::ComplexMatrix* matrix, uintt column, uintt row, floatt value)
{
  uintt columns = oap::cuda::GetColumns(matrix);
  SetImValue(matrix, column + columns * row, value);
}

void SetImValue (math::ComplexMatrix* matrix, uintt index, floatt value)
{
  using namespace oap::utils;
  auto mhost = oap::cuda::GetRefHostMatrix (matrix);
  oap::MemoryRegion regMem = mhost.im.reg;
  oap::Memory mem = mhost.im.mem;

  oap::MemoryLoc loc = {0, 0};

  loc = oap::common::ConvertIdxToMemoryLocRef (index, mem, regMem);

  oap::generic::copy (mem.ptr, mem.dims, loc, &value, {1, 1}, {{0, 0}, {1, 1}}, CudaUtils::CopyHostToDevice, CudaUtils::CopyHostToDevice);
}

void SetValue (math::ComplexMatrix* matrix, uintt column, uintt row, floatt revalue, floatt imvalue)
{
  uintt columns = oap::cuda::GetColumns(matrix);
  SetValue(matrix, column + columns * row, revalue, imvalue);
}

void SetValue (math::ComplexMatrix* matrix, uintt index, floatt revalue, floatt imvalue)
{
  oap::cuda::SetReValue(matrix, index, revalue);
  oap::cuda::SetImValue(matrix, index, imvalue);
}

void ToString (std::string& str, const math::ComplexMatrix* devMatrix)
{
  if (devMatrix == nullptr)
  {
    str = "nullptr";
    return;
  }
  oap::HostComplexMatrixUPtr ptr = oap::cuda::NewHostMatrixCopyOfDeviceMatrix (devMatrix);
  oap::host::ToString (str, ptr);
}

void PrintMatrixInfo(const std::string& msg, const math::ComplexMatrix* devMatrix)
{
  math::MatrixInfo minfo = GetMatrixInfo (devMatrix);
  printf ("%s (columns=%u rows=%u) (isRe=%d isIm=%d)\n",
          msg.c_str(), minfo.m_matrixDim.columns, minfo.m_matrixDim.rows, minfo.isRe, minfo.isIm);
}

math::ComplexMatrix* ReadMatrix(const std::string& path)
{
  math::ComplexMatrix* hostMatrix = oap::host::ReadMatrix(path);
  math::ComplexMatrix* devMatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostMatrix);
  oap::host::DeleteMatrix(hostMatrix);
  return devMatrix;
}

bool WriteMatrix(const std::string& path, const math::ComplexMatrix* devMatrix)
{
  math::MatrixInfo matrixInfo = oap::cuda::GetMatrixInfo(devMatrix);
  math::ComplexMatrix* hostMatrix = oap::host::NewMatrix(matrixInfo);
  oap::cuda::CopyDeviceMatrixToHostMatrix(hostMatrix, devMatrix);
  bool status = oap::host::WriteMatrix(path, hostMatrix);
  oap::host::DeleteMatrix(hostMatrix);
  return status;
}

void SaveMatrixInfo (const math::MatrixInfo& minfo, utils::ByteBuffer& buffer)
{
  oap::host::SaveMatrixInfo (minfo, buffer);
}

void SaveMatrix (const math::ComplexMatrix* matrix, utils::ByteBuffer& buffer)
{
  bool isMatrix = (matrix != nullptr);

  buffer.push_back (isMatrix);

  if (!isMatrix)
  {
    return;
  }

  auto minfo = oap::cuda::GetMatrixInfo (matrix);
  SaveMatrixInfo (minfo, buffer);

  oap::HostComplexMatrixUPtr hmatrix = oap::host::NewMatrix (minfo);

  oap::cuda::CopyDeviceMatrixToHostMatrix (hmatrix, matrix);

  if (minfo.isRe)
  {
    buffer.push_back (hmatrix->re.mem.ptr, gMemoryLength (hmatrix));
  }

  if (minfo.isIm)
  {
    buffer.push_back (hmatrix->im.mem.ptr, gMemoryLength (hmatrix));
  }
}

math::ComplexMatrix* LoadMatrix (const utils::ByteBuffer& buffer)
{
  oap::HostComplexMatrixUPtr hmatrix = oap::host::LoadMatrix (buffer);

  if (!hmatrix)
  {
    return nullptr;
  }

  math::ComplexMatrix* matrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hmatrix);

  return matrix;
}

math::MatrixInfo LoadMatrixInfo (const utils::ByteBuffer& buffer)
{
  return oap::host::LoadMatrixInfo (buffer);
}

std::map<std::vector<std::vector<math::ComplexMatrix*>>, oap::ThreadsMapper> g_threadsMapper;
/*
bool operator<(const std::vector<math::ComplexMatrix*>& arg1, const std::vector<math::ComplexMatrix*>& arg2)
{
  if (arg1.size() != arg2.size())
  {
    return arg1.size() < arg2.size();
  }

  uintt length = arg1.size();

  for (uintt idx = 0; idx < length; ++idx)
  {
    if (arg1[idx] != arg2[idx])
    {
      return arg1[idx] < arg2[idx];
    }
  }
  return false;
}

bool operator<(const std::vector<std::vector<math::ComplexMatrix*>>& arg1, const std::vector<std::vector<math::ComplexMatrix*>>& arg2)
{
  if (arg1.size() != arg2.size())
  {
    return arg1.size() < arg2.size();
  }
  uintt length = arg1.size();
  for (uintt idx = 0; idx < length; ++idx)
  {
    return arg1[idx] < arg2[idx];
  }
  return false;
}*/

oap::ThreadsMapper CreateThreadsMapper (const std::vector<std::vector<math::ComplexMatrix*>>& matrices, oap::threads::ThreadsMapperAlgo algo)
{
  auto it = g_threadsMapper.find(matrices);
  if (it != g_threadsMapper.end())
  {
    return it->second;
  }
  oap::ThreadsMapper mapper = createThreadsMapper (matrices, algo);
  g_threadsMapper.insert(std::make_pair(matrices, mapper));
  return mapper;
}

void CopyDeviceReMatrixToHostBuffer (floatt* buffer, uintt length, const math::ComplexMatrix* matrix)
{
  math::ComplexMatrix ref = oap::cuda::GetRefHostMatrix (matrix);
  oap::cuda::CopyDeviceToHostBuffer (buffer, length, ref.re.mem, ref.re.reg);
}

void CopyHostBufferToDeviceReMatrix (math::ComplexMatrix* matrix, const floatt* buffer, uintt length)
{
  math::ComplexMatrix ref = oap::cuda::GetRefHostMatrix (matrix);
  oap::cuda::CopyHostBufferToDevice (ref.re.mem, ref.re.reg, buffer, length);
}

void CopyDeviceBufferToDeviceReMatrix (math::ComplexMatrix* matrix, const floatt* buffer, uintt length)
{
  math::ComplexMatrix ref = oap::cuda::GetRefHostMatrix (matrix);
  oap::cuda::CopyDeviceBufferToDevice (ref.re.mem, ref.re.reg, buffer, length);
}

std::string to_carraystr(const math::ComplexMatrix* matrix)
{
  oap::HostComplexMatrixUPtr hmatrix = oap::host::NewHostMatrixFromMatrixInfo (oap::cuda::GetMatrixInfo (matrix));
  oap::cuda::CopyDeviceMatrixToHostMatrix (hmatrix, matrix);
  return oap::host::to_carraystr (hmatrix);
}

std::string to_carraystr(const std::vector<math::ComplexMatrix*>& matrices)
{
  std::vector<math::ComplexMatrix*> hptrs;
  for (const math::ComplexMatrix* matrix : matrices)
  {
    auto minfo = oap::cuda::GetMatrixInfo (matrix);
    logTrace("%s", std::to_string (minfo).c_str());
    math::ComplexMatrix* hmatrix = oap::host::NewHostMatrixFromMatrixInfo (minfo);
    oap::cuda::CopyDeviceMatrixToHostMatrix (hmatrix, matrix);
    hptrs.push_back (hmatrix);
  }
  std::string str = oap::host::to_carraystr (hptrs);
  oap::host::deleteMatrices (hptrs);
  return str;
}

math::ComplexMatrix* NewDeviceReMatrixCopyOfArray(uintt columns, uintt rows, floatt* array)
{
  oap::HostComplexMatrixUPtr hmatrix = oap::host::NewReMatrixCopyOfArray (columns, rows, array);
  math::ComplexMatrix* dmatrix = oap::cuda::NewDeviceReMatrix (columns, rows);
  oap::cuda::CopyHostMatrixToDeviceMatrix (dmatrix, hmatrix);
  return dmatrix;
}

std::vector<math::ComplexMatrix*> NewDeviceMatrices (const std::vector<math::MatrixInfo>& minfos)
{
  std::vector<math::ComplexMatrix*> matrices;
  for (const auto& minfo : minfos)
  {
    math::ComplexMatrix* matrix = oap::cuda::NewDeviceMatrixFromMatrixInfo (minfo);
    matrices.push_back (matrix);
  }
  return matrices;
}

std::vector<math::ComplexMatrix*> NewDeviceMatrices (const math::MatrixInfo& minfo, uintt count)
{
  std::vector<math::MatrixInfo> minfos (count, minfo);
  return NewDeviceMatrices (minfos);
}

std::vector<math::ComplexMatrix*> NewDeviceMatricesCopyOfArray(const std::vector<math::MatrixInfo>& minfos, const std::vector<std::vector<floatt>>& arrays)
{
  std::vector<math::ComplexMatrix*> hmatrices = oap::host::NewMatricesCopyOfArray (minfos, arrays);
  std::vector<math::ComplexMatrix*> dmatrices;
  for (math::ComplexMatrix* hmatrix : hmatrices)
  {
    math::ComplexMatrix* dmatrix = NewDeviceMatrixCopyOfHostMatrix (hmatrix);
    dmatrices.push_back (dmatrix);
  }
  oap::host::deleteMatrices (hmatrices);
  return dmatrices;
}

std::vector<math::ComplexMatrix*> NewDeviceMatricesCopyOfArray(const math::MatrixInfo& minfo, const std::vector<std::vector<floatt>>& arrays)
{
  std::vector<math::MatrixInfo> minfos (arrays.size(), minfo);
  return NewDeviceMatricesCopyOfArray (minfos, arrays);
}

math::ComplexMatrix* NewDeviceSharedSubMatrix (const math::MatrixLoc& loc, const math::MatrixDim& dim, const math::ComplexMatrix* matrix)
{
  auto minfo = oap::cuda::GetMatrixInfo (matrix);

  oapAssert (loc.x < minfo.columns());
  oapAssert (loc.y < minfo.rows());
  oapAssert (loc.x + dim.columns <= minfo.columns());
  oapAssert (loc.y + dim.rows <= minfo.rows());

  math::ComplexMatrix refmatrix = oap::cuda::GetRefHostMatrix (matrix);
  math::ComplexMatrix* output = nullptr;

  if (minfo.isRe && minfo.isIm)
  {
    oap::MemoryLoc reloc = oap::common::ConvertRegionLocToMemoryLoc (refmatrix.re.mem, refmatrix.re.reg, {loc.x, loc.y});
    oap::MemoryLoc imloc = oap::common::ConvertRegionLocToMemoryLoc (refmatrix.im.mem, refmatrix.im.reg, {loc.x, loc.y});

    output = oap::cuda::NewDeviceMatrixFromMemory (dim.columns, dim.rows, refmatrix.re.mem, reloc, refmatrix.im.mem, imloc);
  }
  else if (minfo.isRe)
  {
    oap::MemoryLoc reloc = oap::common::ConvertRegionLocToMemoryLoc (refmatrix.re.mem, refmatrix.re.reg, {loc.x, loc.y});
    output = oap::cuda::NewDeviceReMatrixFromMemory (dim.columns, dim.rows, refmatrix.re.mem, reloc);
  }
  else if (minfo.isIm)
  {
    oap::MemoryLoc imloc = oap::common::ConvertRegionLocToMemoryLoc (refmatrix.im.mem, refmatrix.im.reg, {loc.x, loc.y});
    output = oap::cuda::NewDeviceImMatrixFromMemory (dim.columns, dim.rows, refmatrix.im.mem, imloc);
  }

  return output;
}

math::ComplexMatrix* NewDeviceSharedSubMatrix (const math::MatrixDim& dim, const math::ComplexMatrix* matrix)
{
  return NewDeviceSharedSubMatrix ({0, 0}, dim, matrix);
}

}
}
