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

#ifndef OAP_GENERIC_MATRIX_API_H
#define OAP_GENERIC_MATRIX_API_H

#include "ByteBuffer.h"
#include "Logger.h"

#include "Matrix.h"
#include "MatrixInfo.h"
#include "MatrixEx.h"
#include "MatrixUtils.h"
#include "MatrixPrinter.h"

#include "oapMemory_GenericApi.h"
#include "oapThreadsMapperApi.h"

#include "oapHostMemoryApi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>

#define PRINT_MATRIX(m) logInfo ("%s %p\n%s %s", #m, m, oap::host::to_string(m).c_str(), oap::host::GetMatrixInfo(m).toString().c_str());
#define PRINT_DIMS_3_2(m) logInfo ("%s dims = {{%u, %u}, {%u, %u}, {%u, %u}} ", #m, m[0][0], m[0][1], m[1][0], m[1][1], m[2][0], m[2][1]);
#define PRINT_DIMS_2_2_2(m) logInfo ("%s dims = {{{%u, %u}, {%u, %u}}, {{%u, %u}, {%u, %u}}} ", #m, m[0][0][0], m[0][0][1], m[0][1][0], m[0][1][1], m[1][0][0], m[1][0][1], m[1][1][0], m[1][1][1]);

namespace oap
{
namespace generic
{

inline void initDims (uintt dims[2][2][2], const math::MatrixInfo& dist, const math::MatrixInfo& src)
{
  dims[0][0][0] = 0;
  dims[0][0][1] = dist.columns();

  dims[0][1][0] = 0;
  dims[0][1][1] = dist.rows();

  dims[1][0][0] = 0;
  dims[1][0][1] = src.columns();

  dims[1][1][0] = 0;
  dims[1][1][1] = src.rows();
}

inline void initDims (uintt dims[2][2][2], const math::MatrixInfo& minfo)
{
  initDims (dims, minfo, minfo);
}

template<typename GetMatrixInfo>
void initDims (uintt dims[2][2][2], const math::ComplexMatrix* matrix, GetMatrixInfo&& getMatrixInfo)
{
  initDims (dims, getMatrixInfo (matrix));
}

const uintt g_srcIdx = 1;
const uintt g_dstIdx = 0;

inline void setColumnIdx (uintt v, uintt dims[2][2])
{
  dims[0][0] = v;
}

inline void setRowIdx (uintt v, uintt dims[2][2])
{
  dims[1][0] = v;
}

inline void setColumns (uintt v, uintt dims[2][2])
{
  dims[0][1] = v;
}

inline void setRows (uintt v, uintt dims[2][2])
{
  dims[1][1] = v;
}

inline uintt getColumnIdx (uintt dims[2][2])
{
  return dims[0][0];
}

inline uintt getRowIdx (uintt dims[2][2])
{
  return dims[1][0];
}

inline uintt getColumns (uintt dims[2][2])
{
  return dims[0][1];
}

inline uintt getRows (uintt dims[2][2])
{
  return dims[1][1];
}

inline math::MatrixInfo check_CopyMatrixToMatrix (const math::MatrixInfo& dstInfo, const math::MatrixInfo& srcInfo)
{
  debugAssert (dstInfo.columns() == srcInfo.columns());
  debugAssert (dstInfo.rows() == srcInfo.rows());
  debugAssert (dstInfo.isRe == srcInfo.isRe);
  debugAssert (dstInfo.isIm == srcInfo.isIm);

  return dstInfo;
}

inline math::MatrixInfo check_CopyMatrixToMatrixDims (const math::MatrixInfo& dstInfo, const math::MatrixInfo& srcInfo, uintt dims[2][2][2])
{
  debugAssert (getRows (dims[g_srcIdx]) == getRows (dims[g_dstIdx]));
  debugAssert (getColumns (dims[g_srcIdx]) == getColumns (dims[g_dstIdx]));
  debugAssert (dstInfo.isRe == srcInfo.isRe);
  debugAssert (dstInfo.isIm == srcInfo.isIm);

  debugAssert (getColumnIdx (dims[g_srcIdx]) + getColumns (dims[g_srcIdx]) <= srcInfo.columns());
  debugAssert (getRowIdx (dims[g_srcIdx]) + getRows (dims[g_srcIdx]) <= srcInfo.rows());

  debugAssert (getColumnIdx (dims[g_dstIdx]) + getColumns (dims[g_dstIdx]) <= dstInfo.columns());
  debugAssert (getRowIdx (dims[g_dstIdx]) + getRows (dims[g_dstIdx]) <= dstInfo.rows());

  return dstInfo;
}

/* \brief Copy matrix to matrix
 * \params Memcpy - function of type (void* dst, const void* src, size_t size)
 * \param MatrixMemoryApi - see GenericCoreApi -> class MatrixMemoryApi
 *
 */
template<typename Memcpy, typename MatrixMemoryApi>
void copyMatrixToMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src, Memcpy&& memcpy, Memcpy&& memmove, MatrixMemoryApi& dstApi, MatrixMemoryApi& srcApi, bool check = true)
{
  math::MatrixInfo minfo;
  math::MatrixInfo dstInfo = dstApi.getMatrixInfo (dst);

  if (check)
  {
    math::MatrixInfo srcInfo = srcApi.getMatrixInfo (src);
    minfo = check_CopyMatrixToMatrix (dstInfo, srcInfo);
  }
  else
  {
    minfo = dstInfo;
  }

  uintt length = minfo.rows() * minfo.columns();

  auto getRawPointer = [](oap::Memory* memory, MatrixMemoryApi& api)
  {
    floatt* ptr = nullptr;
    api.transferValueToHost (&ptr, &(memory->ptr), sizeof (floatt*));
    return ptr;
  };

  auto getMemoryRegion = [](oap::MemoryRegion* memRegPtr, MatrixMemoryApi& api)
  {
    oap::MemoryRegion reg;
    api.transferValueToHost (&reg, memRegPtr, sizeof (oap::MemoryRegion));
    return reg;
  };

  auto getDims = [](oap::Memory* memory, MatrixMemoryApi& api)
  {
    oap::MemoryDim memDims;
    api.transferValueToHost (&memDims, &(memory->dims), sizeof (oap::MemoryDim));
    return memDims;
  };

  struct MatrixData
  {
    floatt* ptr;
    oap::MemoryDim dims;
    oap::MemoryRegion region;
  };

  auto getMatrixData = [&getRawPointer, &getMemoryRegion, &getDims] (const math::ComplexMatrix* matrix, oap::Memory* const* memPPtr, oap::MemoryRegion* const* regPPtr, MatrixMemoryApi& api)
  {
    oap::Memory* memory = nullptr;
    api.transferValueToHost (&memory, memPPtr, sizeof (oap::Memory*));

    floatt* ptr = getRawPointer (memory, api);
    oap::MemoryDim dims = getDims (memory, api);

    oap::MemoryRegion* regPtr;
    api.transferValueToHost (&regPtr, regPPtr, sizeof (oap::MemoryRegion*));
    oap::MemoryRegion reg = getMemoryRegion (regPtr, api);

    MatrixData data = {ptr, dims, reg};
    return data;
  };

  auto getReMatrixData = [&](const math::ComplexMatrix* matrix, MatrixMemoryApi& api)
  {
    return getMatrixData (matrix, &(matrix->re.mem), &(matrix->re.reg), api);
  };

  auto getImMatrixData = [&](const math::ComplexMatrix* matrix, MatrixMemoryApi& api)
  {
    return getMatrixData (matrix, &(matrix->im.mem), &(matrix->im.reg), api);
  };

  if (minfo.isRe)
  {
    MatrixData dstRe = getReMatrixData (dst, dstApi);
    MatrixData srcRe = getReMatrixData (src, srcApi);
    oap::generic::copy (dstRe.ptr, dstRe.dims, dstRe.region.loc, srcRe.ptr, srcRe.dims, srcRe.region, memcpy, memmove);
  }

  if (minfo.isIm)
  {
    MatrixData dstIm = getImMatrixData (dst, dstApi);
    MatrixData srcIm = getImMatrixData (src, srcApi);
    oap::generic::copy (dstIm.ptr, dstIm.dims, dstIm.region.loc, srcIm.ptr, srcIm.dims, srcIm.region, memcpy, memmove);
  }
}

namespace
{
inline void printCustomMatrix (std::string& output, const math::ComplexMatrix* matrix, const matrixUtils::PrintArgs& args, const math::MatrixInfo& minfo)
{
  uintt columns = minfo.columns ();
  uintt rows = minfo.rows ();

  bool isre = minfo.isRe;
  bool isim = minfo.isIm;

  matrixUtils::PrintArgs printArgs (args);

  if ((columns > 10 && rows == 1) || (columns == 1 && rows > 10) || (columns * rows > 10 && args.section.separator.find ("\n") == std::string::npos))
  {
    printArgs.printIndex = true;
  }

  matrixUtils::PrintMatrix (output, matrix, printArgs);
}
}

template<typename GetMatrixInfo>
void printMatrix (std::string& output, const math::ComplexMatrix* matrix, const matrixUtils::PrintArgs& args, GetMatrixInfo&& getMatrixInfo)
{
  math::MatrixInfo minfo = getMatrixInfo (matrix);

  oap::generic::printCustomMatrix (output, matrix, args, minfo);
}

template<typename GetMatrixInfo, typename NewMatrix, typename DeleteMatrix, typename CopyMatrixToMatrix>
void printMatrix (std::string& output, const math::ComplexMatrix* matrix, const matrixUtils::PrintArgs& args,
                  GetMatrixInfo&& getMatrixInfo, NewMatrix&& newMatrix, DeleteMatrix&& deleteMatrix, CopyMatrixToMatrix&& copyMatrixToMatrix)
{
  math::MatrixInfo minfo = getMatrixInfo(matrix);

  math::ComplexMatrix* hmatrix = newMatrix (minfo);
  copyMatrixToMatrix (hmatrix, matrix);

  oap::generic::printCustomMatrix (output, hmatrix, args, minfo);

  deleteMatrix (hmatrix);
}

template<typename GetMemory, typename GetMemoryRegion, typename Memcpy>
void setMatrix (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt column, uintt row, GetMemory&& getMemory, GetMemoryRegion&& getMemoryRegion, Memcpy&& memcpy, Memcpy&& memmove)
{
  oap::Memory hmem = getMemory (matrix);
  oap::Memory hmem1 = getMemory (matrix1);
  oap::MemoryRegion hreg = getMemoryRegion (matrix);
  oap::MemoryRegion hreg1 = getMemoryRegion (matrix1);

  debugAssert ((!hmem.ptr && !hmem1.ptr) || (hmem.ptr && hmem1.ptr));

  if (!hmem.ptr && !hmem1.ptr)
  {
    return;
  }

  oap::generic::copy (hmem.ptr, hmem.dims, oap::common::addLoc (hreg.loc, {column, row}), hmem1.ptr, hmem1.dims, GetRefMemoryRegion(hmem1, hreg1), memcpy, memmove);
}

template<typename GetRefMatrix, typename GetMemory, typename GetRegion, typename Memcpy>
floatt getDiagonal(const math::ComplexMatrix* matrix, uintt index, GetRefMatrix&& getRefMatrix, GetMemory&& getMemory, GetRegion&& getRegion, Memcpy&& gmemcpy, Memcpy&& memmove)
{
  math::ComplexMatrix hm = getRefMatrix (matrix);

  oapAssert (hm.dim.columns == hm.dim.rows);

  floatt v = 0;

  const oap::Memory memory = getMemory (matrix, hm);
  const oap::MemoryRegion region = getRegion (matrix, hm);

  if (memory.ptr)
  {
    oap::MemoryLoc loc = oap::common::ConvertRegionLocToMemoryLoc (memory, region, {index, index});
    oap::generic::copy (&v, {1, 1}, {0, 0}, memory.ptr, memory.dims, {loc, {1, 1}}, gmemcpy, memmove);
  }

  return v;
}

}
}

#endif /* OAP_HOST_MATRIX_UTILS_H */
