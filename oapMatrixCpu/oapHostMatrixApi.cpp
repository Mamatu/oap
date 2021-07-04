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

#include "oapHostMatrixApi.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdio.h>
#include <sstream>
#include <vector>

#include "oapGenericMatrixApi.hpp"
#include "oapHostMemoryApi.hpp"

#include "oapHostMatrixUPtr.hpp"

#include "MatrixParser.hpp"
#include "ReferencesCounter.hpp"

#include "GenericCoreApi.hpp"

#include "MatricesList.hpp"
#include "oapMemoryCounter.hpp"

namespace oap
{
namespace host
{

MatricesList g_matricesList ("MATRICES_HOST");

void SetValueToMatrix (math::Matrix* matrix, floatt v)
{
  using namespace oap::utils;

  math::Matrix hm = GetRefHostMatrix (matrix);

  if (hm.mem.ptr)
  {
    auto minfo = GetMatrixInfo (matrix);
    oap::HostMatrixUPtr uptr = oap::host::NewMatrixWithValue (minfo.columns(), minfo.rows(), v);

    oap::MemoryLoc loc = GetMatrixMemoryLoc (&hm);
    oap::MemoryRegion reg = GetMatrixMemoryRegion (uptr);
    oap::generic::copy (hm.mem.ptr, hm.mem.dims, loc, uptr->mem.ptr, uptr->mem.dims, reg, memcpy, memmove);
  }
}

oap::MemoryLoc GetMatrixMemoryLoc (const math::Matrix* matrix)
{
  return GetRefHostMatrix(matrix).reg.loc;
}

oap::MemoryRegion GetMatrixMemoryRegion (const math::Matrix* matrix)
{
  return GetRefHostMatrix(matrix).reg;
}

math::Matrix GetRefHostMatrix (const math::Matrix* matrix)
{
  if (!g_matricesList.contains (matrix))
  {
    oapAssert ("Not in list" == nullptr);
  }
  return *matrix;
}

math::MatrixInfo GetMatrixInfo (const math::Matrix* matrix)
{
  return g_matricesList.getUserData (matrix);
}

math::Matrix* NewMatrixWithValue (uintt columns, uintt rows, floatt value)
{
  math::Matrix* matrix = new math::Matrix;;
  matrix->reg.loc = {0, 0};;
  matrix->reg.dims = {columns, rows};;
  matrix->mem = oap::host::NewMemoryWithValues ({columns, rows}, value);
  return matrix;
}

math::Matrix* NewMatrix (uintt columns, uintt rows)
{
  math::Matrix* matrix = new math::Matrix;;
  matrix->reg.loc = {0, 0};;
  matrix->reg.dims = {columns, rows};;
  matrix->mem = oap::host::NewMemory ({columns, rows});
  return matrix;
}

void DeleteMatrix (const math::Matrix* matrix)
{
  if (nullptr == matrix)
  {
    return;
  }

  auto minfo = g_matricesList.remove (matrix);

  oap::host::DeleteMemory (matrix->mem);

  delete matrix;

  if (minfo.isInitialized ())
  {
    logTrace ("Deallocate: host matrix = %p %s", matrix, minfo.toString().c_str());
  }
  
}

uintt GetColumns (const math::Matrix* matrix)
{
  return matrix->reg.dims.width;
}

uintt GetRows (const math::Matrix* matrix)
{
  return matrix->reg.dims.height;
}

floatt GetValue (const math::Matrix* matrix, uintt column, uintt row)
{
  return oap::host::GetValue(matrix->mem, matrix->reg, column, row);
}

void CopyHostMatrixToHostMatrix (math::Matrix* dst, const math::Matrix* src)
{
  oapAssert(dst->reg.dims == src->reg.dims);
  oap::host::CopyHostToHost(dst->mem, dst->reg.loc, src->mem, src->reg);
}
}
}
