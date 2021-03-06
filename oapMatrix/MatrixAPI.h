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

#ifndef OAP_MATRIXAPI_H
#define OAP_MATRIXAPI_H

#include "Matrix.h"
#include "MatrixEx.h"
#include "MatrixTestAPI.h"

#include "CuCore.h"
#include "Dim3.h"

#include "oapAssertion.h"
#include "oapMemory_CommonApi.h"

#include "oapMemoryPrimitives.h"

__hostdeviceinline__ uintt gMemoryWidth (const math::ComplexMatrix* matrix)
{
  if (matrix->re.mem.ptr)
  {
    return matrix->re.mem.dims.width;
  }
  if (matrix->im.mem.ptr)
  {
    return matrix->im.mem.dims.width;
  }
  return 0;
}

__hostdeviceinline__ uintt gMemoryHeight (const math::ComplexMatrix* matrix)
{
  if (matrix->re.mem.ptr)
  {
    return matrix->re.mem.dims.height;
  }
  if (matrix->im.mem.ptr)
  {
    return matrix->im.mem.dims.height;
  }
  return 0;
}

__hostdeviceinline__ uintt gMemoryColumns (const math::ComplexMatrix* matrix)
{
  return gMemoryWidth (matrix);
}

__hostdeviceinline__ uintt gMemoryRows (const math::ComplexMatrix* matrix)
{
  return gMemoryHeight (matrix);
}

__hostdeviceinline__ uintt gMemoryLength (const math::ComplexMatrix* matrix)
{
  return gMemoryWidth (matrix) * gMemoryHeight (matrix);
}


__hostdeviceinline__ uintt gColumns (const math::ComplexMatrix* matrix)
{
  return matrix->dim.columns;
}

__hostdeviceinline__ uintt gRows (const math::ComplexMatrix* matrix)
{
  return matrix->dim.rows;
}

__hostdeviceinline__ uintt GetColumns (const math::ComplexMatrix* matrix)
{
  return gColumns (matrix);
}

__hostdeviceinline__ uintt GetRows (const math::ComplexMatrix* matrix)
{
  return gRows (matrix);
}

__hostdeviceinline__ floatt* gReValues (const math::ComplexMatrix* matrix)
{
  if (matrix != NULL)
  {
    return matrix->re.mem.ptr;
  }
  return NULL;
}

__hostdeviceinline__ floatt* gImValues (const math::ComplexMatrix* matrix)
{
  if (matrix != NULL)
  {
    return matrix->im.mem.ptr;
  }
  return NULL;
}

__hostdeviceinline__ math::MatrixDim convertToMatrixDim (const math::ComplexMatrix* matrix)
{
  math::MatrixDim dim = {gColumns (matrix), gRows (matrix)};
  return dim;
}

__hostdeviceinline__ void SetRe(math::ComplexMatrix* m, uintt c, uintt r, floatt v)
{
  oap::common::SetValue (m->re.mem, m->re.reg, c, r, v);
}

__hostdeviceinline__ void SetReIdx (math::ComplexMatrix* m, uintt idx, floatt v)
{
  oap::common::SetValueRegionIdx (m->re.mem, m->re.reg, idx, v);
}

__hostdeviceinline__ floatt GetRe(const math::ComplexMatrix* m, uintt c, uintt r)
{
  debugAssert (c + r * gColumns (m) < gRows (m) * gColumns (m));
  return oap::common::GetValue (m->re.mem, m->re.reg, c, r);
}

__hostdeviceinline__ floatt GetReIdx(const math::ComplexMatrix* m, uintt index) {
  return oap::common::GetValueRegionIdx (m->re.mem, m->re.reg, index);
}

__hostdeviceinline__ floatt GetReIndex(const math::ComplexMatrix* m, uintt index) {
  return GetReIdx (m, index);
}

__hostdeviceinline__ floatt* GetRePtr(const math::ComplexMatrix* m, uintt c, uintt r)
{
  debugAssert (c + r * gColumns (m) < gRows (m) * gColumns (m));
  return oap::common::GetPtr (m->re.mem, m->re.reg, c, r);
}

__hostdeviceinline__ floatt* GetRePtrIdx(const math::ComplexMatrix* m, uintt index) {
  return oap::common::GetPtrRegionIdx (m->re.mem, m->re.reg, index);
}

__hostdeviceinline__ floatt* GetRePtrIndex(const math::ComplexMatrix* m, uintt index) {
  return GetRePtrIdx (m, index);
}

__hostdeviceinline__ void SetIm(math::ComplexMatrix* m, uintt c, uintt r, floatt v) {
  oap::common::SetValue (m->im.mem, m->im.reg, c, r, v);
}

__hostdeviceinline__ void SetImIdx (math::ComplexMatrix* m, uintt idx, floatt v)
{
  oap::common::SetValueRegionIdx (m->im.mem, m->im.reg, idx, v);
}

__hostdeviceinline__ floatt GetIm(const math::ComplexMatrix* m, uintt c, uintt r) {
  debugAssert (c + r * gColumns (m) < gRows (m) * gColumns (m));
  return oap::common::GetValue (m->im.mem, m->im.reg, c, r);
}

__hostdeviceinline__ floatt GetImIdx(const math::ComplexMatrix* m, uintt index) {
  return oap::common::GetValueRegionIdx (m->im.mem, m->im.reg, index);
}

__hostdeviceinline__ floatt GetImIndex(const math::ComplexMatrix* m, uintt index) {
  return GetImIdx (m, index);
}

__hostdeviceinline__ floatt* GetImPtr(const math::ComplexMatrix* m, uintt c, uintt r)
{
  debugAssert (c + r * gColumns (m) < gRows (m) * gColumns (m));
  return oap::common::GetPtr (m->im.mem, m->im.reg, c, r);
}

__hostdeviceinline__ floatt* GetImPtrIdx(const math::ComplexMatrix* m, uintt index) {
  return oap::common::GetPtrRegionIdx (m->im.mem, m->im.reg, index);
}

__hostdeviceinline__ floatt* GetImPtrIndex(const math::ComplexMatrix* m, uintt index) {
  return GetImPtrIdx (m, index);
}

__hostdeviceinline__ MatrixEx ConvertToMatrixEx (const oap::MemoryRegion& reg)
{
  MatrixEx mex = {reg.loc.x, reg.loc.y, reg.dims.width, reg.dims.height};
  return mex;
}

__hostdeviceinline__ oap::MemoryLoc GetReMatrixMemoryLoc (const math::ComplexMatrix* m)
{
  if (oap::common::isRegion (m->re.reg))
  {
    return m->re.reg.loc;
  }
  return {0, 0};
}

__hostdeviceinline__ oap::MemoryLoc GetImMatrixMemoryLoc (const math::ComplexMatrix* m)
{
  if (oap::common::isRegion (m->im.reg))
  {
    return m->im.reg.loc;
  }
  return {0, 0};
}

__hostdeviceinline__ oap::MemoryRegion GetReMatrixMemoryRegion (const math::ComplexMatrix* m)
{
  if (oap::common::isRegion (m->re.reg))
  {
    return m->re.reg;
  }
  return {{0, 0}, m->re.mem.dims};
}

__hostdeviceinline__ oap::MemoryRegion GetImMatrixMemoryRegion (const math::ComplexMatrix* m)
{
  if (oap::common::isRegion (m->im.reg))
  {
    return m->im.reg;
  }
  return {{0, 0}, m->im.mem.dims};
}

__hostdeviceinline__ oap::MemoryRegion MergeRegions (const oap::MemoryRegion* region, const oap::MemoryRegion* in)
{
  debugAssert (in->loc.x + in->dims.width < region->dims.width);
  debugAssert (in->loc.y + in->dims.height < region->dims.height);
  oap::MemoryRegion reg = {{region->loc.x + in->loc.x, region->loc.y + in->loc.y}, in->dims};
  return reg;
}

__hostdeviceinline__ oap::MemoryRegion GetReMatrixMemoryRegion (const math::ComplexMatrix* m, const oap::MemoryRegion* region)
{
  oap::MemoryRegion matrixRegion = GetReMatrixMemoryRegion (m);
  if (region)
  {
    matrixRegion = MergeRegions (&matrixRegion, region);
  }
  return matrixRegion;
}

__hostdeviceinline__ oap::MemoryRegion GetImMatrixMemoryRegion (const math::ComplexMatrix* m, const oap::MemoryRegion* region)
{
  oap::MemoryRegion matrixRegion = GetImMatrixMemoryRegion (m);
  if (region)
  {
    matrixRegion = MergeRegions (&matrixRegion, region);
  }
  return matrixRegion;
}

__hostdeviceinline__ oap::MemoryLoc GetReMatrixMemoryLoc (const math::ComplexMatrix* m, const oap::MemoryLoc* loc)
{
  oap::MemoryLoc matrixLoc = GetReMatrixMemoryLoc (m);
  if (loc)
  {
    debugAssert (loc->x < gColumns (m));
    debugAssert (loc->y < gRows (m));
    matrixLoc.x = matrixLoc.x + loc->x;
    matrixLoc.y = matrixLoc.y + loc->y;
  }
  return matrixLoc;
}

__hostdeviceinline__ oap::MemoryLoc GetImMatrixMemoryLoc (const math::ComplexMatrix* m, const oap::MemoryLoc* loc)
{
  oap::MemoryLoc matrixLoc = GetImMatrixMemoryLoc (m);
  if (loc)
  {
    debugAssert (loc->x < gColumns (m));
    debugAssert (loc->y < gRows (m));
    matrixLoc.x = matrixLoc.x + loc->x;
    matrixLoc.y = matrixLoc.y + loc->y;
  }
  return matrixLoc;
}

__hostdeviceinline__ oap::MemoryRegion ConvertToMemoryRegion (const MatrixEx& matrixEx)
{
  oap::MemoryRegion region;

  region.loc.x = matrixEx.column;
  region.loc.y = matrixEx.row;
  region.dims.width = matrixEx.columns;
  region.dims.height = matrixEx.rows;

  return region;
}

__hostdeviceinline__ oap::MemoryRegion GetRefMemoryRegion (const oap::Memory& memory, const oap::MemoryRegion& region)
{
  if (oap::common::IsNoneRegion (region))
  {
    oap::MemoryRegion refreg = {{0, 0}, memory.dims};
    return refreg;
  }
  return region;
}


__hostdeviceinline__ void Reset(math::ComplexMatrix* m) {}

__hostdeviceinline__ void Push(math::ComplexMatrix* m) {}

__hostdeviceinline__ void Pop(math::ComplexMatrix* m) {}


#endif  // MATRIXAPI_H
