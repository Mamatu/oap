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

__hostdeviceinline__ uintt gMemoryWidth (const math::Matrix* matrix)
{
  if (matrix->re.ptr)
  {
    return matrix->re.dims.width;
  }
  if (matrix->im.ptr)
  {
    return matrix->im.dims.width;
  }
  return 0;
}

__hostdeviceinline__ uintt gMemoryHeight (const math::Matrix* matrix)
{
  if (matrix->re.ptr)
  {
    return matrix->re.dims.height;
  }
  if (matrix->im.ptr)
  {
    return matrix->im.dims.height;
  }
  return 0;
}

__hostdeviceinline__ uintt gMemoryColumns (const math::Matrix* matrix)
{
  return gMemoryWidth (matrix);
}

__hostdeviceinline__ uintt gMemoryRows (const math::Matrix* matrix)
{
  return gMemoryHeight (matrix);
}

__hostdeviceinline__ uintt gMemoryLength (const math::Matrix* matrix)
{
  return gMemoryWidth (matrix) * gMemoryHeight (matrix);
}


__hostdeviceinline__ uintt gColumns (const math::Matrix* matrix)
{
  if (matrix->re.ptr)
  {
    //if (!oap::common::IsNoneRegion (matrix->reReg))
    if (matrix->reReg.dims.width != 0)
    {
      return matrix->reReg.dims.width;
    }
    return matrix->re.dims.width;
  }
  if (matrix->im.ptr)
  {
    //if (!oap::common::IsNoneRegion (matrix->imReg))
    if (matrix->imReg.dims.width != 0)
    {
      return matrix->imReg.dims.width;
    }
    return matrix->im.dims.width;
  }
  return 0;
}

__hostdeviceinline__ uintt gRows (const math::Matrix* matrix)
{
  if (matrix->re.ptr)
  {
    //if (!oap::common::IsNoneRegion (matrix->reReg))
    if (matrix->reReg.dims.height != 0)
    {
      return matrix->reReg.dims.height;
    }
    return matrix->re.dims.height;
  }
  if (matrix->im.ptr)
  {
    //if (!oap::common::IsNoneRegion (matrix->imReg))
    if (matrix->imReg.dims.height != 0)
    {
      return matrix->imReg.dims.height;
    }
    return matrix->im.dims.height;
  }
  return 0;
}

__hostdeviceinline__ uintt GetColumns (const math::Matrix* matrix)
{
  return gColumns (matrix);
}

__hostdeviceinline__ uintt GetRows (const math::Matrix* matrix)
{
  return gRows (matrix);
}

__hostdeviceinline__ floatt* gReValues (const math::Matrix* matrix)
{
  if (matrix != NULL)
  {
    return matrix->re.ptr;
  }
  return NULL;
}

__hostdeviceinline__ floatt* gImValues (const math::Matrix* matrix)
{
  if (matrix != NULL)
  {
    return matrix->im.ptr;
  }
  return NULL;
}

__hostdeviceinline__ math::MatrixDim convertToMatrixDim (const math::Matrix* matrix)
{
  math::MatrixDim dim = {gColumns (matrix), gRows (matrix)};
  return dim;
}

__hostdeviceinline__ void SetRe(math::Matrix* m, uintt c, uintt r, floatt v)
{
  oap::common::SetValue (m->re, m->reReg, c, r, v);
}

__hostdeviceinline__ void SetReIdx (math::Matrix* m, uintt idx, floatt v)
{
  oap::common::SetValueRegionIdx (m->re, m->reReg, idx, v);
}

__hostdeviceinline__ floatt GetRe(const math::Matrix* m, uintt c, uintt r)
{
  debugAssert (c + r * gColumns (m) < gRows (m) * gColumns (m));
  return oap::common::GetValue (m->re, m->reReg, c, r);
}

__hostdeviceinline__ floatt GetReIdx(const math::Matrix* m, uintt index) {
  return oap::common::GetValueRegionIdx (m->re, m->reReg, index);
}

__hostdeviceinline__ floatt GetReIndex(const math::Matrix* m, uintt index) {
  return GetReIdx (m, index);
}

__hostdeviceinline__ floatt* GetRePtr(const math::Matrix* m, uintt c, uintt r)
{
  debugAssert (c + r * gColumns (m) < gRows (m) * gColumns (m));
  return oap::common::GetPtr (m->re, m->reReg, c, r);
}

__hostdeviceinline__ floatt* GetRePtrIdx(const math::Matrix* m, uintt index) {
  return oap::common::GetPtrRegionIdx (m->re, m->reReg, index);
}

__hostdeviceinline__ floatt* GetRePtrIndex(const math::Matrix* m, uintt index) {
  return GetRePtrIdx (m, index);
}

__hostdeviceinline__ void SetIm(math::Matrix* m, uintt c, uintt r, floatt v) {
  oap::common::SetValue (m->im, m->imReg, c, r, v);
}

__hostdeviceinline__ void SetImIdx (math::Matrix* m, uintt idx, floatt v)
{
  oap::common::SetValueRegionIdx (m->im, m->imReg, idx, v);
}

__hostdeviceinline__ floatt GetIm(const math::Matrix* m, uintt c, uintt r) {
  debugAssert (c + r * gColumns (m) < gRows (m) * gColumns (m));
  return oap::common::GetValue (m->im, m->imReg, c, r);
}

__hostdeviceinline__ floatt GetImIdx(const math::Matrix* m, uintt index) {
  return oap::common::GetValueRegionIdx (m->im, m->imReg, index);
}

__hostdeviceinline__ floatt GetImIndex(const math::Matrix* m, uintt index) {
  return GetImIdx (m, index);
}

__hostdeviceinline__ floatt* GetImPtr(const math::Matrix* m, uintt c, uintt r)
{
  debugAssert (c + r * gColumns (m) < gRows (m) * gColumns (m));
  return oap::common::GetPtr (m->im, m->imReg, c, r);
}

__hostdeviceinline__ floatt* GetImPtrIdx(const math::Matrix* m, uintt index) {
  return oap::common::GetPtrRegionIdx (m->im, m->imReg, index);
}

__hostdeviceinline__ floatt* GetImPtrIndex(const math::Matrix* m, uintt index) {
  return GetImPtrIdx (m, index);
}

__hostdeviceinline__ MatrixEx ConvertToMatrixEx (const oap::MemoryRegion& reg)
{
  MatrixEx mex = {reg.loc.x, reg.loc.y, reg.dims.width, reg.dims.height};
  return mex;
}

__hostdeviceinline__ oap::MemoryLoc GetReMatrixMemoryLoc (const math::Matrix* m)
{
  if (oap::common::isRegion (m->reReg))
  {
    return m->reReg.loc;
  }
  return {0, 0};
}

__hostdeviceinline__ oap::MemoryLoc GetImMatrixMemoryLoc (const math::Matrix* m)
{
  if (oap::common::isRegion (m->imReg))
  {
    return m->imReg.loc;
  }
  return {0, 0};
}

__hostdeviceinline__ oap::MemoryRegion GetReMatrixMemoryRegion (const math::Matrix* m)
{
  if (oap::common::isRegion (m->reReg))
  {
    return m->reReg;
  }
  return {{0, 0}, m->re.dims};
}

__hostdeviceinline__ oap::MemoryRegion GetImMatrixMemoryRegion (const math::Matrix* m)
{
  if (oap::common::isRegion (m->imReg))
  {
    return m->imReg;
  }
  return {{0, 0}, m->im.dims};
}

__hostdeviceinline__ oap::MemoryRegion MergeRegions (const oap::MemoryRegion* region, const oap::MemoryRegion* in)
{
  debugAssert (in->loc.x + in->dims.width < region->dims.width);
  debugAssert (in->loc.y + in->dims.height < region->dims.height);
  oap::MemoryRegion reg = {{region->loc.x + in->loc.x, region->loc.y + in->loc.y}, in->dims};
  return reg;
}

__hostdeviceinline__ oap::MemoryRegion GetReMatrixMemoryRegion (const math::Matrix* m, const oap::MemoryRegion* region)
{
  oap::MemoryRegion matrixRegion = GetReMatrixMemoryRegion (m);
  if (region)
  {
    matrixRegion = MergeRegions (&matrixRegion, region);
  }
  return matrixRegion;
}

__hostdeviceinline__ oap::MemoryRegion GetImMatrixMemoryRegion (const math::Matrix* m, const oap::MemoryRegion* region)
{
  oap::MemoryRegion matrixRegion = GetImMatrixMemoryRegion (m);
  if (region)
  {
    matrixRegion = MergeRegions (&matrixRegion, region);
  }
  return matrixRegion;
}

__hostdeviceinline__ oap::MemoryLoc GetReMatrixMemoryLoc (const math::Matrix* m, const oap::MemoryLoc* loc)
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

__hostdeviceinline__ oap::MemoryLoc GetImMatrixMemoryLoc (const math::Matrix* m, const oap::MemoryLoc* loc)
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


__hostdeviceinline__ void Reset(math::Matrix* m) {}

__hostdeviceinline__ void Push(math::Matrix* m) {}

__hostdeviceinline__ void Pop(math::Matrix* m) {}


#endif  // MATRIXAPI_H
