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

#ifndef OAP_MATRIXAPI_H
#define OAP_MATRIXAPI_H

#include "Matrix.h"
#include "oapMemory_CommonApi.h"
#include "MatrixTestAPI.h"

#include "CuCore.h"
#include "Dim3.h"

#include "oapAssertion.h"

__hostdeviceinline__ uintt gColumns (const math::Matrix* matrix)
{
  if (matrix->re)
  {
    if (matrix->reReg)
    {
      return matrix->reReg->dims.width;
    }
    return matrix->re->dims.width;
  }
  if (matrix->im)
  {
    if (matrix->imReg)
    {
      return matrix->imReg->dims.width;
    }
    return matrix->im->dims.width;
  }
  return 0;
}

__hostdeviceinline__ uintt gRows (const math::Matrix* matrix)
{
  if (matrix->re)
  {
    if (matrix->reReg)
    {
      return matrix->reReg->dims.height;
    }
    return matrix->re->dims.height;
  }
  if (matrix->im)
  {
    if (matrix->imReg)
    {
      return matrix->imReg->dims.height;
    }
    return matrix->im->dims.height;
  }
  return 0;
}

__hostdeviceinline__ floatt* gReValues (const math::Matrix* matrix)
{
  return matrix->re->ptr;
}

__hostdeviceinline__ floatt* gImValues (const math::Matrix* matrix)
{
  return matrix->im->ptr;
}

__hostdeviceinline__ void SetRe(math::Matrix* m, uintt c, uintt r, floatt v)
{
  oap::utils::SetValue (m->re, m->reReg, c, r, v);
}

__hostdeviceinline__ floatt GetRe(const math::Matrix* m, uintt c, uintt r)
{
  debugAssert (c + r * columns (m) < rows (m) * columns (m));
  return oap::utils::GetValue (m->re, m->reReg, c, r);
}

/*__hostdeviceinline__ floatt GetReIndex(const math::Matrix* m, uintt index) {
  debugAssert (index < rows (m) * columns (m));
  return m->reValues[index];
}*/

__hostdeviceinline__ void SetIm(math::Matrix* m, uintt c, uintt r, floatt v) {
  oap::utils::SetValue (m->im, m->imReg, c, r, v);
}

__hostdeviceinline__ floatt GetIm(const math::Matrix* m, uintt c, uintt r) {
  debugAssert (c + r * columns (m) < rows (m) * columns (m));
  return oap::utils::GetValue (m->im, m->imReg, c, r);
}

/*__hostdeviceinline__ floatt GetImIndex(const math::Matrix* m, uintt index) {
  debugAssert (index < rows (m) * columns (m));
  return m->imValues[index];
}*/

__hostdeviceinline__ void Reset(math::Matrix* m) {}

__hostdeviceinline__ void Push(math::Matrix* m) {}

__hostdeviceinline__ void Pop(math::Matrix* m) {}


#endif  // MATRIXAPI_H
