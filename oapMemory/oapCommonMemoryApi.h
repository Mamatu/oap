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

#ifndef OAP_COMMON_MEMORY_API_H
#define OAP_COMMON_MEMORY_API_H

#include "Math.h"
#include "oapMemoryPrimitives.h"

#include "CuCore.h"

const oap::MemoryRegion OAP_NONE_REGION = {{0, 0}, {0, 0}};

namespace oap
{
namespace utils
{

__hostdeviceinline__ bool isNone (const oap::MemoryRegion& reg)
{
  return reg.loc.x == 0 && reg.loc.y == 0 && reg.dims.width == 0 && reg.dims.height == 0;
}

__hostdeviceinline__ void setToRegion (oap::MemoryRegion& reg, const oap::MemoryDims& memoryDims, const oap::MemoryLoc& loc = {0, 0})
{
  reg.loc = loc;
  reg.dims.width = memoryDims.width - loc.x;
  reg.dims.height = memoryDims.height - loc.y;
}

__hostdeviceinline__ oap::MemoryRegion convertToRegion (const oap::MemoryDims& memoryDims, const oap::MemoryLoc& loc = {0, 0})
{
  oap::MemoryRegion reg;
  setToRegion (reg, memoryDims, loc);
  return reg;
}

__hostdeviceinline__ uintt GetIdx (const oap::Memory* memory, const oap::MemoryRegion& reg, uintt x, uintt y)
{
  debugAssert (isNone (reg) || memory->dims.width >= reg.loc.x + reg.dims.width);
  debugAssert (isNone (reg) || memory->dims.height >= reg.loc.y + reg.dims.height);
  return (x + reg.loc.x) + memory->dims.width * (y + reg.loc.y);
}

__hostdeviceinline__ floatt* GetPtr (const oap::Memory* memory, const oap::MemoryRegion& reg, uintt x, uintt y)
{
  return memory->ptr + (GetIdx (memory, reg, x, y));
}

__hostdeviceinline__ floatt GetValue (const oap::Memory* memory, const oap::MemoryRegion& reg, uintt x, uintt y)
{
  return memory->ptr[GetIdx (memory, reg, x, y)];
}
}
}
#endif
