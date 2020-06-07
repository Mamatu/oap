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

#ifndef OAP_MEMORY__COMMON_API_H
#define OAP_MEMORY__COMMON_API_H

#include "Math.h"
#include "oapMemoryPrimitives.h"

#include "CuCore.h"

namespace oap
{
namespace common
{

__hostdeviceinline__ oap::Memory OAP_NONE_MEMORY()
{
  oap::Memory memory = {0, {0, 0}};
  return memory;
}

__hostdeviceinline__ oap::MemoryRegion OAP_NONE_REGION()
{
  oap::MemoryRegion region = {{0, 0}, {0, 0}};
  return region;
}

__hostdeviceinline__ oap::MemoryLoc OAP_NONE_LOCATION()
{
  oap::MemoryLoc loc = {0, 0};
  return loc;
}

__hostdeviceinline__ const oap::Memory OAP_NONE_MEMORY_CONST()
{
  oap::Memory memory = {0, {0, 0}};
  return memory;
}

__hostdeviceinline__ const oap::MemoryRegion OAP_NONE_REGION_CONST()
{
  oap::MemoryRegion region = {{0, 0}, {0, 0}};
  return region;
}

__hostdeviceinline__ const oap::MemoryLoc OAP_NONE_LOCATION_CONST()
{
  oap::MemoryLoc loc = {0, 0};
  return loc;
}

__hostdeviceinline__ bool CompareMemoryDims (const oap::MemoryDims& dims1, const oap::MemoryDims& dims2)
{
  return dims1.width == dims2.width && dims1.height == dims2.height;
}

  __hostdeviceinline__ bool IsNoneMemory (const oap::Memory& mem1)
{
  return mem1.ptr == OAP_NONE_MEMORY().ptr && CompareMemoryDims (mem1.dims, OAP_NONE_MEMORY().dims);
}

__hostdeviceinline__ bool isRegion (const oap::MemoryRegion& reg)
{
  return !(reg.dims.width == 0 || reg.dims.height == 0);
}

__hostdeviceinline__ bool IsNoneRegion (const oap::MemoryRegion& region)
{
  return !isRegion (region);
}

__hostdeviceinline__ bool CompareMemoryRegion (const oap::MemoryRegion& reg1, const oap::MemoryRegion& reg2)
{
  return reg1.loc.x == reg2.loc.x && reg1.loc.y == reg2.loc.y && reg1.dims.width == reg2.dims.width && reg1.dims.height == reg2.dims.height;
}

#if 0
__hostdeviceinline__ bool isNone (const oap::MemoryRegion& reg)
{
  return reg.loc.x == 0 && reg.loc.y == 0 && reg.dims.width == 0 && reg.dims.height == 0;
}
#endif

__hostdeviceinline__ uintt GetLocX (const oap::MemoryRegion& reg)
{
  if (!oap::common::isRegion (reg))
  {
    return 0;
  }
  return reg.loc.x;
}

__hostdeviceinline__ uintt GetLocY (const oap::MemoryRegion& reg)
{
  if (!oap::common::isRegion (reg))
  {
    return 0;
  }
  return reg.loc.y;
}

__hostdeviceinline__ uintt GetWidth (const oap::MemoryRegion& reg)
{
  if (!oap::common::isRegion (reg))
  {
    return 0;
  }
  return reg.dims.width;
}

__hostdeviceinline__ uintt GetHeight (const oap::MemoryRegion& reg)
{
  if (!oap::common::isRegion (reg))
  {
    return 0;
  }
  return reg.dims.height;
}

__hostdeviceinline__ oap::MemoryLoc addLoc (const oap::MemoryLoc& loc1, const oap::MemoryLoc& loc2)
{
  oap::MemoryLoc l = {loc1.x + loc2.x, loc1.y + loc2.y};
  return l;
}

__hostdeviceinline__ void setToRegion (oap::MemoryRegion& reg, const oap::MemoryDims& memoryDims, const oap::MemoryLoc& loc)
{
  reg.loc = loc;
  reg.dims.width = memoryDims.width - loc.x;
  reg.dims.height = memoryDims.height - loc.y;
}

__hostdeviceinline__ oap::MemoryRegion convertToRegion (const oap::MemoryDims& memoryDims, const oap::MemoryLoc& loc)
{
  oap::MemoryRegion reg;
  setToRegion (reg, memoryDims, loc);
  return reg;
}

__hostdeviceinline__ uintt GetIdx (const oap::Memory& memory, const oap::MemoryRegion& reg, uintt x, uintt y)
{
  debugAssert (!isRegion (reg) || memory.dims.width >= reg.loc.x + reg.dims.width);
  debugAssert (!isRegion (reg) || memory.dims.height >= reg.loc.y + reg.dims.height);
  const uintt bufferIndex = (x + GetLocX (reg)) + memory.dims.width * (y + GetLocY (reg));
  debugAssert (bufferIndex < memory.dims.width * memory.dims.height);
  return bufferIndex;
}

__hostdeviceinline__ oap::MemoryLoc ConvertDimsIdxToLoc (uintt idx, const oap::MemoryDims& dims)
{
  debugAssert (idx < dims.width * dims.height);
  uintt y = idx / dims.width;
  uintt x = idx - (y * dims.width);
  oap::MemoryLoc loc = {x, y};
  return loc;
}

__hostdeviceinline__ oap::MemoryLoc ConvertRegionLocToMemoryLoc (const oap::Memory& memory, const oap::MemoryRegion& reg, const oap::MemoryLoc& loc)
{
  debugAssert (loc.x < reg.dims.width);
  debugAssert (loc.y < reg.dims.height);

  uintt mx = reg.loc.x + loc.x;
  uintt my = reg.loc.y + loc.y;

  debugAssert (mx < memory.dims.width);
  debugAssert (my < memory.dims.height);

  oap::MemoryLoc memLoc = {mx, my};
  return memLoc;
}

__hostdeviceinline__ oap::MemoryLoc ConvertIdxToMemoryLocRef (uintt idx, const oap::Memory& memory, const oap::MemoryRegion& region)
{
  debugAssert (!IsNoneMemory (memory));
  if (!oap::common::isRegion (region))
  {
    return ConvertDimsIdxToLoc (idx, memory.dims);
  }
  oap::MemoryLoc loc = ConvertDimsIdxToLoc (idx, region.dims);
  return ConvertRegionLocToMemoryLoc (memory, region, loc);
}

__hostdeviceinline__ oap::MemoryLoc ConvertIdxToMemoryLoc (uintt idx, const oap::Memory& memory, const oap::MemoryRegion& region)
{
  return ConvertIdxToMemoryLocRef (idx, memory, region);
}

__hostdeviceinline__ oap::MemoryLoc ConvertIdxToRegionLocRef (uintt idx, const oap::Memory& memory, const oap::MemoryRegion& region)
{
  debugAssert (!IsNoneMemory (memory));
  if (oap::common::IsNoneRegion (region))
  {
    return ConvertDimsIdxToLoc (idx, memory.dims);
  }
  return ConvertDimsIdxToLoc (idx, region.dims);
}

__hostdeviceinline__ oap::MemoryLoc ConvertIdxToRegionLoc (uintt idx, const oap::Memory& memory, const oap::MemoryRegion& region)
{
  return ConvertIdxToRegionLocRef (idx, memory, region);
}

__hostdeviceinline__ uintt GetMemoryRegionIdx (const oap::Memory& memory, const oap::MemoryRegion& reg, uintt idx)
{
  oap::MemoryLoc loc = ConvertIdxToRegionLoc (idx, memory, reg);
  return GetIdx (memory, reg, loc.x, loc.y);
}

__hostdeviceinline__ floatt* GetPtr (const oap::Memory& memory, const oap::MemoryRegion& reg, uintt x, uintt y)
{
  return memory.ptr + (GetIdx (memory, reg, x, y));
}

__hostdeviceinline__ floatt GetValue (const oap::Memory& memory, const oap::MemoryRegion& reg, uintt x, uintt y)
{
  return memory.ptr[GetIdx (memory, reg, x, y)];
}

__hostdeviceinline__ floatt GetValueRef (const oap::Memory& memory, const oap::MemoryRegion& reg, uintt x, uintt y)
{
  return memory.ptr[GetIdx (memory, reg, x, y)];
}

__hostdeviceinline__ void SetValue (oap::Memory& memory, const oap::MemoryRegion& reg, uintt x, uintt y, floatt v)
{
  memory.ptr[GetIdx (memory, reg, x, y)] = v;
}

__hostdeviceinline__ floatt* GetPtrRegionIdx (const oap::Memory& memory, const oap::MemoryRegion& reg, uintt idx)
{
  return memory.ptr + (GetMemoryRegionIdx (memory, reg, idx));
}

__hostdeviceinline__ floatt GetValueRegionIdx (const oap::Memory& memory, const oap::MemoryRegion& reg, uintt idx)
{
  return memory.ptr[GetMemoryRegionIdx (memory, reg, idx)];
}

__hostdeviceinline__ void SetValueRegionIdx (oap::Memory& memory, const oap::MemoryRegion& reg, uintt idx, floatt v)
{
  memory.ptr[GetMemoryRegionIdx (memory, reg, idx)] = v;
}
}
}
#endif
