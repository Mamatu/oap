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

#ifndef OAP_GENERIC_MEMORY_API_H
#define OAP_GENERIC_MEMORY_API_H

#include <vector>

#include "Math.h"
#include "Logger.h"

#include "oapMemoryPrimitives.h"

namespace oap
{
namespace utils
{
  inline void check (const oap::MemoryDims& dims, const oap::MemoryRegion& region)
  {
    logAssert (dims.height >= region.loc.y + region.dims.height && dims.width >= region.loc.x + region.dims.width);
  }

  template<typename Ptr>
  void getPtrs (Ptr* ptrs, Ptr mem, const oap::MemoryDims& dims, const oap::MemoryRegion& region)
  {
    check (dims, region);
    Ptr ptr = mem + region.loc.y * dims.width + region.loc.x;
    for (uintt idx = 0; idx < region.dims.height; ++idx)
    {
      ptrs[idx] = ptr;
      ptr += dims.width;
    }
  }

  template<typename Container, typename Ptr>
  void getPtrs (Container& container, Ptr mem, const oap::MemoryDims& dims, const oap::MemoryRegion& region)
  {
    if (container.size() < region.dims.height)
    {
      container.resize (region.dims.height);
    }
    getPtrs <Ptr> (container.data(), mem, dims, region);
  }

  inline void getPtrs (floatt** ptrs, const oap::Memory* memory, const oap::MemoryRegion& region)
  {
    getPtrs<floatt*> (ptrs, memory->ptr, memory->dims, region);
  }

  template<typename Container>
  void getPtrs (Container& container, const oap::Memory* memory, const oap::MemoryRegion& region)
  {
    getPtrs (container, memory->ptr, memory->dims, region);
  }

  inline void convertToRegion (oap::MemoryRegion& reg, const oap::Memory* memory, const oap::MemoryLoc& loc = {0, 0})
  {
    reg.loc = loc;
    reg.dims.width = memory->dims.width - loc.x;
    reg.dims.height = memory->dims.height - loc.y;
  }

  inline oap::MemoryRegion convertToRegion (const oap::Memory* memory, const oap::MemoryLoc& loc = {0, 0})
  {
    oap::MemoryRegion reg;
    convertToRegion (reg, memory, loc);
    return reg;
  }

  inline uintt GetIdx (oap::Memory* memory, const oap::MemoryRegion& reg, uintt x, uintt y)
  {
    uintt idx = (x + reg.loc.x) + memory->dims.width * (y + reg.loc.y);
    return idx;
  }

  inline floatt* GetPtr (oap::Memory* memory, const oap::MemoryRegion& reg, uintt x, uintt y)
  {
    return memory->ptr + (GetIdx (memory, reg, x, y));
  }

  inline floatt GetValue (oap::Memory* memory, const oap::MemoryRegion& reg, uintt x, uintt y)
  {
    return memory->ptr[GetIdx (memory, reg, x, y)];
  }

  inline uintt GetIdx (oap::Memory* memory, uintt x, uintt y)
  {
    return GetIdx (memory, convertToRegion (memory), x, y);
  }

  inline floatt* GetPtr (oap::Memory* memory, uintt x, uintt y)
  {
    return memory->ptr + (GetIdx (memory, x, y));
  }

  inline floatt GetValue (oap::Memory* memory, uintt x, uintt y)
  {
    return *GetPtr (memory, x, y);
  }

}
namespace generic
{
  template<typename Memcpy>
  void copyMemoryRegion (oap::Memory* dst, const oap::MemoryLoc& dstLoc, const oap::Memory* src, const oap::MemoryRegion& srcReg, Memcpy&& memcpy)
  {
    logAssert (dst->dims.width >= srcReg.dims.width);
    logAssert (dst->dims.height >= srcReg.dims.height);
  
    std::vector<floatt*> dstPtrs;
    std::vector<floatt*> srcPtrs;

    auto dstReg = utils::convertToRegion (dst, dstLoc);

    utils::getPtrs (dstPtrs, dst, dstReg);
    utils::getPtrs (srcPtrs, src, srcReg);

    for (size_t idx = 0; idx < srcPtrs.size (); ++idx)
    {
      floatt* srcPtr = srcPtrs[idx];
      floatt* dstPtr = dstPtrs[idx];
      memcpy (dstPtr, srcPtr, srcReg.dims.width * sizeof (floatt));
    }
  }
}
}

#endif
