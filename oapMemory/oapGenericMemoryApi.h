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
#include "oapCommonMemoryApi.h"

namespace oap
{
namespace utils
{
  inline void check (const oap::MemoryDims& dims, const oap::MemoryRegion& region)
  {
    logAssert (dims.height >= region.loc.y + region.dims.height && dims.width >= region.loc.x + region.dims.width);
  }

  template<typename Ptr1, typename Ptr2>
  void getPtrs (Ptr1* ptrs, Ptr2 mem, const oap::MemoryDims& dims, const oap::MemoryRegion& region)
  {
    check (dims, region);
    Ptr2 ptr = mem + region.loc.y * dims.width + region.loc.x;
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
    getPtrs <typename Container::value_type, Ptr> (container.data(), mem, dims, region);
  }

  inline void getPtrs (floatt** ptrs, floatt* ptr, const oap::MemoryDims& memDims, const oap::MemoryRegion& region)
  {
    getPtrs<floatt*> (ptrs, ptr, memDims, region);
  }

  template<typename Container>
  void getPtrs (Container& container, const oap::Memory* memory, const oap::MemoryRegion& region)
  {
    getPtrs (container, memory->ptr, memory->dims, region);
  }
}
namespace generic
{
  template<typename Allocator>
  oap::Memory* newMemory (const MemoryDims& dims, Allocator&& allocator)
  {
    logAssert (dims.width > 0 && dims.height > 0);
    return allocator (dims);
  }

  template<typename Allocator>
  oap::Memory* newMemoryWithValues (const MemoryDims& dims, floatt value, Allocator&& allocator)
  {
    logAssert (dims.width > 0 && dims.height > 0);
    return allocator (dims, value);
  }

  template<typename Allocator>
  oap::Memory* newMemoryCopy (const oap::Memory* src, Allocator&& allocator)
  {
    return allocator (src);
  }

  template<typename Allocator>
  oap::Memory* newMemoryCopyMem (const oap::Memory* src, uintt width, uintt height, Allocator&& allocator)
  {
    return allocator (src, width, height);
  }

  template<typename Allocator>
  oap::Memory* reuseMemory (const oap::Memory* src, uintt width, uintt height, Allocator&& allocator)
  {
    return allocator (src, width, height);
  }

  template<typename Deallocator>
  void deleteMemory (const oap::Memory* mem, Deallocator&& deallocator)
  {
    return deallocator (mem);
  }

  template<typename GetD>
  oap::MemoryDims getDims (const oap::Memory* mem, GetD&& getD)
  {
    return getD (mem);
  }

  template<typename GetRM>
  floatt* getRawMemory (const oap::Memory* mem, GetRM&& getRM)
  {
    return getRM (mem);
  }

  template<typename Memcpy>
  void copy (oap::Memory* dst, const oap::MemoryLoc& dstLoc, const oap::Memory* src, const oap::MemoryRegion& srcReg, Memcpy&& memcpy)
  {
    logAssert (dst->dims.width >= srcReg.dims.width);
    logAssert (dst->dims.height >= srcReg.dims.height);
  
    std::vector<floatt*> dstPtrs;
    std::vector<const floatt*> srcPtrs;

    auto dstReg = utils::convertToRegion (dst->dims, dstLoc);

    utils::getPtrs (dstPtrs, dst, dstReg);
    utils::getPtrs (srcPtrs, src, srcReg);

    for (size_t idx = 0; idx < srcPtrs.size (); ++idx)
    {
      const floatt* srcPtr = srcPtrs[idx];
      floatt* dstPtr = dstPtrs[idx];
      memcpy (dstPtr, srcPtr, srcReg.dims.width * sizeof (floatt));
    }
  }

  template<typename Memcpy>
  void copy (oap::Memory* dst, const oap::Memory* src, Memcpy&& memcpy)
  {
    const auto& srcReg = utils::convertToRegion (src->dims);
    copy<Memcpy> (dst, {0, 0}, src, srcReg, memcpy);
  }

  template<typename Memcpy>
  void copy (floatt* dst, const oap::MemoryDims& dstDims, const oap::MemoryLoc& dstLoc, const floatt* src, const oap::MemoryDims& srcDims, const oap::MemoryRegion& srcReg, Memcpy&& memcpy)
  {
    logAssert (dstDims.width >= srcReg.dims.width);
    logAssert (dstDims.height >= srcReg.dims.height);

    std::vector<floatt*> dstPtrs;
    std::vector<const floatt*> srcPtrs;

    auto dstReg = utils::convertToRegion (dstDims, dstLoc);

    utils::getPtrs (dstPtrs, dst, dstDims, dstReg);
    utils::getPtrs (srcPtrs, src, srcDims, srcReg);

    for (size_t idx = 0; idx < srcPtrs.size (); ++idx)
    {
      const floatt* srcPtr = srcPtrs[idx];
      floatt* dstPtr = dstPtrs[idx];
      memcpy (dstPtr, srcPtr, srcReg.dims.width * sizeof (floatt));
    }
  }
}
}

#endif
