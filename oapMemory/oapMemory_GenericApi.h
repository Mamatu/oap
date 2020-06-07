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

#ifndef OAP_MEMORY__GENERIC_API_H
#define OAP_MEMORY__GENERIC_API_H

#include <vector>

#include "Math.h"
#include "Logger.h"

#include "oapMemoryPrimitives.h"
#include "oapMemoryPrimitivesApi.h"
#include "oapMemory_CommonApi.h"

namespace oap
{

  enum DataDirection
  {
    HORIZONTAL,
    VERTICAL
  };

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
  void getPtrs (Container& container, const oap::Memory& memory, const oap::MemoryRegion& region)
  {
    getPtrs (container, memory.ptr, memory.dims, region);
  }
}
namespace generic
{
  template<typename Allocator>
  oap::Memory newMemory (const MemoryDims& dims, Allocator&& allocator)
  {
    logAssert (dims.width > 0 && dims.height > 0);
    floatt* ptr = allocator (dims);
    logTrace ("ptr = %p dims = %s", ptr, std::to_string(dims).c_str());
    return {ptr, dims};
  }

  template<typename Allocator>
  oap::Memory newMemoryWithValues (const MemoryDims& dims, floatt value, Allocator&& allocator)
  {
    logAssert (dims.width > 0 && dims.height > 0);
    floatt* ptr = allocator (dims, value);
    logTrace ("ptr = %p dims = %s", ptr, std::to_string(dims).c_str());
    return {ptr, dims};
  }

  template<typename Allocator>
  oap::Memory newMemoryCopy (const oap::Memory& src, Allocator&& allocator)
  {
    floatt* ptr = allocator (src.ptr, src.dims);
    logTrace ("ptr = %p dims = %s", ptr, std::to_string(src.dims).c_str());
    return {ptr, src.dims};
  }

  template<typename Allocator>
  oap::Memory newMemoryCopyMem (const oap::Memory& src, uintt width, uintt height, Allocator&& allocator)
  {
    debugAssert (width * height == src.dims.width * src.dims.height && width * height != 0);
    floatt* ptr = allocator (src.ptr, src.dims, {width, height});
    oap::MemoryDims dims = {width, height};
    logTrace ("ptr = %p dims = %s", ptr, std::to_string(dims).c_str());
    return {ptr, dims};
  }

  template<typename Allocator>
  oap::Memory reuseMemory (const oap::Memory& src, uintt width, uintt height, Allocator&& allocator)
  {
    floatt* ptr = allocator (src.ptr, src.dims, {width, height});
    oap::MemoryDims dims = {width, height};
    logTrace ("ptr = %p dims = %s", ptr, std::to_string(dims).c_str());
    return {ptr, dims};
  }

  template<typename Deallocator>
  void deleteMemory (const oap::Memory& mem, Deallocator&& deallocator)
  {
    return deallocator (mem);
  }

  template<typename GetD>
  oap::MemoryDims getDims (const oap::Memory& mem, GetD&& getD)
  {
    return getD (mem.ptr, mem.dims);
  }

  template<typename GetRM>
  floatt* getRawMemory (const oap::Memory& mem, GetRM&& getRM)
  {
    return getRM (mem.ptr, mem.dims);
  }

  template<typename Memcpy>
  void copy (oap::Memory& dst, const oap::MemoryLoc& dstLoc, const oap::Memory& src, const oap::MemoryRegion& srcReg, Memcpy&& memcpy)
  {
    logAssert (dst.dims.width >= srcReg.dims.width + dstLoc.x);
    logAssert (dst.dims.height >= srcReg.dims.height + dstLoc.y);
  
    std::vector<floatt*> dstPtrs;
    std::vector<const floatt*> srcPtrs;

    auto dstReg = common::convertToRegion (dst.dims, dstLoc);

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
  void copy (oap::Memory& dst, const oap::Memory& src, Memcpy&& memcpy)
  {
    const auto& srcReg = common::convertToRegion (src.dims, common::OAP_NONE_LOCATION());
    copy<Memcpy> (dst, {0, 0}, src, srcReg, memcpy);
  }

  /**
   *  \brief Cheks if memory can be copied as linear continous block of memory
   */
  inline bool isLinearMemory (const oap::MemoryDims& dstDims, const oap::MemoryLoc& dstLoc, const oap::MemoryDims& srcDims, const oap::MemoryRegion& srcReg)
  {
    bool islinear = (srcReg.dims.width == 1 && (dstDims.height == 1 || dstDims.width == 1)) || (srcReg.dims.height == 1);
    return islinear;
  }

  inline bool isBlockMemory (const oap::MemoryDims& dstDims, const oap::MemoryLoc& dstLoc, const oap::MemoryDims& srcDims, const oap::MemoryRegion& srcReg)
  {
    return !isLinearMemory (dstDims, dstLoc, srcDims, srcReg);
  }

  template<typename Memcpy>
  void copyBlock (floatt* dst, const oap::MemoryDims& dstDims, const oap::MemoryLoc& dstLoc, const floatt* src, const oap::MemoryDims& srcDims, const oap::MemoryRegion& srcReg, Memcpy&& memcpy)
  {
    logTrace ("%s %p %s %s %p %s %s", __FUNCTION__, dst, std::to_string(dstDims).c_str(), std::to_string(dstLoc).c_str(), src, std::to_string(srcDims).c_str(), std::to_string(srcReg).c_str());

    logAssert (dstDims.width >= dstLoc.x + srcReg.dims.width);
    logAssert (dstDims.height >= dstLoc.y + srcReg.dims.height);

    logAssert (srcDims.width >= srcReg.loc.x + srcReg.dims.width);
    logAssert (srcDims.height >= srcReg.loc.y + srcReg.dims.height);

    std::vector<floatt*> dstPtrs;
    std::vector<const floatt*> srcPtrs;

    auto dstReg = common::convertToRegion (dstDims, dstLoc);

    utils::getPtrs (dstPtrs, dst, dstDims, dstReg);
    utils::getPtrs (srcPtrs, src, srcDims, srcReg);

    for (size_t idx = 0; idx < srcPtrs.size (); ++idx) {
      const floatt* srcPtr = srcPtrs[idx];
      floatt* dstPtr = dstPtrs[idx];
      memcpy (dstPtr, srcPtr, srcReg.dims.width * sizeof (floatt));
    }
  }

  template<typename Memcpy>
  void copyLinear (floatt* dst, const oap::MemoryDims& dstDims, const oap::MemoryLoc& dstLoc, const floatt* src, const oap::MemoryDims& srcDims, const oap::MemoryRegion& srcReg, Memcpy&& memcpy)
  {
    logTrace ("%s %p %s %s %p %s %s", __FUNCTION__, dst, std::to_string(dstDims).c_str(), std::to_string(dstLoc).c_str(), src, std::to_string(srcDims).c_str(), std::to_string(srcReg).c_str());
    logAssert ((srcReg.dims.width == 1 && (dstDims.height == 1 || dstDims.width == 1)) || (srcReg.dims.height == 1));
    uintt srcLen = srcReg.dims.width * srcReg.dims.height;
    uintt dstLen = dstDims.width * dstDims.height;

    uint dstPos = dstLoc.x + dstLoc.y * dstDims.width;
    uint srcPos = srcReg.loc.x + srcReg.loc.y * srcDims.width;

    logAssert (dstPos + srcLen <= dstLen);

    memcpy (&dst[dstPos], &src[srcPos], srcLen * sizeof (floatt));
  }

  template<typename Memcpy>
  void copy (floatt* dst, const oap::MemoryDims& dstDims, const oap::MemoryLoc& dstLoc, const floatt* src, const oap::MemoryDims& srcDims, const oap::MemoryRegion& srcReg, Memcpy&& memcpy)
  {
    if (isBlockMemory (dstDims, dstLoc, srcDims, srcReg))
    {
      copyBlock (dst, dstDims, dstLoc, src, srcDims, srcReg, memcpy);
    }
    else
    {
      copyLinear (dst, dstDims, dstLoc, src, srcDims, srcReg, memcpy);
    }
  }

  template<typename Memcpy>
  void copy (floatt* dst, const oap::MemoryDims& dstDims, const oap::MemoryLoc& dstLoc, const floatt* src, const oap::MemoryDims& srcDims, const oap::MemoryRegion* srcReg, Memcpy&& memcpy)
  {
    copy<Memcpy> (dst, dstDims, dstLoc, src, srcDims, srcReg == nullptr ? common::OAP_NONE_REGION() : *srcReg, memcpy);
  }


  template<typename MemoryVec, typename Allocator, typename Memcpy>
  oap::Memory newMemory_bulk (const MemoryVec& vec, const oap::DataDirection& dd, Allocator&& alloc, Memcpy&& memcpy)
  {
    logAssert (vec.size() > 0);

    oap::MemoryDims dim = {0, 0};

    switch (dd) {
      case DataDirection::VERTICAL:
        dim.width = vec[0].dims.width;
        for (size_t idx = 0; idx < vec.size(); ++idx)
        {
          dim.height = dim.height + vec[idx].dims.height;
          logAssert (idx == 0 || vec[idx].dims.width == vec[idx - 1].dims.width);
        }
        break;
      case DataDirection::HORIZONTAL:
        dim.height = vec[0].dims.height;
        for (size_t idx = 0; idx < vec.size(); ++idx)
        {
          dim.width = dim.width + vec[idx].dims.width;
          logAssert (idx == 0 || vec[idx].dims.height == vec[idx - 1].dims.height);
        }
        break;
    };

    oap::Memory memory = alloc (dim);
    oap::MemoryLoc loc = {0, 0};

    switch (dd) {
      case DataDirection::VERTICAL:
        for (size_t idx = 0; idx < vec.size(); ++idx)
        {
          oap::MemoryDims dim = vec[idx].dims;
          floatt* array = vec[idx].ptr;
          oap::generic::copy (memory, loc, {array, dim}, {{0, 0}, dim}, memcpy);
          loc.y += dim.height;
        }
        break;
      case DataDirection::HORIZONTAL:
        for (size_t idx = 0; idx < vec.size(); ++idx)
        {
          oap::MemoryDims dim = vec[idx].dims;
          floatt* array = vec[idx].ptr;
          oap::generic::copy (memory, loc, {array, dim}, {{0, 0}, dim}, memcpy);
          loc.x += dim.width;
        }
        break;
    }
    return memory;
  }
}
}

#endif
