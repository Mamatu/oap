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

#ifndef OAP_HOST_MEMORY_API_HPP
#define OAP_HOST_MEMORY_API_HPP

#include "ByteBuffer.hpp"
#include "Logger.hpp"

#include "MatrixPrinter.hpp"
#include "oapMemory.hpp"
#include "oapMemory_GenericApi.hpp"

namespace oap
{
namespace host
{

oap::Memory NewMemory (const MemoryDim& dims);
oap::Memory NewMemoryWithValues (const MemoryDim& dims, floatt value);
oap::Memory NewMemoryCopy (const oap::Memory& src);
oap::Memory NewMemoryCopyMem (const oap::Memory& src, uintt width, uintt height);
oap::Memory ReuseMemory (const oap::Memory& src, uintt width, uintt height);
oap::Memory ReuseMemory (const oap::Memory& src);

void DeleteMemory (const oap::Memory& mem);

oap::MemoryDim GetDims (const oap::Memory& mem);
floatt* GetRawMemory (const oap::Memory& mem);

void CopyHostToHost (oap::Memory& dst, const oap::MemoryLoc& dstLoc, const oap::Memory& src, const oap::MemoryRegion& srcReg);
void CopyHostToHost (oap::Memory& dst, const oap::Memory& src, const oap::MemoryRegion& srcReg);
void CopyHostToHost (oap::Memory& dst, const oap::Memory& src);

template<typename MemoryVec>
oap::Memory NewMemoryBulk (const MemoryVec& vec, const oap::DataDirection& dd)
{
  return oap::generic::newMemory_bulk (vec, dd, oap::host::NewMemory, memcpy);
}

template<typename MemoryVec, typename MemLocDimCallback>
oap::Memory NewMemoryBulk (const MemoryVec& vec, const oap::DataDirection& dd, MemLocDimCallback&& mldCallback)
{
  return oap::generic::newMemory_bulk (vec, dd, oap::host::NewMemory, memcpy, mldCallback);
}

void CopyHostToHostBuffer (floatt* buffer, uintt length, const oap::Memory& src, const oap::MemoryRegion& srcReg);
void CopyHostBufferToHost (oap::Memory& src, const oap::MemoryRegion& srcReg, const floatt* buffer, uintt length);

floatt GetValue(const oap::Memory& memory, const oap::MemoryRegion& reg, uintt x, uintt y);

}
}

#endif
