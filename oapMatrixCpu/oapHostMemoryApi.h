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

#ifndef OAP_HOST_MEMORY_API_H
#define OAP_HOST_MEMORY_API_H

#include "ByteBuffer.h"
#include "Logger.h"

#include "MatrixPrinter.h"
#include "oapMemoryPrimitives.h"
#include "oapMemory_GenericApi.h"

namespace oap
{
namespace host
{

oap::Memory NewMemory (const MemoryDims& dims);
oap::Memory NewMemoryWithValues (const MemoryDims& dims, floatt value);
oap::Memory NewMemoryCopy (const oap::Memory& src);
oap::Memory NewMemoryCopyMem (const oap::Memory& src, uintt width, uintt height);
oap::Memory ReuseMemory (const oap::Memory& src, uintt width, uintt height);

void DeleteMemory (const oap::Memory& mem);

oap::MemoryDims GetDims (const oap::Memory& mem);
floatt* GetRawMemory (const oap::Memory& mem);

void CopyHostToHost (oap::Memory& dst, const oap::MemoryLoc& dstLoc, const oap::Memory& src, const oap::MemoryRegion& srcReg);
void CopyHostToHost (oap::Memory& dst, const oap::Memory& src);

template<typename MemoryVec>
oap::Memory NewMemoryBulk (const MemoryVec& vec, const oap::DataDirection& dd)
{
  return oap::generic::newMemory_bulk (vec, dd, oap::host::NewMemory, memcpy);
}

}
}

#endif
