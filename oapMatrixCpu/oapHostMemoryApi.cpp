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

#include <string.h>

#include "MatrixParser.h"
#include "ReferencesCounter.h"

#include "oapHostMemoryApi.h"

#include "oapMemoryList.h"
#include "oapMemoryPrimitives.h"
#include "oapMemory_GenericApi.h"
#include "oapMemoryManager.h"

#define ReIsNotNULL(m) m->reValues != nullptr
#define ImIsNotNULL(m) m->imValues != nullptr

#ifdef DEBUG
/*
std::ostream& operator<<(std::ostream& output, const math::Matrix*& matrix)
{
  return output << matrix << ", [" << gColumns (matrix) << ", " << gRows (matrix)
         << "]";
}
*/
#define NEW_MATRIX() new math::Matrix();

#define DELETE_MATRIX(matrix) delete matrix;

#else

#define NEW_MATRIX() new math::Matrix();

#define DELETE_MATRIX(matrix) delete matrix;

#endif

inline void fillWithValue (floatt* values, floatt value, uintt length)
{
  math::Memset (values, value, length);
}

inline void fillRePart(math::Matrix* output, floatt value)
{
  fillWithValue (output->re.ptr, value, gColumns (output) * gRows (output));
}

inline void fillImPart(math::Matrix* output, floatt value)
{
  fillWithValue (output->im.ptr, value, gColumns (output) * gRows (output));
}

namespace oap
{
namespace host
{

namespace
{

MemoryList g_memoryList ("HOST");

floatt* allocateBuffer (size_t length)
{
  floatt* buffer = new floatt [length];
  g_memoryList.add (buffer, length);
  return buffer;
}

void deallocateBuffer (floatt* const buffer)
{
  g_memoryList.remove (buffer);
  delete[] buffer;
}

oap::MemoryManagement<floatt*, decltype(allocateBuffer), decltype(deallocateBuffer), nullptr> g_memoryMng (allocateBuffer, deallocateBuffer);

floatt* allocateMem (const oap::MemoryDims& dims)
{
  floatt* raw = g_memoryMng.allocate (dims.width * dims.height);

  return raw;
}

void deallocateMem (const oap::Memory& memory)
{
  g_memoryMng.deallocate (memory.ptr);
}

}

oap::Memory NewMemory (const oap::MemoryDims& dims)
{
  return oap::generic::newMemory (dims, allocateMem);
}

oap::Memory NewMemoryWithValues (const MemoryDims& dims, floatt value)
{
  return oap::generic::newMemoryWithValues (dims, value, [](const MemoryDims& dims, floatt value)
  {
    oap::Memory memory = NewMemory (dims);
    math::Memset (memory.ptr, value, dims.width * dims.height);
    return memory.ptr;
  });
}

oap::Memory NewMemoryCopy (const oap::Memory& src)
{
  return oap::generic::newMemoryCopy (src, [](floatt* const src, const MemoryDims& dims)
  {
    oap::Memory memory = NewMemory (dims);
    const oap::Memory srcMem = {src, dims};
    oap::host::CopyHostToHost (memory, srcMem);
    return memory.ptr;
  });
}

oap::Memory NewMemoryCopyMem (const oap::Memory& src, uintt width, uintt height)
{
  return oap::generic::newMemoryCopyMem (src, width, height, [](floatt* const src, const oap::MemoryDims& oldDims, const oap::MemoryDims& newDims)
  {
    oap::Memory memory = NewMemory (newDims);
    oap::host::CopyHostToHost (memory, {src, oldDims});
    return memory.ptr;
  });
}

oap::Memory ReuseMemory (const oap::Memory& src, uintt width, uintt height)
{
  return oap::generic::reuseMemory (src, width, height, [](floatt* const src, const oap::MemoryDims& oldDims, const oap::MemoryDims& newDims)
  {
    return g_memoryMng.reuse (src);
  });
}

void DeleteMemory (const oap::Memory& mem)
{
  return oap::generic::deleteMemory (mem, [](const oap::Memory& mem)
  {
    deallocateMem (mem);
  });
}

oap::MemoryDims GetDims (const oap::Memory& mem)
{
  return mem.dims;
}

floatt* GetRawMemory (const oap::Memory& mem)
{
  return mem.ptr;
}

void CopyHostToHost (oap::Memory& dst, const oap::MemoryLoc& dstLoc, const oap::Memory& src, const oap::MemoryRegion& srcReg)
{
  oap::generic::copy (dst, dstLoc, src, srcReg, memcpy);
}

void CopyHostToHost (oap::Memory& dst, const oap::Memory& src)
{
  oap::generic::copy (dst, src, memcpy);
}

}
}
