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
#include "oapMemoryCounter.h"

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

MemoryList g_memoryList ("MEMORY_HOST");
MemoryCounter g_memoryCounter;

floatt* allocateMem (const oap::MemoryDims& dims)
{
  const uintt length = dims.width * dims.height;
  floatt* buffer = new floatt [length];
  g_memoryList.add (buffer, length);
  return buffer;
}

void deallocateMem (const oap::Memory& memory)
{
  delete[] memory.ptr;
  g_memoryList.remove (memory.ptr);
}

}

oap::Memory NewMemory (const oap::MemoryDims& dims)
{
  return oap::generic::newMemory (dims, allocateMem, [](floatt* ptr)
      {
        g_memoryCounter.increase (ptr);
      });
}

oap::Memory NewMemoryWithValues (const MemoryDims& dims, floatt value)
{
  oap::Memory memory = NewMemory (dims);
  math::Memset (memory.ptr, value, dims.width * dims.height);
  return memory;
}

oap::Memory NewMemoryCopy (const oap::Memory& src)
{
  oap::Memory memory = NewMemory (src.dims);
  oap::host::CopyHostToHost (memory, src);
  return memory;
}

oap::Memory NewMemoryCopyMem (const oap::Memory& src, uintt width, uintt height)
{
  oap::Memory memory = NewMemory ({width, height});
  oap::generic::copyMemory (memory, src, memcpy);
  return memory;
}

oap::Memory ReuseMemory (const oap::Memory& src, uintt width, uintt height)
{
  return oap::generic::reuseMemory (src, width, height, [](floatt* ptr)
      {
        g_memoryCounter.increase (ptr);
      });
}

oap::Memory ReuseMemory (const oap::Memory& src)
{
  return oap::generic::reuseMemory (src, src.dims.width, src.dims.height, [](floatt* ptr)
      {
        g_memoryCounter.increase (ptr);
      });
}

void DeleteMemory (const oap::Memory& mem)
{
  oap::generic::deleteMemory (mem, deallocateMem, [](floatt* ptr)
      {
        return g_memoryCounter.decrease (ptr);
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
