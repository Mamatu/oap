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

#include <string.h>

#include "MatrixParser.hpp"
#include "ReferencesCounter.hpp"

#include "oapAssertion.hpp"
#include "oapHostMemoryApi.hpp"

#include "oapMemoryList.hpp"
#include "oapMemory.hpp"
#include "oapMemory_CommonApi.hpp"
#include "oapMemory_GenericApi.hpp"
#include "oapMemoryCounter.hpp"

#define ReIsNotNULL(m) m->reValues != nullptr
#define ImIsNotNULL(m) m->imValues != nullptr

#ifdef DEBUG
/*
std::ostream& operator<<(std::ostream& output, const math::ComplexMatrix*& matrix)
{
  return output << matrix << ", [" << gColumns (matrix) << ", " << gRows (matrix)
         << "]";
}
*/
#define NEW_MATRIX() new math::ComplexMatrix();

#define DELETE_MATRIX(matrix) delete matrix;

#else

#define NEW_MATRIX() new math::ComplexMatrix();

#define DELETE_MATRIX(matrix) delete matrix;

#endif

inline void fillWithValue (floatt* values, floatt value, uintt length)
{
  math::Memset (values, value, length);
}

inline void fillRePart(math::ComplexMatrix* output, floatt value)
{
  fillWithValue (output->re.mem.ptr, value, gColumns (output) * gRows (output));
}

inline void fillImPart(math::ComplexMatrix* output, floatt value)
{
  fillWithValue (output->im.mem.ptr, value, gColumns (output) * gRows (output));
}

namespace oap
{
namespace host
{

namespace
{

MemoryList g_memoryList ("MEMORY_HOST");
MemoryCounter g_memoryCounter;

floatt* allocateMem (const oap::MemoryDim& dims)
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

oap::Memory NewMemory (const oap::MemoryDim& dims)
{
  return oap::generic::newMemory (dims, allocateMem, [](floatt* ptr)
      {
        g_memoryCounter.increase (ptr);
      });
}

oap::Memory NewMemoryWithValues (const MemoryDim& dims, floatt value)
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

oap::MemoryDim GetDims (const oap::Memory& mem)
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

void CopyHostToHost (oap::Memory& dst, const oap::Memory& src, const oap::MemoryRegion& srcReg)
{
  oap::generic::copy (dst, {0, 0}, src, srcReg, memcpy);
}

void CopyHostToHost (oap::Memory& dst, const oap::Memory& src)
{
  oap::generic::copy (dst, src, memcpy);
}

void CopyHostToHostBuffer (floatt* buffer, uintt length, const oap::Memory& src, const oap::MemoryRegion& srcReg)
{
  oap::generic::copyMemoryRegionToBuffer (buffer, length, src.ptr, src.dims, srcReg, memcpy, memmove);
}

void CopyHostBufferToHost (oap::Memory& dst, const oap::MemoryRegion& dstReg, const floatt* buffer, uintt length)
{
  oap::generic::copyBufferToMemoryRegion (dst.ptr, dst.dims, dstReg, buffer, length, memcpy, memmove);
}

floatt GetValue(const oap::Memory& memory, const oap::MemoryRegion& reg, uintt x, uintt y)
{
  const uintt idx = oap::common::GetIdx (memory, reg, x, y);
  return memory.ptr[idx];
}

}
}
