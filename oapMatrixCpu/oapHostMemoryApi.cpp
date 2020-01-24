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
#include "oapGenericMemoryApi.h"
#include "oapMemoryManager.h"

#define ReIsNotNULL(m) m->reValues != nullptr
#define ImIsNotNULL(m) m->imValues != nullptr

#ifdef DEBUG
/*
std::ostream& operator<<(std::ostream& output, const math::Matrix*& matrix)
{
  return output << matrix << ", [" << matrix->columns << ", " << matrix->rows
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
  fillWithValue (output->reValues, value, output->columns * output->rows);
}

inline void fillImPart(math::Matrix* output, floatt value)
{
  fillWithValue (output->imValues, value, output->columns * output->rows);
}

namespace oap
{
namespace host
{

namespace
{
floatt* allocateBuffer (size_t length)
{
  return new floatt [length];
}

void deallocateBuffer (const floatt* buffer)
{
  delete[] buffer;
}

oap::MemoryManagement<floatt*, decltype(allocateBuffer), decltype(deallocateBuffer), nullptr> g_memoryMng (allocateBuffer, deallocateBuffer);
MemoryList g_memoryList ("HOST");

oap::Memory* allocateMemStructure (const oap::MemoryDims& dims, floatt* ptr)
{
  oap::Memory* memory = new oap::Memory;
  memory->dims = dims;
  memory->ptr = ptr;
  return memory;
}

oap::Memory* allocateMem (const oap::MemoryDims& dims)
{
  floatt* raw = g_memoryMng.allocate (dims.width * dims.height);

  oap::Memory* memory = allocateMemStructure (dims, raw);
  g_memoryList.add (memory, *memory);

  return memory;
}

void deallocateMem (const oap::Memory* memory)
{
  g_memoryList.remove (memory);
  g_memoryMng.deallocate (memory->ptr);
  delete memory;
}

}

oap::Memory* NewMemory (const oap::MemoryDims& dims)
{
  return oap::generic::newMemory (dims, allocateMem);
}

oap::Memory* NewMemoryWithValues (const MemoryDims& dims, floatt value)
{
  return oap::generic::newMemoryWithValues (dims, value, [](const MemoryDims& dims, floatt value)
  {
    oap::Memory* memory = NewMemory (dims);
    math::Memset (memory->ptr, value, dims.width * dims.height);
    return memory;
  });
}

oap::Memory* NewMemoryCopy (const oap::Memory* src)
{
  return oap::generic::newMemoryCopy (src, [](const oap::Memory* src)
  {
    oap::Memory* memory = NewMemory (src->dims);
    oap::host::Copy (memory, src);
    return memory;
  });
}

oap::Memory* NewMemoryCopyMem (const oap::Memory* src, uintt width, uintt height)
{
  return oap::generic::newMemoryCopyMem (src, width, height, [](const oap::Memory* src, uintt width, uintt height)
  {
    logAssert (src->dims.height * src->dims.width == width * height);

    oap::Memory* memory = NewMemory (src->dims);
    oap::host::Copy (memory, src);
    return memory;
  });
}

oap::Memory* ReuseMemory (const oap::Memory* src, uintt width, uintt height)
{
  return oap::generic::reuseMemory (src, width, height, [](const oap::Memory* src, uintt width, uintt height)
  {
    logAssert (src->dims.height * src->dims.width == width * height);
    floatt* ptr = GetRawMemory (src);
    oap::Memory* out = allocateMemStructure ({width, height}, g_memoryMng.reuse (ptr));
    return out;
  });
}

void DeleteMemory (const oap::Memory* mem)
{
  return oap::generic::deleteMemory (mem, [](const oap::Memory* mem)
  {
    deallocateMem (mem);
  });
}

oap::MemoryDims GetDims (const oap::Memory* mem)
{
  return oap::generic::getDims (mem, [](const oap::Memory* mem)
  {
    return mem->dims;
  });
}

floatt* GetRawMemory (const oap::Memory* mem)
{
  return oap::generic::getRawMemory (mem, [](const oap::Memory* mem)
  {
    return mem->ptr;
  });
}

void Copy (oap::Memory* dst, const oap::MemoryLoc& dstLoc, const oap::Memory* src, const oap::MemoryRegion& srcReg)
{
  oap::generic::copy (dst, dstLoc, src, srcReg, memcpy);
}

void Copy (oap::Memory* dst, const oap::Memory* src)
{
  oap::generic::copy (dst, src, memcpy);
}

}
}
