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
MemoryList gMemoryList ("HOST");

oap::Memory* allocateMem (const oap::MemoryDims& dims)
{
  floatt* raw = new floatt [dims.width * dims.height];
  oap::Memory* memory = new oap::Memory;
  memory->ptr = raw;
  memory->dims = dims;

  gMemoryList.add (memory, *memory);

  return memory;
}

void deallocateMem (const oap::Memory* memory)
{
  gMemoryList.remove (memory);
  delete[] memory->ptr;
  delete memory;
}

}

oap::Memory* NewMemory (const oap::MemoryDims& dims)
{
  return allocateMem (dims); 
}

oap::Memory* NewMemoryWithValues (const MemoryDims& dims, floatt value)
{
  oap::Memory* memory = NewMemory (dims);
  math::Memset (memory->ptr, value, dims.width * dims.height);
  return memory;
}

void DeleteMemory (const oap::Memory* mem)
{
  deallocateMem (mem);
}

oap::MemoryDims GetDims (const oap::Memory* mem)
{
  return mem->dims;
}

void CopyMemoryRegion (oap::Memory* dst, const oap::MemoryLoc& dstLoc, const oap::Memory* src, const oap::MemoryRegion& srcReg)
{
  oap::generic::copyMemoryRegion (dst, dstLoc, src, srcReg, memcpy);
}

}
}
