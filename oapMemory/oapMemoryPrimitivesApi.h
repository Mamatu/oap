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

#ifndef OAP_MEMORY_PRIMITIVES_API_H
#define OAP_MEMORY_PRIMITIVES_API_H

#include <string>
#include <sstream>
#include <functional>
#include "oapMemoryPrimitives.h"
#include "oapMemory_CommonApi.h"

namespace oap
{
namespace
{
  auto defaultEndl = [](){};
}

  template<typename SetValue, typename Endl = decltype(defaultEndl)>
  void iterate (SetValue&& setValue, const oap::Memory& memory, Endl&& endl = std::forward<Endl>(defaultEndl))
  {
    for (uintt y = 0; y < memory.dims.height; ++y)
    {
      for (uintt x = 0; x < memory.dims.width; ++x)
      {
        setValue (memory.ptr[x + y * memory.dims.width]);
      }
      endl ();
    }
  }

  inline void to_vector (std::vector<floatt>& vector, const oap::Memory& memory)
  {
    iterate ([&vector](floatt value) { vector.push_back (value); }, memory);
  }
}

namespace std
{
  std::string to_string (const oap::Memory& memory);

  std::string to_string (const oap::MemoryDim& dims);

  std::string to_string (const oap::MemoryLoc& loc);

  std::string to_string (const oap::MemoryRegion& region);
}

bool operator== (const oap::MemoryRegion& reg1, const oap::MemoryRegion& reg2);

bool operator!= (const oap::MemoryRegion& reg1, const oap::MemoryRegion& reg2);

bool operator== (const oap::MemoryDim& dim1, const oap::MemoryDim& dim2);

bool operator!= (const oap::MemoryDim& dim1, const oap::MemoryDim& dim2);

bool operator< (const oap::MemoryRegion& reg1, const oap::MemoryRegion& reg2);

#endif
