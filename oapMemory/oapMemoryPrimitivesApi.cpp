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

#include "oapMemoryPrimitivesApi.h"
#include "oapMemoryUtils.h"

namespace std
{
#if 0
  std::string to_string (const oap::Memory& memory)
  {
    std::stringstream sstream;
    sstream << "(ptr: " << memory.ptr << " width: " << memory.dims.width << ", height: " << memory.dims.height << ")";
    return sstream.str();
  }
#endif
  std::string to_string (const oap::Memory& memory)
  {
    std::stringstream sstream;

    auto setValue = [&sstream](floatt v)
    {
      sstream << v << " ";
    };

    auto endl = [&sstream]()
    {
      sstream << std::endl;
    };

    iterate (setValue, memory, endl);
    return sstream.str();
  }

  std::string to_string (const oap::MemoryDims& dims)
  {
    std::stringstream sstream;
    sstream << "(width: " << dims.width << ", height: " << dims.height << ")";
    return sstream.str();
  }

  std::string to_string (const oap::MemoryLoc& loc)
  {
    std::stringstream sstream;
    sstream << "(x: " << loc.x << ", y: " << loc.y << ")";
    return sstream.str();
  }

  std::string to_string (const oap::MemoryRegion& region)
  {
    std::stringstream sstream;
    sstream << to_string (region.loc);
    sstream << " ";
    sstream << to_string (region.dims);
    return sstream.str();
  }
}

bool operator== (const oap::MemoryRegion& reg1, const oap::MemoryRegion& reg2)
{
  return oap::common::CompareMemoryRegion (reg1, reg2);
}

bool operator!= (const oap::MemoryRegion& reg1, const oap::MemoryRegion& reg2)
{
  return !(reg1 == reg2);
}

bool operator== (const oap::MemoryDims& dim1, const oap::MemoryDims& dim2)
{
  return dim1.width == dim2.width && dim1.height == dim2.height;
}

bool operator!= (const oap::MemoryDims& dim1, const oap::MemoryDims& dim2)
{
  return !(dim1 == dim2);
}

bool operator< (const oap::MemoryRegion& reg1, const oap::MemoryRegion& reg2)
{
  bool b1 = oap::utils::lessByX (reg1, reg2);
  if (!b1)
  {
    bool b2 = oap::utils::lessByX (reg2, reg1);
    if (!b2)
    {
      return oap::utils::lessByY (reg1, reg2);
    }
  }
  return b1;
}
