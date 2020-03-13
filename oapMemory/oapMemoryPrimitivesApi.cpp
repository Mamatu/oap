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

#include "oapMemoryPrimitivesApi.h"

namespace std
{
  std::string to_string (const oap::Memory& memory)
  {
    std::stringstream sstream;
    sstream << "(ptr: " << memory.ptr << " width: " << memory.dims.width << ", height: " << memory.dims.height << ")";
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
