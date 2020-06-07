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

#ifndef OAP_MEMORY_UTILS_H
#define	OAP_MEMORY_UTILS_H

#include <algorithm>
#include "oapMemoryPrimitives.h"

namespace oap {
namespace utils {

  inline bool lessByX (const oap::MemoryRegion& mr1, const oap::MemoryRegion& mr2)
  {
    const bool isequal = mr1.loc.x == mr2.loc.x;
    if (isequal)
    {
      return mr1.dims.width < mr2.dims.width;
    }
    return mr1.loc.x < mr2.loc.x;
  }

  inline bool lessByY (const oap::MemoryRegion& mr1, const oap::MemoryRegion& mr2)
  {
    const bool isequal = mr1.loc.y == mr2.loc.y;
    if (isequal)
    {
      return mr1.dims.height < mr2.dims.height;
    }
    return mr1.loc.y < mr2.loc.y;
  }

  template<typename Container>
  void sortByX (Container& container)
  {
    std::sort (container.begin(), container.end(), lessByX);
  }

  template<typename Container>
  void sortByY (Container& container)
  {
    std::sort (container.begin(), container.end(), lessByY);
  }
}
}

#endif	/* THREADSMAPPER_H */
