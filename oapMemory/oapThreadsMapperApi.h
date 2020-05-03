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

#ifndef OAP_THREADS_MAPPER_API_H
#define	OAP_THREADS_MAPPER_API_H

#include "oapMemoryPrimitives.h"
#include "oapMemoryUtils.h"

namespace oap {

namespace threads {

using Pair = std::pair<uintt, uintt>;

template<typename Pairs, typename MemoryRegions>
void getThreads (Pairs& pairs, const MemoryRegions& regions, bool canOverlap = false)
{
  using RegionsPair = std::pair<oap::MemoryRegion, oap::MemoryRegion>;

  std::map<RegionsPair, bool> overlapMatrix;

  struct RegionsCompObj
  {
    bool operator()(const RegionsPair& pair1, const RegionsPair& pair2)
    {
      const bool x12 = utils::lessByX (pair1.first, pair2.first);
      if (x12)
      {
        return true;
      }
      const bool x21 = utils::lessByX (pair2.first, pair1.first);
      if (!x12 && !x21)
      {
        return false;
      }
      if (utils::lessByX (pair1.first, pair2.first))
      {
        return true;
      }

      if (utils::lessByY (pair1.first, pair2.first))
      {
        return true;
      }
    }
  };

  {
    MemoryRegions regions_x = regions;
    oap::utils::sortByX (regions_x);
    for (size_t idx = 1; idx < regions_x.size(); ++idx)
    {
      const auto& prev = regions_x[idx - 1];
      const auto& current = regions_x[idx];
      if (prev.loc.x + prev.dims.width >= current.loc.x)
      {
        if (prev.loc.x + prev.dims.width >= current.loc.x + current.dims.width)
        {
          uintt x = prev.loc.x;
          uintt width = prev.dims.width;

          pairs.push_back (std::make_pair (x, width));
        }
        else
        {
          uintt x = prev.loc.x;
          uintt width = current.loc.x + current.dims.width - prev.loc.x;

          pairs.push_back (std::make_pair (x, width));
        }
      }
      else
      {
        pairs.push_back (std::make_pair (prev.loc.x, prev.dims.width));
        if (idx == regions_x.size() - 1)
        {
          pairs.push_back (std::make_pair (current.loc.x, current.dims.width));
        }
      }
    }
  }

  {
    MemoryRegions regions_y = regions;
    oap::utils::sortByY (regions_y);
  }
}

template<typename MemoryRegions>
uintt getXThreads (const MemoryRegions& regions)
{
  
}

template<typename MemoryRegions>
uintt getYThreads (const MemoryRegions& regions)
{

}

void createMove (std::vector<uintt>& move, const oap::Memory& memory, const std::vector<oap::MemoryRegion>& regions)
{
  
}

}
}

#endif	/* THREADSMAPPER_H */
