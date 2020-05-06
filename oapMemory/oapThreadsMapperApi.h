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

#include "oapMemoryPrimitivesApi.h"
#include <set>

namespace oap {

namespace threads {

using Section = std::pair<uintt, uintt>;
using PairMR = std::pair<oap::MemoryRegion, oap::MemoryRegion>;

namespace
{
inline PairMR makePair (const oap::MemoryRegion& region1, const oap::MemoryRegion& region2)
{
  auto pair = region1 < region2 ? std::make_pair(region1, region2) : std::make_pair(region2, region1);
  return pair;
}

template<typename Container>
void insertRegion (Container& container, const oap::MemoryRegion& region1, const oap::MemoryRegion& region2)
{
  container.insert (makePair (region1, region2));
}

template<typename Container>
bool hasRegions (Container& container, const oap::MemoryRegion& region1, const oap::MemoryRegion& region2)
{
  auto pair = makePair (region1, region2);
  auto it = container.find (pair);
  return (it != container.end());
}

struct RegionsCompObj
{
  bool operator()(const PairMR& pair1, const PairMR& pair2)
  {
    const bool b1 = utils::lessByX (pair1.first, pair2.first);
    if (b1)
    {
      return true;
    }
    return utils::lessByX (pair1.second, pair2.second);
  }
};

template<typename Sections, typename MemoryRegions, typename Sort, typename GetLoc, typename GetDim>
void getThreadsSections (Sections& sections, const MemoryRegions& _regions, Sort&& sort, GetLoc&& getLoc, GetDim&& getDim)
{
  MemoryRegions regions = _regions;
  sort (regions);

  auto getSection = [&regions, &getLoc, &getDim](uintt index)
  {
    const auto& prev = regions[index];
    uintt ploc = getLoc (prev);
    uintt pdim = getDim (prev);
    return std::make_pair (ploc, pdim);
  };

  auto section = getSection (0);
  sections = {section};

  for (size_t idx = 1; idx < regions.size(); ++idx)
  {
    const auto& current = regions[idx];

    const auto& prev = sections.back();
    uintt ploc = prev.first;
    uintt pdim = prev.second;
    uintt cloc = getLoc (current);
    uintt cdim = getDim (current);

    if (ploc + pdim >= cloc)
    {
      if (ploc + pdim < cloc + cdim)
      {
        uintt x = ploc;
        uintt width = cloc + cdim - ploc;

        sections.back().second = width;
      }
    }
    else
    {
      sections.push_back (std::make_pair (cloc, cdim));
    }
  }
}

template<typename Sections>
uintt getThreadsSectionsSum (Sections& sections)
{
  uintt count = 0;
  for (const auto& pair : sections)
  {
    count += pair.second;
  }
  return count;
}

template<typename Sections, typename MemoryRegions, typename Sort, typename GetLoc, typename GetDim>
uintt getThreadsCount (Sections& sections, const MemoryRegions& _regions, Sort&& sort, GetLoc&& getLoc, GetDim&& getDim)
{
  getThreadsSections (sections, _regions, sort, getLoc, getDim);
  return getThreadsSectionsSum (sections);
}

}

template<typename MemoryRegions>
uintt getXThreads (const MemoryRegions& regions)
{
  using MR = oap::MemoryRegion;
  std::vector<Section> sections;
  return getThreadsCount (sections, regions, [](MemoryRegions& rs) { utils::sortByX (rs); }, [](const MR& r){ return r.loc.x; }, [](const MR& r){ return r.dims.width; });
}

template<typename MemoryRegions>
uintt getYThreads (const MemoryRegions& regions)
{
  using MR = oap::MemoryRegion;
  std::vector<Section> sections;
  return getThreadsCount (sections, regions, [](MemoryRegions& rs) { utils::sortByY (rs); }, [](const MR& r){ return r.loc.y; }, [](const MR& r){ return r.dims.height; });
}

}
}

#endif	/* THREADSMAPPER_H */
