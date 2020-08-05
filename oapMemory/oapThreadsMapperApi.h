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

#include "Matrix.h"
#include "MatrixAPI.h"

#include <functional>
#include <set>
#include <vector>

#include "oapMemoryPrimitivesApi.h"
#include "oapThreadsMapperApi_AbsIndexAlgo.h"
#include "oapThreadsMapperS.h"
#include "oapThreadsMapperC.h"

namespace oap {

namespace threads {

using Section = std::pair<uintt, uintt>;
using PairMR = std::pair<oap::MemoryRegion, oap::MemoryRegion>;

namespace
{
#if 0

template<typename GetLoc, typename GetDim, typename GetRegion>
oap::MemoryRegion getCommonRegion (const oap::MemoryRegion& region1, const oap::MemoryRegion& region2, GetLoc&& getLoc, GetDim&& getDim, GetRegion&& getRegion)
{
  uintt cx = 0;
  uintt cl = 0;
  uintt x1 = getLoc (region1);
  uintt l1 = getDim (region1);
  uintt x2 = getLoc (region2);
  uintt l2 = getDim (region2);

  intt d12 = x1 - x2 + 1;
  d12 = d12 < 0 ? -d12 : d12;

  if (x1 < x2)
  {
    if (l1 > d12)
    {
      cx = x1;
      cl = d12;
    }
  }
  else if (x2 > x1)
  {
    if (l2 > d12)
    {
      cx = x2;
      cl = d12;
    }
  }
  return getRegion (cx, cl);
}

inline oap::MemoryRegion GetCommonRegionX (const oap::MemoryRegion& region1, const oap::MemoryRegion& region2)
{
  using MR = oap::MemoryRegion;
  return getCommonRegion (region1, region2, [](const MR& mr) { return mr.loc.x; }, [](const MR& mr) { return mr.dims.width; }, [](uintt cp, uintt cl) -> oap::MemoryRegion { return {{cp, 0}, {cl, 0}}; });
}

inline oap::MemoryRegion GetCommonRegionY (const oap::MemoryRegion& region1, const oap::MemoryRegion& region2)
{
  using MR = oap::MemoryRegion;
  return getCommonRegion (region1, region2, [](const MR& mr) { return mr.loc.y; }, [](const MR& mr) { return mr.dims.height; }, [](uintt cp, uintt cl) -> oap::MemoryRegion { return {{0, cp}, {0, cl}}; });
}

inline bool IsCommonRegionX (const oap::MemoryRegion& region1, const oap::MemoryRegion& region2)
{
  auto region = GetCommonRegionX (region1, region2);
  return region.dims.width > 0;
}

inline bool IsCommonRegionY (const oap::MemoryRegion& region1, const oap::MemoryRegion& region2)
{
  auto region = GetCommonRegionX (region1, region2);
  return region.dims.height > 0;
}

inline oap::MemoryRegion MergeCommonRegion (const oap::MemoryRegion& x, const oap::MemoryRegion& y)
{
  return {{x.loc.x, y.loc.y}, {x.dims.width, y.dims.height}};
}

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
uintt getThreadsSectionsSum (const Sections& sections)
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
#endif
}

template<typename MatricesLine, typename GetMatrixInfo,  typename Malloc, typename Memcpy, typename Free>
ThreadsMapper createThreadsMapper (const std::vector<MatricesLine>& matricesArgs, GetMatrixInfo&& getMatrixInfo, Malloc&& malloc, Memcpy&& memcpy, Free&& free)
{
  return oap::aia::getThreadsMapper (matricesArgs, getMatrixInfo, malloc, memcpy, free);
}

}
}

#endif
