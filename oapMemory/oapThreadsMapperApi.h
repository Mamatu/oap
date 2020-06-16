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
#include "oapThreadsMapperPrimitives.h"

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

  ggGG  const auto& prev = sections.back();
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

enum ThreadsCalcAlgo
{
  SIMPLE_ALGO_1,
};

class ThreadsMapper
{
  public:
    using CreateCallback = std::function<oap::ThreadsMapperS* ()>;
    using DestroyCallback = std::function<void (const oap::ThreadsMapperS*)>;

    ThreadsMapper (uintt width, uintt height, const CreateCallback& createCallback, const DestroyCallback& destroyCallback) :
      m_width(width), m_height(height), m_createCallback (createCallback), m_destroyCallback (destroyCallback)
    {}

    ThreadsMapper (uintt width, uintt height, CreateCallback&& createCallback, DestroyCallback&& destroyCallback) :
      m_width(width), m_height(height), m_createCallback (std::move(createCallback)), m_destroyCallback (std::move(destroyCallback))
    {}

    uintt getWidth () const
    {
      return m_width;
    }

    uintt getHeight () const
    {
      return m_height;
    }

    uintt getLength () const 
    {
      return getWidth() * getHeight();
    }

    oap::ThreadsMapperS* create () const
    {
      return m_createCallback ();
    }

    void destroy (const oap::ThreadsMapperS* tms)
    {
      m_destroyCallback (tms);
    }

  private:
    uintt m_width;
    uintt m_height;
    CreateCallback m_createCallback;
    DestroyCallback m_destroyCallback;
};

template<typename Matrices, typename GetV1V2>
std::pair<uintt, uintt> getThreadsMapper_SubSimpleAlgo1 (const Matrices& matrices, GetV1V2&& getV1V2)
{
  uintt lv1 = 0;
  uintt lv2 = 0;
  for (size_t idx = 0; idx < matrices.size(); ++idx) {
    const math::Matrix* matrix = matrices[idx];

    const auto pair = getV1V2 (matrix);
    const auto v1 = pair.first;
    const auto v2 = pair.second;

    lv1 = std::max (v1, lv1);
    lv2 += v2;
  }
  return std::make_pair (lv1, lv2);
}

template<typename Matrices, typename GetMatrixInfo, typename Malloc, typename Memcpy>
ThreadsMapper getThreadsMapper_SimpleAlgo1 (const Matrices& matrices, GetMatrixInfo&& getMatrixInfo, Malloc&& malloc, Memcpy&& memcpy)
{
  using Buffer = std::vector<uintt>;

  floatt* reMem = nullptr;
  floatt* imMem = nullptr;
  for (const math::Matrix* matrix : matrices)
  {
    oapAssert (reMem == matrix->re.ptr || reMem == nullptr);
    reMem = matrix->re.ptr;
    oapAssert (imMem == matrix->im.ptr || imMem == nullptr);
    imMem = matrix->im.ptr;
  }

  if (matrices[0]->re.dims.width == 1)
  {
    auto pair1 = getThreadsMapper_SubSimpleAlgo1 (matrices, [&getMatrixInfo](const math::Matrix* matrix)
    {
      auto minfo = getMatrixInfo (matrix);
      return std::make_pair (minfo.columns(), minfo.rows());
    });

    auto algo1 = [matrices, &malloc, &memcpy, &getMatrixInfo, pair1]()
    {
      Buffer buffer1;
      for (size_t idx = 0; idx < matrices.size(); ++idx)
      {
        math::Matrix* matrix = matrices[idx];
        auto minfo = getMatrixInfo (matrix);

        debugAssert (minfo.columns() <= pair1.first);

        if (minfo.columns() < pair1.first)
        {
          Buffer membuf1 (minfo.columns(), idx);
          Buffer membuf2 (pair1.first - minfo.columns(), MAX_UINTT);
          for (size_t row = 0; row < minfo.rows(); ++row)
          {
            buffer1.insert (buffer1.end(), membuf1.begin(), membuf1.end());
            buffer1.insert (buffer1.end(), membuf2.begin(), membuf2.end());
          }
        }
        else
        {
          std::vector<uintt> membuf1 (minfo.columns(), idx);
          for (size_t row = 0; row < minfo.rows(); ++row)
          {
            buffer1.insert (buffer1.end(), membuf1.begin(), membuf1.end());
          }
        }
      }

      const size_t size = sizeof(uintt*) * pair1.width * pair1.height;

      oap::ThreadsMapperS* tms = static_cast<oap::ThreadsMapperS*>(malloc(sizeof(oap::ThreadsMapperS)));
      void* buffer = static_cast<void*>(malloc(size));

      uintt mode = 1;
      memcpy (tms->data, buffer, sizeof (void*));
      memcpy (&tms->mode, mode, sizeof(decltype(tms->mode)));

      return tms;
    };
    return ThreadsMapper (pair1.first, pair1.second, algo1);
  }
  else
  {
    auto pair2 = getThreadsMapper_SubSimpleAlgo1 (matrices, [&getMatrixInfo](const math::Matrix* matrix)
    {
      auto minfo = getMatrixInfo (matrix);
      return std::make_pair (minfo.rows(), minfo.columns());
    });

    auto algo2 = [matrices, &malloc, &memcpy, &getMatrixInfo, pair2]()
    {
      Buffer membuf1;
      uintt row = 0, rows = 0;
      do
      {
        for (size_t idx = 0; idx < matrices.size(); ++idx)
        {
          math::Matrix* matrix = matrices[idx];
          auto minfo = getMatrixInfo (matrix);
          rows = std::max (rows, minfo.rows());
          if (row < minfo.rows ())
          {
            membuf1.insert (membuf1.end(), minfo.columns(), idx);
          }
          else
          {
            membuf1.insert (membuf1.end(), minfo.columns(), MAX_UINTT);
          }
        }
        ++row;
      } while (row < rows);

      const size_t size = sizeof(uintt*) * pair2.width * pair2.height;

      oap::ThreadsMapperS* tms = static_cast<oap::ThreadsMapperS*>(malloc(sizeof(oap::ThreadsMapperS)));
      void* buffer = static_cast<void*>(malloc(size));

      uintt mode = 1;
      memcpy (tms->data, buffer, sizeof (void*));
      memcpy (&tms->mode, mode, sizeof(decltype(tms->mode)));

      return tms;
    };
    return ThreadsMapper (pair2.second, pair2.first, algo2);
  }
}

template<typename Matrices, typename GetMatrixInfo,  typename Malloc, typename Memcpy>
ThreadsMapper createThreadsMapper (const Matrices& matrices, GetMatrixInfo&& getMatrixInfo, Malloc&& malloc, Memcpy&& memcpy)
{
  return getThreadsMapper_SimpleAlgo1 (matrices, getMatrixInfo, malloc, memcpy);
}

}
}

#endif
