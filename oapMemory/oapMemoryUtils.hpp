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

#ifndef OAP_MEMORY_UTILS_H
#define	OAP_MEMORY_UTILS_H

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>
#include "oapMemory.hpp"
#include "MatrixInfo.hpp"

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

namespace
{
  template<typename MatrixInfoVec, typename CreateCallback>
  bool createThreadsDim_simple (std::pair<uintt, uintt>& dim, const MatrixInfoVec& infos, CreateCallback&& createCallback)
  {
    math::MatrixInfo mref;
    for (uintt idx = 0; idx < infos.size(); ++idx)
    {
      auto minfo = infos[idx];
      if (idx == 0)
      {
        mref = minfo;
      }
      if (mref != minfo)
      {
        return false;
      }
    }
    uintt columns = mref.columns();
    uintt rows = mref.rows() * infos.size();
    dim = std::make_pair (columns, rows);
    for (uintt r = 0; r < rows; ++r)
    {
      for (uintt c = 0; c < columns; ++c)
      {
        uintt index = r / mref.rows();
        createCallback (c, r, index, columns, rows);
      }
    }
    return true;
  }
}

  /**
   * @param createCallback - (uintt x, uintt y, uintt index, uintt columns, uintt rows)
   */
  template<typename MatrixInfoVec, typename CreateCallback>
  std::pair<uintt, uintt> createThreadsDim (const MatrixInfoVec& infos, CreateCallback&& createCallback)
  {
    oapAssert (!infos.empty());

    {
      std::pair<uintt, uintt> dim;
      bool b = createThreadsDim_simple (dim, infos, createCallback);
      if (b)
      {
        return dim;
      }
    }
    using Tuple = std::tuple<uintt, uintt, uintt>;
    using Pair = std::pair<uintt, uintt>;

    using MapPosIndex = std::map<Pair, uintt>;

    using Dim = std::pair<uintt, uintt>;

    using Output = std::pair<Dim, MapPosIndex>;
    using SubOutputKey = std::pair<std::vector<Dim>, uintt>;

    std::map<SubOutputKey, Output> subOutputs;
    std::map<const math::MatrixInfo*, uintt> mii;

    auto calcSize = [](const Output& output)
    {
      return output.first.first * output.first.second;
    };

    auto sortFunc = [&calcSize](const Output& output1, const Output& output2)
    {
      return calcSize (output1) < calcSize (output2);
    };

    using MInfoPtrs = std::vector<const math::MatrixInfo*>;
    MInfoPtrs f_infosPtr;

    for (uintt idx = 0; idx < infos.size(); ++idx)
    {
      f_infosPtr.push_back (&infos[idx]);
      mii[f_infosPtr[idx]] = idx;
    }

    auto createDPB = [&mii](const MInfoPtrs& minfosPtr, uintt id)
    {
      using SubOutcome = std::pair<std::vector<Dim>, uintt>;
      SubOutcome dpb_set;
      for (const auto& ptr : minfosPtr)
      {
        dpb_set.first.push_back ({ptr->columns(), ptr->rows()});
      }
      dpb_set.second = id;
      return dpb_set;
    };

    auto fill = [](uintt c, uintt c1, uintt r, uintt r1, uintt idx)
    {
      MapPosIndex map;
      for (uintt x = c; x < c1; ++x)
      {
        for (uintt y = r; y < r1; ++y)
        {
          map[std::make_pair(x, y)] = idx;
        }
      }
      return map;
    };

    auto calcMap = [&fill](const Tuple& tuple, uintt c, uintt r, uintt index)
    {
      auto pos = std::make_pair(std::get<0>(tuple), std::get<1>(tuple));
      uintt w = std::get<0>(tuple);
      uintt h = std::get<1>(tuple);
      MapPosIndex map;
      switch (std::get<2>(tuple))
      {
        case 0:
            map = fill (w - c, w, 0, r, index);
          break;
        case 1:
            map = fill (0, c, h - r, r, index);
          break;
        case 2:
            map = fill (w - r, w, 0, c, index);
          break;
        case 3:
            map = fill (0, r, h - c, c, index);
          break;
      };
      return map;
    };

    std::function<Output(const MInfoPtrs& infos, const Dim& o_dim, const MapPosIndex& mpi, uintt id)> calc;
    calc = [&calc, &calcMap, &subOutputs, &createDPB, &f_infosPtr, &mii, &sortFunc](const MInfoPtrs& minfosPtr, const Dim& o_dim, const MapPosIndex& mpi, uintt id)
    {
      auto dpd_set = std::move (createDPB (minfosPtr, id));
      auto dpd_it = subOutputs.find (dpd_set);
      if (dpd_it != subOutputs.end())
      {
        return dpd_it->second;
      }

      std::vector<Output> outputs;

      for (uintt idx = 0; idx < minfosPtr.size(); ++idx)
      {
        MInfoPtrs new_infoPtrs = minfosPtr;
        auto it = new_infoPtrs.begin();
        std::advance (it, idx);
        auto info = *it;
        new_infoPtrs.erase (it);

        const uintt o_columns = o_dim.first;
        const uintt o_rows = o_dim.second;

        const uintt c = info->columns();
        const uintt r = info->rows();

        auto tuple1 = std::make_tuple(o_columns + c, r > o_rows ? r : o_rows, 0);
        auto tuple2 = std::make_tuple(c > o_columns ? c : o_columns, o_rows + r, 1);
        auto tuple3 = std::make_tuple(o_columns + r, c > o_rows ? c : o_rows, 2);
        auto tuple4 = std::make_tuple(r > o_columns ? r : o_columns, o_rows + c, 3);

        std::vector<Tuple> tuples = {tuple1, tuple2, tuple3, tuple4};
        std::vector<Output> outputs1;

        for (uintt tupleIdx = 0; tupleIdx < tuples.size(); ++tupleIdx)
        {
          auto tuple = tuples[tupleIdx];
          Dim dim = {std::get<0>(tuple), std::get<1>(tuple)};

          auto mapPosIndex = calcMap (tuple, c, r, mii[info]);
          mapPosIndex.insert (mpi.begin(), mpi.end());

          Output output1;
          if (new_infoPtrs.size() > 0)
          {
            output1 = calc (new_infoPtrs, dim, mapPosIndex, tupleIdx);
          }
          else
          {
            output1 = std::make_pair (dim, mapPosIndex);
          }
          outputs1.push_back (output1);
        }

        std::sort (outputs1.begin(), outputs1.end(), sortFunc);
        subOutputs[dpd_set] = outputs1.front();

        outputs.push_back (outputs1.front());
      }
      std::sort (outputs.begin(), outputs.end(), sortFunc);
      return outputs.front();
    };

    Dim initDim = {0u, 0u};
    MapPosIndex mpi;
    auto output = calc (f_infosPtr, initDim, mpi, 0);

    auto dim = output.first;
    auto& map = output.second;

    for (auto it = map.begin(); it != map.end(); ++it)
    {
      createCallback (it->first.first, it->first.second, it->second, dim.first, dim.second);
    }
    return std::make_pair(dim.first, dim.second);
  }
}
}

#endif	/* THREADSMAPPER_H */
