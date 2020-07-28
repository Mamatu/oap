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

#ifndef OAP_THREADS_MAPPER_API__SIMPLE_ALGO_1_H
#define	OAP_THREADS_MAPPER_API__SIMPLE_ALGO_1_H

#include "Matrix.h"
#include "MatrixAPI.h"

#include <functional>
#include <map>
#include <set>
#include <vector>

#include "oapThreadsMapperC.h"
#include "oapThreadsMapperApi.h"
#include "oapMemory_ThreadMapperApi_AbsIndexAlgo_CommonApi.h"

namespace oap {

namespace aia {

using Section = std::pair<uintt, uintt>;
using PairMR = std::pair<oap::MemoryRegion, oap::MemoryRegion>;

template<typename MatricesLine, typename GetMatrixInfo, typename Malloc, typename Memcpy, typename Free>
ThreadsMapper getThreadsMapper (const std::vector<const MatricesLine*>& matricesArgs, GetMatrixInfo&& getMatrixInfo, Malloc&& malloc, Memcpy&& memcpy, Free&& free)
{
  using Buffer = std::vector<uintt>;
  static std::map<oap::ThreadsMapperS*, std::pair<uintt*,UserData*>> s_mapperBufferMap;

  uintt linesCount = matricesArgs.size();

  for (uintt l = 0; l < linesCount; ++l)
  {
    uintt argsCount = matricesArgs[l]->size();
    logAssert (argsCount > 0);
    const math::Matrix* output = (*matricesArgs[l])[0];
    auto minfo = getMatrixInfo (output);
  }

  auto destroyS = [&free](oap::ThreadsMapperS* tms)
  {
    oapDebugAssert(s_mapperBufferMap.find(tms) != s_mapperBufferMap.end());
    auto pair = s_mapperBufferMap[tms];
    free (pair.first);
    free (pair.second);
    free (tms);
  };

  auto pair2 = std::make_pair (0, 0);

  auto algo2 = [matricesArgs, &malloc, &memcpy, &getMatrixInfo, &pair2]()
  {
    Buffer membuf1;
    uintt row = 0, rows = 0;
    do
    {
      for (size_t idx = 0; idx < matricesArgs.size(); ++idx)
      {
        const MatricesLine* matricesLine = matricesArgs[idx];
        auto minfo = getMatrixInfo ((*matricesLine)[0]);
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

    const size_t size = sizeof(uintt*) * pair2.first * pair2.second;

    oap::ThreadsMapperS* tms = static_cast<oap::ThreadsMapperS*>(malloc(sizeof(oap::ThreadsMapperS)));
    /*UserData* ud = static_cast<UserData*>(malloc(sizeof(UserData)));

    uintt* buffer = static_cast<uintt*>(malloc(size));
    ::memcpy (buffer, membuf1.data(), membuf1.size() * sizeof (Buffer::value_type));

    uintt mode = OAP_THREADS_MAPPER_MODE__SIMPLE;
    uintt margsCount = matrices.size();

    memcpy (&tms->data, &ud, sizeof (UserData*));
    memcpy (&tms->mode, &mode, sizeof(decltype(tms->mode)));
    memcpy (&ud->buffer, &buffer, sizeof (uintt*));
    memcpy (&ud->argsCount, &margsCount, sizeof (decltype(margsCount)));

    s_mapperBufferMap[tms] = std::make_pair (buffer, ud);*/
    return tms;
  };
  return ThreadsMapper (pair2.second, pair2.first, algo2, destroyS);
}
}
}

#endif
