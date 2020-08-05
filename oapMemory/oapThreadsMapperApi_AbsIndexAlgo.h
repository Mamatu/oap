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
#include "MatrixInfo.h"
#include "MatrixAPI.h"

#include <functional>
#include <map>
#include <set>
#include <vector>

#include "oapThreadsMapperC.h"
#include "oapThreadsMapperApi.h"
#include "oapMemory_ThreadMapperApi_AbsIndexAlgo_CommonApi.h"
#include "oapMemoryUtils.h"

namespace oap {

namespace aia {

using Section = std::pair<uintt, uintt>;
using PairMR = std::pair<oap::MemoryRegion, oap::MemoryRegion>;

template<typename MatricesLine, typename GetMatrixInfo, typename Malloc, typename Memcpy, typename Free>
ThreadsMapper getThreadsMapper (const std::vector<MatricesLine>& matricesArgs, GetMatrixInfo&& getMatrixInfo, Malloc&& malloc, Memcpy&& memcpy, Free&& free)
{
  using Buffer = std::vector<uintt>;
  static std::map<oap::ThreadsMapperS*, uintt*> s_mapperBufferMap;

  uintt linesCount = matricesArgs.size();

  std::vector<math::MatrixInfo> matrixInfos;
  uintt argsCount = 0;
  for (uintt l = 0; l < linesCount; ++l)
  {
    logAssert (l == 0 || argsCount == matricesArgs[l].size());
    argsCount = matricesArgs[l].size();
    logAssert (argsCount > 0);
    const math::Matrix* output = matricesArgs[l][0];
    auto minfo = getMatrixInfo (output);
    matrixInfos.push_back (minfo);
  }

  std::map<std::pair<uintt, uintt>, std::vector<uintt>> map;
  auto dim = oap::utils::createThreadsBlocks<std::vector<uintt>> (matrixInfos,
      [&matricesArgs](uintt x, uintt y, uintt index)
      {
        const uintt len = matricesArgs[index].size();
        std::vector<uintt> indecies;
        for (uintt idx = 0; idx < len; ++idx)
        {
          indecies.push_back (0);
        }
        return indecies;
      },
      [&map](uintt x, uintt y, const std::vector<uintt>& indecies, uintt width, uintt height)
      {
        map[std::make_pair(x,y)] = indecies;
      }
      );

  std::vector<uintt> buffer;
  for (uintt y = 0; y < dim.second; ++y)
  {
    for (uintt x = 0; x < dim.first; ++x)
    {
      auto it = map.find (std::make_pair(x, y));
      if (it != map.end())
      {
        for (auto it1 = it->second.begin(); it1 != it->second.end(); ++it1)
        {
          buffer.push_back (*it1);
        }
      }
      else
      {
        buffer.push_back (MAX_UINTT);
      }
    }
  }

  auto destroyS = [&free](oap::ThreadsMapperS* tms)
  {
    oapDebugAssert(s_mapperBufferMap.find(tms) != s_mapperBufferMap.end());
    auto buffer = s_mapperBufferMap[tms];
    free (buffer);
    free (tms);
  };

  auto allocS = [dim, map, buffer, argsCount, &malloc, &memcpy]()
  {
    const size_t len = dim.first * dim.second;

    oap::ThreadsMapperS* tms = static_cast<oap::ThreadsMapperS*>(malloc(sizeof(oap::ThreadsMapperS)));
    UserData* userData = static_cast<UserData*>(malloc(sizeof(UserData)));
    uintt* cuBuffer = static_cast<uintt*>(malloc (len));
    char mode = 1;

    memcpy (cuBuffer, buffer.data(), len * sizeof(decltype(cuBuffer)));
    memcpy (&tms->data, &userData, sizeof (decltype(userData)));
    memcpy (&tms->mode, &mode, sizeof (decltype(mode)));

    memcpy (&userData->buffer, &cuBuffer, sizeof(decltype(cuBuffer)));
    memcpy (&userData->argsCount, &argsCount, sizeof(decltype(argsCount)));

    s_mapperBufferMap[tms] = cuBuffer;

    return tms;
  };
  return ThreadsMapper (dim.first, dim.second, allocS, destroyS);
}
}
}

#endif
