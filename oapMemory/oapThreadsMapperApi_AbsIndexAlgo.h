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

template<typename MatricesLine, typename GetRefHostMatrix, typename Malloc, typename Memcpy, typename Free>
ThreadsMapper getThreadsMapper (const std::vector<MatricesLine>& matricesArgs, GetRefHostMatrix&& getRefHostMatrix, Malloc&& malloc, Memcpy&& memcpy, Free&& free)
{
  using Buffer = std::vector<uintt>;
  struct AllocatedData
  {
    UserData* userData;
    uintt* buffer;
  };
  static std::map<oap::ThreadsMapperS*, AllocatedData> s_allocMap;

  uintt linesCount = matricesArgs.size();

  std::vector<std::vector<math::Matrix>> matricesRefs;
  std::vector<math::MatrixInfo> matrixInfos;
  uintt argsCount = 0;
  for (uintt l = 0; l < linesCount; ++l)
  {
    logAssert (l == 0 || argsCount == matricesArgs[l].size());
    argsCount = matricesArgs[l].size();
    logAssert (argsCount > 0);
    std::vector<math::Matrix> matricesRef;
    for (uintt argIdx = 0; argIdx < matricesArgs[l].size(); ++argIdx)
    {
      const math::Matrix* matrix = matricesArgs[l][argIdx];
      auto refmatrix = getRefHostMatrix (matrix);
      matricesRef.push_back (refmatrix);
      if (argIdx == 0)
      {
        matrixInfos.push_back (math::MatrixInfo (refmatrix));
      }
    }
    matricesRefs.push_back (matricesRef);
  }

  std::map<std::pair<uintt, uintt>, std::vector<uintt>> map;
  std::map<uintt, uintt> matrixIdxCounter;

  auto dim = oap::utils::createThreadsDim<std::vector<uintt>> (matrixInfos,
      [&matricesArgs, &matrixIdxCounter, &matricesRefs](uintt x, uintt y, uintt index)
      {
        const uintt arglen = matricesArgs[index].size();
        std::vector<uintt> indecies;

        uintt& matrixIdx = matrixIdxCounter[index];

        for (uintt argIdx = 0; argIdx < arglen; ++argIdx)
        {
          indecies.push_back (index);
          indecies.push_back (oap::common::GetMemIdxFromMatrixIdx (matricesRefs[index][argIdx].re, matricesRefs[index][argIdx].reReg, matrixIdx));
        }

        matrixIdx = matrixIdx + 1;

        return indecies;
      },
      [&map](uintt x, uintt y, const std::vector<uintt>& indecies, uintt width, uintt height)
      {
        map[std::make_pair(x,y)] = indecies;
      });

  std::vector<uintt> buffer;
  for (uintt x = 0; x < dim.first; ++x)
  {
    for (uintt y = 0; y < dim.second; ++y)
    {
      auto it = map.find (std::make_pair(x, y));
      const uintt indeciesLen = argsCount * INDECIES_COUNT;
      if (it != map.end())
      {
        for (uintt argIdx = 0; argIdx < indeciesLen; ++argIdx)
        {
          auto vec = it->second;
          logAssert (vec.size() == indeciesLen);
          buffer.push_back (vec[argIdx]);
        }
      }
      else
      {
        buffer.insert (buffer.end(), indeciesLen, MAX_UINTT);
      }
    }
  }

  auto destroy = [&free](oap::ThreadsMapperS* tms)
  {
    oapDebugAssert(s_allocMap.find(tms) != s_allocMap.end());
    const auto& allocatedData = s_allocMap[tms];
    free (allocatedData.userData);
    free (allocatedData.buffer);
    free (tms);
  };

  auto create = [dim, map, buffer, argsCount, &malloc, &memcpy]()
  {
    const size_t len = dim.first * dim.second * argsCount * INDECIES_COUNT;
    logInfo ("Created buffer with length = %u", len);
    logAssert (buffer.size() == len);

    oap::ThreadsMapperS* tms = static_cast<oap::ThreadsMapperS*>(malloc(sizeof(oap::ThreadsMapperS)));
    UserData* userData = static_cast<UserData*>(malloc(sizeof(UserData)));
    uintt* cuBuffer = static_cast<uintt*>(malloc (len * sizeof (uintt)));
    char mode = 1;

    memcpy (cuBuffer, buffer.data(), len * sizeof(uintt));
    memcpy (&tms->data, &userData, sizeof (decltype(userData)));
    memcpy (&tms->mode, &mode, sizeof (decltype(mode)));

    memcpy (&userData->buffer, &cuBuffer, sizeof(decltype(cuBuffer)));
    memcpy (&userData->argsCount, &argsCount, sizeof(decltype(argsCount)));

    s_allocMap[tms] = {userData, cuBuffer};

    return tms;
  };
  return ThreadsMapper (dim.first, dim.second, create, destroy);
}
}
}

#endif
