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

#ifndef OAP_THREADS_MAPPER_API__COMMON_H
#define	OAP_THREADS_MAPPER_API__COMMON_H

#include "Matrix.h"
#include "MatrixInfo.h"
#include "MatrixAPI.h"

#include <functional>
#include <map>
#include <set>
#include <vector>

#include "oapThreadsMapperC.h"
#include "oapThreadsMapperApi.h"
#include "oapMemory_ThreadMapperApi_Types.h"
#include "oapMemoryUtils.h"
#include "oapAssertion.h"

namespace oap {

namespace common {

template<typename MatricesLine, typename GetRefHostMatrix, typename Malloc, typename Memcpy, typename Free, typename CreateBuffer>
ThreadsMapper getThreadsMapper (const std::vector<MatricesLine>& matricesArgs, GetRefHostMatrix&& getRefHostMatrix, Malloc&& malloc, Memcpy&& memcpy, Free&& free, CreateBuffer&& createBuffer, uintt elementsLen, char mode)
{
  using Buffer = std::vector<uintt>;
  struct AllocatedData
  {
    oap::threads::UserData* userData;
    uintt* mapperBuffer;
    uintt* dataBuffer;
  };
  static std::map<oap::ThreadsMapperS*, AllocatedData> s_allocMap;

  uintt linesCount = matricesArgs.size();

  std::vector<std::vector<math::ComplexMatrix>> matricesRefs;
  std::vector<math::MatrixInfo> matrixInfos;
  uintt argsCount = 0;

  std::map<uintt, floatt*> memoryCheckerMap; // to check if args

  for (uintt l = 0; l < linesCount; ++l)
  {
    logAssert (l == 0 || argsCount == matricesArgs[l].size());
    argsCount = matricesArgs[l].size();
    logAssert (argsCount > 0);
    std::vector<math::ComplexMatrix> matricesRef;
    for (uintt argIdx = 0; argIdx < matricesArgs[l].size(); ++argIdx)
    {
      const math::ComplexMatrix* matrix = matricesArgs[l][argIdx];
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

  auto dim = oap::utils::createThreadsDim (matrixInfos,
      [&matricesArgs, &matrixIdxCounter, &matricesRefs, &createBuffer, &map](uintt x, uintt y, uintt lineIndex, uintt width, uintt height)
      {
        const uintt arglen = matricesArgs[lineIndex].size();
        std::vector<uintt> indecies;

        uintt& matrixIdx = matrixIdxCounter[lineIndex];
        for (uintt argIdx = 0; argIdx < arglen; ++argIdx)
        {
          createBuffer (indecies, matricesRefs[lineIndex][argIdx], lineIndex, argIdx, matrixIdx);
        }

        matrixIdx = matrixIdx + 1;
        map[std::make_pair(x,y)] = indecies;
      });


  auto destroy = [&free](oap::ThreadsMapperS* tms)
  {
    oapDebugAssert(s_allocMap.find(tms) != s_allocMap.end());
    const auto& allocatedData = s_allocMap[tms];
    free (allocatedData.userData);
    free (allocatedData.mapperBuffer);
    free (allocatedData.dataBuffer);
    free (tms);
  };

  auto create = [dim, map, argsCount, elementsLen, mode, &malloc, &memcpy] (uintt blockDim[2], uintt gridDim[2])
  {
    std::vector<uintt> mapper_buffer;
    std::vector<uintt> data_buffer;

    const uintt xtb = blockDim[0] * gridDim[0];
    const uintt ytb = blockDim[1] * gridDim[1];

    for (uintt x = 0; x < xtb; ++x)
    {
      for (uintt y = 0; y < ytb; ++y)
      {
        auto it = map.find (std::make_pair(x, y));
        const uintt indeciesLen = argsCount * elementsLen;
        if (it != map.end())
        {
          mapper_buffer.push_back (data_buffer.size());
          for (uintt argIdx = 0; argIdx < indeciesLen; ++argIdx)
          {
            auto vec = it->second;
            logAssert (vec.size() == indeciesLen);
            data_buffer.push_back (vec[argIdx]);
          }
        }
        else
        {
          mapper_buffer.push_back (MAX_UINTT);
        }
      }
    }

    const size_t mapper_len = mapper_buffer.size();
    const size_t data_len = data_buffer.size();
    logTrace ("Created buffer:  data_buffer (length : %u) and mapper_buffer (length : %u)", data_len, mapper_len);

    oap::ThreadsMapperS* tms = static_cast<oap::ThreadsMapperS*>(malloc(sizeof(oap::ThreadsMapperS)));
    oap::threads::UserData* userData = static_cast<oap::threads::UserData*>(malloc(sizeof(oap::threads::UserData)));
    uintt* cuMapperBuffer = static_cast<uintt*>(malloc (mapper_len * sizeof (uintt)));
    uintt* cuDataBuffer = static_cast<uintt*>(malloc (data_len * sizeof (uintt)));

    memcpy (cuMapperBuffer, mapper_buffer.data(), mapper_len * sizeof(uintt));
    memcpy (cuDataBuffer, data_buffer.data(), data_len * sizeof(uintt));
    memcpy (&tms->data, &userData, sizeof (decltype(userData)));
    memcpy (&tms->mode, &mode, sizeof (decltype(mode)));

    memcpy (&userData->mapperBuffer, &cuMapperBuffer, sizeof(decltype(cuMapperBuffer)));
    memcpy (&userData->dataBuffer, &cuDataBuffer, sizeof(decltype(cuDataBuffer)));
    memcpy (&userData->argsCount, &argsCount, sizeof(decltype(argsCount)));

    s_allocMap[tms] = {userData, cuMapperBuffer, cuDataBuffer};

    return tms;
  };

  return ThreadsMapper (dim.first, dim.second, create, destroy);
}
}
}

#endif
