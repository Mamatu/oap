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

#ifndef OAP_THREADS_MAPPER_API__MATRIX_POS_ALGO_H
#define	OAP_THREADS_MAPPER_API__MATRIX_POS_ALGO_H

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

namespace oap {

namespace mp {

template<typename MatricesLine, typename GetRefHostMatrix, typename Malloc, typename Memcpy, typename Free>
ThreadsMapper getThreadsMapper (const std::vector<MatricesLine>& matricesArgs, GetRefHostMatrix&& getRefHostMatrix, Malloc&& malloc, Memcpy&& memcpy, Free&& free)
{
  auto createBuffer = [](std::vector<uintt>& indecies, const math::ComplexMatrix& matrixRef, uintt lineIndex, uintt /*argIdx*/, uintt matrixIdx)
  {
    common::Pos pos = oap::common::GetMatrixPosFromMatrixIdx (matrixRef.re.mem, matrixRef.re.reg, matrixIdx);

    indecies.push_back (lineIndex);
    indecies.push_back (pos.x);
    indecies.push_back (pos.y);
  };

  return oap::common::getThreadsMapper (matricesArgs, getRefHostMatrix, malloc, memcpy, free, createBuffer, MP_INDECIES_COUNT, OAP_THREADS_MAPPER_MODE__MP);  
}
}
}

#endif
