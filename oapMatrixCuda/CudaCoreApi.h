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

#ifndef OAP_CUDA_CORE_API_H
#define OAP_CUDA_CORE_API_H

#include "GenericCoreApi.h"
#include "oapCudaMatrixUtils.h"
#include "CudaUtils.h"

#include <functional>

namespace oap
{
namespace cuda
{
  using TypeGetValueIdx = std::function<floatt (const math::Matrix*, uintt)>;
  using TypeSetValueIdx = std::function<void (math::Matrix*, uintt, floatt)>;

  class CudaMatrixApi : public oap::generic::MatrixApi<decltype(oap::cuda::GetMatrixInfo), TypeGetValueIdx, TypeSetValueIdx>
  {
    public:
      CudaMatrixApi () :
        oap::generic::MatrixApi<decltype(oap::cuda::GetMatrixInfo), TypeGetValueIdx, TypeSetValueIdx> (oap::cuda::GetMatrixInfo, CudaUtils::GetReValue, CudaUtils::SetReValue, CudaUtils::GetImValue, CudaUtils::SetImValue)
      {}
  };
}
}

#endif
