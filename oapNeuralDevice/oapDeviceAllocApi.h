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

#ifndef OAP_DEVICE_ALLOC_API_H
#define OAP_DEVICE_ALLOC_API_H

#include "Matrix.h"

#include "oapGenericAllocApi.h"
#include "oapCudaMatrixUtils.h"

namespace oap
{
namespace alloc
{
namespace cuda
{
namespace
{
  inline math::Matrix* NewReMatrix (size_t columns, size_t rows)
  {
    return oap::host::NewReMatrix (columns, rows);
  }

  using GenericAllocNeuronsApi = oap::alloc::AllocNeuronsApi<decltype(oap::cuda::NewDeviceReMatrix), decltype(oap::cuda::NewDeviceMatrixDeviceRef), decltype(NewReMatrix)>;
  using GenericAllocWeightsApi = oap::alloc::AllocWeightsApi<decltype(oap::cuda::NewDeviceReMatrix), decltype(oap::cuda::NewDeviceMatrixDeviceRef), decltype(NewReMatrix), decltype(oap::cuda::CopyHostMatrixToDeviceMatrix)>;

}

class AllocNeuronsApi : public GenericAllocNeuronsApi
{
  public:
    AllocNeuronsApi () :
    GenericAllocNeuronsApi (oap::cuda::NewDeviceReMatrix, oap::cuda::NewDeviceMatrixDeviceRef, oap::alloc::cuda::NewReMatrix)
    {}
};

class AllocWeightsApi : public GenericAllocWeightsApi
{
  public:
    AllocWeightsApi () :
    GenericAllocWeightsApi (oap::cuda::NewDeviceReMatrix, oap::cuda::NewDeviceMatrixDeviceRef, NewReMatrix, oap::cuda::CopyHostMatrixToDeviceMatrix)
    {}
};

}
}
}
#endif
