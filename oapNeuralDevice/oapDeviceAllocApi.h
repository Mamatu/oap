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

#ifndef OAP_DEVICE_ALLOC_API_H
#define OAP_DEVICE_ALLOC_API_H

#include "Matrix.h"

#include "oapGenericAllocApi.h"
#include "oapCudaMatrixUtils.h"
#include "oapCudaMemoryApi.h"

namespace oap
{
namespace alloc
{
namespace cuda
{
namespace
{
  inline math::Matrix* _newHostMatrixFromMatrixInfo (const math::MatrixInfo& minfo)
  {
    return oap::host::NewHostMatrixFromMatrixInfo (minfo);
  }

  inline math::Matrix* _newDeviceMatrixDeviceRef (const math::Matrix* matrix)
  {
    return oap::cuda::NewDeviceMatrixDeviceRef (matrix);
  }

  inline math::Matrix* _newDeviceSharedSubMatrix (const math::MatrixDim& mdim, const math::Matrix* matrix)
  {
    return oap::cuda::NewDeviceSharedSubMatrix (mdim, matrix);
  }

  inline oap::Memory _newDeviceMemory (const oap::MemoryDim& dim)
  {
    return oap::cuda::NewMemory (dim);
  }

  inline math::Matrix* _newDeviceMatrixFromMatrixInfo (const math::MatrixInfo& minfo, const oap::Memory& memory)
  {
    return oap::cuda::NewDeviceMatrixFromMatrixInfo (minfo);
  }

  using GenericAllocNeuronsApi = oap::alloc::AllocNeuronsApi<decltype(_newDeviceMemory), decltype(_newDeviceMatrixFromMatrixInfo), decltype(_newDeviceSharedSubMatrix)>;

  using GenericAllocWeightsApi = oap::alloc::AllocWeightsApi<decltype(_newDeviceMatrixFromMatrixInfo), decltype(_newDeviceMatrixDeviceRef), decltype(_newHostMatrixFromMatrixInfo), decltype(oap::cuda::CopyHostMatrixToDeviceMatrix)>;

  using GenericDeallocLayerApi = oap::alloc::DeallocLayerApi<decltype(oap::cuda::DeleteDeviceMatrix), decltype(oap::host::DeleteMatrix)>;
}

class AllocNeuronsApi : public GenericAllocNeuronsApi
{
  public:
    AllocNeuronsApi () :
    GenericAllocNeuronsApi (_newDeviceMemory, _newDeviceMatrixFromMatrixInfo, _newDeviceSharedSubMatrix)
    {}
};

class AllocWeightsApi : public GenericAllocWeightsApi
{
  public:
    AllocWeightsApi () :
    GenericAllocWeightsApi (_newDeviceMatrixFromMatrixInfo, _newDeviceMatrixDeviceRef, _newHostMatrixFromMatrixInfo, oap::cuda::CopyHostMatrixToDeviceMatrix)
    {}
};

class DeallocLayerApi : public GenericDeallocLayerApi
{
  public:
    DeallocLayerApi ():
    GenericDeallocLayerApi (oap::cuda::DeleteDeviceMatrix, oap::host::DeleteMatrix)
    {}
};

}
}
}
#endif
