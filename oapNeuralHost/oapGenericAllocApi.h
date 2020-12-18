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

#ifndef OAP_GENERIC_ALLOC_API_H
#define OAP_GENERIC_ALLOC_API_H

#include "Matrix.h"

#include "oapHostMatrixUtils.h"
#include "oapHostMemoryApi.h"

namespace oap
{
namespace alloc
{
namespace
{
  inline oap::Memory _newHostMemory (const oap::MemoryDim& dim)
  {
    return oap::host::NewMemory (dim);
  }

  inline math::Matrix* _newHostMatrixFromMatrixInfo (const math::MatrixInfo& minfo, const oap::Memory& memory)
  {
    return oap::host::NewHostMatrixFromMatrixInfo (minfo);
  }

  inline math::Matrix* _newSharedSubMatrix (const math::MatrixDim& mdim, const math::Matrix* matrix)
  {
    return oap::host::NewSharedSubMatrix (mdim, matrix);
  }

  inline math::Matrix* _newMatrixRef (const math::Matrix* matrix)
  {
    return oap::host::NewMatrixRef (matrix);
  }
}

template<typename NewMemory, typename NewMatrixFromMatrixInfo, typename NewSharedSubMatrix>
class AllocNeuronsApi
{
  public:
    AllocNeuronsApi (NewMemory&& _newMemory, NewMatrixFromMatrixInfo&& _newMatrixFromMatrixInfo, NewSharedSubMatrix&& _newSharedSubMatrix) :
                     newMemory (_newMemory), newMatrixFromMatrixInfo (_newMatrixFromMatrixInfo), newSharedSubMatrix(_newSharedSubMatrix)
    {}

    NewMemory&& newMemory;
    NewMatrixFromMatrixInfo&& newMatrixFromMatrixInfo;
    NewSharedSubMatrix&& newSharedSubMatrix;
};

template<typename NewDeviceMatrixFromMatrixInfo, typename NewDeviceMatrixDeviceRef, typename NewHostMatrixFromMatrixInfo, typename CopyHostMatrixToDeviceMatrix>
class AllocWeightsApi
{
  public:
    AllocWeightsApi (NewDeviceMatrixFromMatrixInfo&& _newDeviceMatrixFromMatrixInfo, NewDeviceMatrixDeviceRef&& _newDeviceMatrixDeviceRef,
                     NewHostMatrixFromMatrixInfo&& _newHostMatrixFromMatrixInfo, CopyHostMatrixToDeviceMatrix&& _copyHostMatrixToDeviceMatrix) :
                     newDeviceMatrixFromMatrixInfo (_newDeviceMatrixFromMatrixInfo), newDeviceMatrixDeviceRef (_newDeviceMatrixDeviceRef),
                     newHostMatrixFromMatrixInfo (_newHostMatrixFromMatrixInfo), copyHostMatrixToDeviceMatrix (_copyHostMatrixToDeviceMatrix)
    {}

    NewDeviceMatrixFromMatrixInfo&& newDeviceMatrixFromMatrixInfo;
    NewDeviceMatrixDeviceRef&& newDeviceMatrixDeviceRef;
    NewHostMatrixFromMatrixInfo&& newHostMatrixFromMatrixInfo;
    CopyHostMatrixToDeviceMatrix&& copyHostMatrixToDeviceMatrix;
};

template<typename DeleteKernelMatrix, typename DeleteHostMatrix>
class DeallocLayerApi
{
  public:
    DeallocLayerApi (DeleteKernelMatrix&& _deleteKernelMatrix, DeleteHostMatrix&& _deleteHostMatrix):
                     deleteKernelMatrix (_deleteKernelMatrix), deleteHostMatrix (_deleteHostMatrix)
    {}

    DeleteKernelMatrix&& deleteKernelMatrix;
    DeleteHostMatrix&& deleteHostMatrix;
};

namespace host
{
namespace
{

  using GenericAllocNeuronsApi = oap::alloc::AllocNeuronsApi<decltype(_newHostMemory), decltype(_newHostMatrixFromMatrixInfo), decltype(_newSharedSubMatrix)>;
  using GenericAllocWeightsApi = oap::alloc::AllocWeightsApi<decltype(_newHostMatrixFromMatrixInfo), decltype(_newMatrixRef), decltype(_newHostMatrixFromMatrixInfo), decltype(oap::host::CopyHostMatrixToHostMatrix)>;
  using GenericDeallocLayerApi = oap::alloc::DeallocLayerApi<decltype(oap::host::DeleteMatrix), decltype(oap::host::DeleteMatrix)>;

}

class AllocNeuronsApi : public GenericAllocNeuronsApi
{
  public:
    AllocNeuronsApi () :
    GenericAllocNeuronsApi (_newHostMemory, _newHostMatrixFromMatrixInfo, _newSharedSubMatrix)
    {}
};

class AllocWeightsApi : public GenericAllocWeightsApi
{
  public:
    AllocWeightsApi () :
    GenericAllocWeightsApi (_newHostMatrixFromMatrixInfo, _newMatrixRef, _newHostMatrixFromMatrixInfo, oap::host::CopyHostMatrixToHostMatrix)
    {}
};

class DeallocLayerApi : public GenericDeallocLayerApi
{
  public:
    DeallocLayerApi ():
    GenericDeallocLayerApi (oap::host::DeleteMatrix, oap::host::DeleteMatrix)
    {}
};

}
}
}
#endif
