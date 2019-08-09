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

#ifndef OAP_GENERIC_ALLOC_API_H
#define OAP_GENERIC_ALLOC_API_H

#include "Matrix.h"

#include "oapHostMatrixUtils.h"

namespace oap
{
namespace alloc
{

template<typename NewDeviceReMatrix, typename NewDeviceMatrixDeviceRef, typename NewReMatrix>
class AllocNeuronsApi
{
  public:
    AllocNeuronsApi (NewDeviceReMatrix&& _newDeviceReMatrix, NewDeviceMatrixDeviceRef&& _newDeviceMatrixDeviceRef, NewReMatrix&& _newReMatrix) :
                     newDeviceReMatrix (_newDeviceReMatrix), newDeviceMatrixDeviceRef (_newDeviceMatrixDeviceRef), newReMatrix (_newReMatrix)
    {}

    NewDeviceReMatrix&& newDeviceReMatrix;
    NewDeviceMatrixDeviceRef&& newDeviceMatrixDeviceRef;
    NewReMatrix&& newReMatrix;
};

template<typename NewDeviceReMatrix, typename NewDeviceMatrixDeviceRef, typename NewReMatrix, typename CopyHostMatrixToDeviceMatrix>
class AllocWeightsApi
{
  public:
    AllocWeightsApi (NewDeviceReMatrix&& _newDeviceReMatrix, NewDeviceMatrixDeviceRef&& _newDeviceMatrixDeviceRef,
                     NewReMatrix&& _newReMatrix, CopyHostMatrixToDeviceMatrix&& _copyHostMatrixToDeviceMatrix) :
                     newDeviceReMatrix (_newDeviceReMatrix), newDeviceMatrixDeviceRef (_newDeviceMatrixDeviceRef),
                     newReMatrix (_newReMatrix), copyHostMatrixToDeviceMatrix (_copyHostMatrixToDeviceMatrix)
    {}

    NewDeviceReMatrix&& newDeviceReMatrix;
    NewDeviceMatrixDeviceRef&& newDeviceMatrixDeviceRef;
    NewReMatrix&& newReMatrix;
    CopyHostMatrixToDeviceMatrix&& copyHostMatrixToDeviceMatrix;
};

template<typename DeleteMatrix, typename DeleteErrorsMatrix>
class DeallocLayerApi
{
  public:
    DeallocLayerApi (DeleteMatrix&& _deleteMatrix, DeleteErrorsMatrix&& _deleteErrorsMatrix):
                     deleteMatrix (_deleteMatrix), deleteErrorsMatrix (_deleteErrorsMatrix)
    {}

    DeleteMatrix&& deleteMatrix;
    DeleteErrorsMatrix&& deleteErrorsMatrix;
};

namespace host
{
namespace
{
  inline math::Matrix* NewReMatrix (uintt columns, uintt rows)
  {
    return oap::host::NewReMatrix (columns, rows);
  }

  inline math::Matrix* NewMatrixRef (const math::Matrix* matrix)
  {
    return oap::host::NewMatrixRef (matrix);
  }

  using GenericAllocNeuronsApi = oap::alloc::AllocNeuronsApi<decltype(NewReMatrix), decltype(NewMatrixRef), decltype(NewReMatrix)>;
  using GenericAllocWeightsApi = oap::alloc::AllocWeightsApi<decltype(NewReMatrix), decltype(NewMatrixRef), decltype(NewReMatrix), decltype(oap::host::CopyHostMatrixToHostMatrix)>;
  using GenericDeallocLayerApi = oap::alloc::DeallocLayerApi<decltype(oap::host::DeleteMatrix), decltype(oap::host::DeleteMatrix)>;

}

class AllocNeuronsApi : public GenericAllocNeuronsApi
{
  public:
    AllocNeuronsApi () :
    GenericAllocNeuronsApi (NewReMatrix, NewMatrixRef, NewReMatrix)
    {}
};

class AllocWeightsApi : public GenericAllocWeightsApi
{
  public:
    AllocWeightsApi () :
    GenericAllocWeightsApi (NewReMatrix, NewMatrixRef, NewReMatrix, oap::host::CopyHostMatrixToHostMatrix)
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
