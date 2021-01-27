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

}
}
#endif
