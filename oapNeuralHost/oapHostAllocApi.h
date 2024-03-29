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

#ifndef OAP_HOST_ALLOC_API_H
#define OAP_HOST_ALLOC_API_H

#include "Matrix.h"

#include "oapGenericAllocApi.h"
#include "oapHostMatrixUtils.h"
#include "oapHostMemoryApi.h"

namespace oap
{
namespace alloc
{
namespace host
{
namespace
{
  inline math::ComplexMatrix* _newHostMatrixFromMatrixInfo (const math::MatrixInfo& minfo)
  {
    return oap::host::NewHostMatrixFromMatrixInfo (minfo);
  }

  inline math::ComplexMatrix* _newHostMatrixRef (const math::ComplexMatrix* matrix)
  {
    return oap::host::NewComplexMatrixRef (matrix);
  }

  inline math::ComplexMatrix* _newHostSharedSubMatrix (const math::MatrixDim& mdim, const math::ComplexMatrix* matrix)
  {
    return oap::host::NewSharedSubMatrix (mdim, matrix);
  }

  inline oap::Memory _newHostMemory (const oap::MemoryDim& dim)
  {
    return oap::host::NewMemory (dim);
  }

  using GenericAllocNeuronsApi = oap::alloc::AllocNeuronsApi<decltype(_newHostMemory), decltype(_newHostMatrixFromMatrixInfo), decltype(_newHostSharedSubMatrix)>;

  using GenericAllocWeightsApi = oap::alloc::AllocWeightsApi<decltype(_newHostMatrixFromMatrixInfo), decltype(_newHostMatrixRef), decltype(_newHostMatrixFromMatrixInfo), decltype(oap::host::CopyHostMatrixToHostMatrix)>;

  using GenericDeallocLayerApi = oap::alloc::DeallocLayerApi<decltype(oap::host::DeleteComplexMatrix), decltype(oap::host::DeleteComplexMatrix)>;
}

class AllocNeuronsApi : public GenericAllocNeuronsApi
{
  public:
    AllocNeuronsApi () :
    GenericAllocNeuronsApi (_newHostMemory, _newHostMatrixFromMatrixInfo, _newHostSharedSubMatrix)
    {}
};

class AllocWeightsApi : public GenericAllocWeightsApi
{
  public:
    AllocWeightsApi () :
    GenericAllocWeightsApi (_newHostMatrixFromMatrixInfo, _newHostMatrixRef, _newHostMatrixFromMatrixInfo, oap::host::CopyHostMatrixToHostMatrix)
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
