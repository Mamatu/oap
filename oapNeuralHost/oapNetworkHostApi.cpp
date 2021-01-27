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

#include "oapNetworkHostApi.h"
#include "oapHostNeuralApi.h"
#include "oapHostAllocApi.h"
#include "oapHostLayer.h"

namespace oap
{

NetworkHostApi::~NetworkHostApi()
{}

void NetworkHostApi::setReValue (math::Matrix* matrix, uintt c, uintt r, floatt v)
{
  oap::host::SetReValue (matrix, c, r, v);
}

void NetworkHostApi::setHostWeights (Layer& layer, math::Matrix* weights)
{
  oap::host::setHostWeights (layer, weights);
}

math::MatrixInfo NetworkHostApi::getMatrixInfo (const math::Matrix* matrix) const
{
  return oap::host::GetMatrixInfo (matrix);
}
/*
FPMatrices* NetworkHostApi::allocateFPMatrices (const Layer& layer, uintt samplesCount)
{
  return oap::generic::allocateFPMatrices<oap::alloc::host::AllocNeuronsApi> (layer, samplesCount);
}

FPMatrices* NetworkHostApi::allocateSharedFPMatrices (const Layer& layer, FPMatrices* orig)
{
  return oap::generic::allocateSharedFPMatrices<oap::alloc::host::AllocNeuronsApi> (layer, orig);
}

BPMatrices* NetworkHostApi::allocateBPMatrices (NBPair& pnb, NBPair& nnb)
{
  return oap::generic::allocateBPMatrices<oap::alloc::host::AllocWeightsApi> (pnb, nnb);
}

void NetworkHostApi::deallocateFPMatrices (FPMatrices* fpmatrices)
{
  oap::generic::deallocateFPMatrices<oap::alloc::host::DeallocLayerApi>(fpmatrices);
}

void NetworkHostApi::deallocateBPMatrices (BPMatrices* bpmatrices)
{
  oap::generic::deallocateBPMatrices<oap::alloc::host::DeallocLayerApi>(bpmatrices);
}
*/
Layer* NetworkHostApi::createLayer (uintt neurons, bool hasBias, uintt samplesCount, Activation activation)
{
  return new oap::HostLayer (neurons, hasBias, samplesCount, activation);
}

void NetworkHostApi::copyKernelMatrixToKernelMatrix (math::Matrix* dst, const math::Matrix* src)
{
  oap::host::CopyHostMatrixToHostMatrix (dst, src);
}

void NetworkHostApi::copyKernelMatrixToHostMatrix (math::Matrix* dst, const math::Matrix* src)
{
  oap::host::CopyHostMatrixToHostMatrix (dst, src);
}

void NetworkHostApi::copyHostMatrixToKernelMatrix (math::Matrix* dst, const math::Matrix* src)
{
  oap::host::CopyHostMatrixToHostMatrix (dst, src);
}

void NetworkHostApi::deleteKernelMatrix (const math::Matrix* matrix)
{
  oap::host::DeleteMatrix (matrix);
}

math::Matrix* NetworkHostApi::newKernelReMatrix (uintt columns, uintt rows)
{
  return oap::host::NewReMatrix (columns, rows);
}

math::Matrix* NetworkHostApi::newKernelMatrixHostRef (const math::Matrix* matrix)
{
  return oap::host::NewMatrixRef (matrix);
}

math::Matrix* NetworkHostApi::newKernelMatrixKernelRef (const math::Matrix* matrix)
{
  return oap::host::NewMatrixRef (matrix);
}

/*
void NetworkHostApi::connectLayers (oap::Layer* previous, oap::Layer* layer)
{
  oap::generic::connectLayers<Layer, oap::alloc::host::AllocWeightsApi>(previous, layer);
}
*/

math::Matrix* NetworkHostApi::newKernelSharedSubMatrix (const math::MatrixDim& mdim, const math::Matrix* matrix)
{
  return oap::host::NewSharedSubMatrix (mdim, matrix);
}

oap::Memory NetworkHostApi::newKernelMemory (const oap::MemoryDim& dim)
{
  return oap::host::NewMemory (dim);
}

math::Matrix* NetworkHostApi::newKernelMatrixFromMatrixInfo (const math::MatrixInfo& minfo)
{
  return oap::host::NewHostMatrixFromMatrixInfo (minfo);
}

}
