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

#include "oapNetworkCudaApi.h"
#include "oapDeviceNeuralApi.h"
#include "oapDeviceAllocApi.h"
#include "oapDeviceLayer.h"
#include "oapGenericNeuralApi.h"

namespace oap
{

NetworkCudaApi::~NetworkCudaApi()
{}

void NetworkCudaApi::setReValue (math::ComplexMatrix* matrix, uintt c, uintt r, floatt v)
{
  oap::cuda::SetReValue (matrix, c, r, v);
}

void NetworkCudaApi::setHostWeights (Layer& layer, math::ComplexMatrix* weights)
{
  oap::device::setHostWeights (layer, weights);
}

math::MatrixInfo NetworkCudaApi::getMatrixInfo (const math::ComplexMatrix* matrix) const
{
  return oap::cuda::GetMatrixInfo (matrix);
}

/*FPMatrices* NetworkCudaApi::allocateFPMatrices (const Layer& layer, uintt samplesCount)
{
  return oap::generic::allocateFPMatrices<oap::alloc::cuda::AllocNeuronsApi> (layer, samplesCount);
}

FPMatrices* NetworkCudaApi::allocateSharedFPMatrices (const Layer& layer, FPMatrices* orig)
{
  return oap::generic::allocateSharedFPMatrices<oap::alloc::cuda::AllocNeuronsApi> (layer, orig);
}

BPMatrices* NetworkCudaApi::allocateBPMatrices (NBPair& pnb, NBPair& nnb)
{
  return oap::generic::allocateBPMatrices<oap::alloc::cuda::AllocWeightsApi> (pnb, nnb);
}

void NetworkCudaApi::deallocateFPMatrices (FPMatrices* fpmatrices)
{
  oap::generic::deallocateFPMatrices<oap::alloc::cuda::DeallocLayerApi>(fpmatrices);
}

void NetworkCudaApi::deallocateBPMatrices (BPMatrices* bpmatrices)
{
  oap::generic::deallocateBPMatrices<oap::alloc::cuda::DeallocLayerApi>(bpmatrices);
}*/

Layer* NetworkCudaApi::createLayer (uintt neurons, bool hasBias, uintt samplesCount, Activation activation)
{
  return new oap::DeviceLayer (neurons, hasBias ? 1 : 0, samplesCount, activation);
}

void NetworkCudaApi::copyKernelMatrixToKernelMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
  oap::cuda::CopyDeviceMatrixToDeviceMatrix (dst, src);
}

void NetworkCudaApi::copyKernelMatrixToHostMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
  oap::cuda::CopyDeviceMatrixToHostMatrix (dst, src);
}

void NetworkCudaApi::copyHostMatrixToKernelMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
  oap::cuda::CopyHostMatrixToDeviceMatrix (dst, src);
}

void NetworkCudaApi::deleteKernelMatrix (const math::ComplexMatrix* matrix)
{
  oap::cuda::DeleteDeviceMatrix (matrix);
}

math::ComplexMatrix* NetworkCudaApi::newKernelReMatrix (uintt columns, uintt rows)
{
  return oap::cuda::NewDeviceReMatrix (columns, rows);
}

math::ComplexMatrix* NetworkCudaApi::newKernelMatrixHostRef (const math::ComplexMatrix* matrix)
{
  return oap::cuda::NewDeviceMatrixHostRef (matrix);
}

math::ComplexMatrix* NetworkCudaApi::newKernelMatrixKernelRef (const math::ComplexMatrix* matrix)
{
  return oap::cuda::NewDeviceMatrixDeviceRef (matrix);
}
/*
void NetworkCudaApi::connectLayers (oap::Layer* previous, oap::Layer* layer)
{
  oap::generic::connectLayers<Layer, oap::alloc::cuda::AllocWeightsApi>(previous, layer);
}
*/
math::ComplexMatrix* NetworkCudaApi::newKernelSharedSubMatrix (const math::MatrixLoc& loc, const math::MatrixDim& mdim, const math::ComplexMatrix* matrix)
{
  return oap::cuda::NewDeviceSharedSubMatrix (loc, mdim, matrix);
}

oap::Memory NetworkCudaApi::newKernelMemory (const oap::MemoryDim& dim)
{
  return oap::cuda::NewMemory (dim); 
}

math::ComplexMatrix* NetworkCudaApi::newKernelMatrixFromMatrixInfo (const math::MatrixInfo& minfo)
{
  return oap::cuda::NewDeviceMatrixFromMatrixInfo (minfo);
}

}
