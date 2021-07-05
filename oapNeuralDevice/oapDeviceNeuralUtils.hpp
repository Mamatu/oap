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

#ifndef OAP_DEVICE_NEURAL_UTILS_H
#define OAP_DEVICE_NEURAL_UTILS_H

#include "oapGenericNeuralUtils.hpp"
#include "oapCudaMatrixUtils.hpp"
#include "oapNetwork.hpp"

namespace oap
{
namespace nutils
{
  
inline void copyHostBufferToDeviceReMatrix (math::ComplexMatrix* matrix, size_t index, const floatt* buffer, size_t size)
{
  floatt* re = oap::cuda::GetReValuesPtr (matrix);
  re += index * size;
  CudaUtils::CopyHostToDevice (re, buffer, size * sizeof(floatt));
}

template<typename LayerT, typename CopyBufferToMatrix>
void copyToInputs (Network* network, LHandler handler, size_t index, const floatt* buffer, size_t size, CopyBufferToMatrix&& copyBufferToMatrix)
{
  LayerT* ilayer = network->getLayer (0, handler);
  oap::nutils::copyToInputs (ilayer, index, buffer, size, copyBufferToMatrix);
}

template<typename LayerT, typename Container2D, typename CopyBufferToMatrix>
void copyToInputs (Network* network, LHandler handler, const Container2D& container2D, CopyBufferToMatrix&& copyBufferToMatrix)
{
  LayerT* ilayer = network->getLayer (0, handler);
  oap::nutils::copyToInputs (ilayer, container2D, copyBufferToMatrix);
}

template<typename Container2D, typename CopyBufferToMatrix>
void copyCudaToExpectedOutput (Network* network, LHandler handler, const Container2D& container2D, CopyBufferToMatrix&& copyBufferToMatrix)
{
  auto matrices = network->getExpected (handler);
  debugAssert (!matrices.empty());

  size_t fsize = 0;
  iterate (container2D, [&matrices, &copyBufferToMatrix, &fsize](const Container2D& container2D, size_t idx)
  {
    const size_t size = container2D[idx].size();
    if (idx == 0) { fsize = size; }
    debugAssert (fsize == size);
    copyBufferToMatrix (matrices[idx], container2D[idx].data(), size);
  });
}

template<typename Container2D>
void createDeviceExpectedOutput (Network* network, LHandler handler, const Container2D& container2D, ArgType containerType)
{
#ifdef DEBUG
  const auto* layer = network->getLayer (network->getLayersCount() - 1, handler);
  debugAssert (layer != nullptr);

  size_t containerLength = oap::nutils::getElementsCount (container2D);
  size_t errorsLength = layer->getRowsCount();
  debugAssertMsg (containerLength == errorsLength, "Number of elements in container is different than allocated size of errors matrix in layer");
#endif

  if (containerType == ArgType::DEVICE)
  {
    createExpectedOutput (network, handler, container2D, containerType, oap::cuda::NewDeviceReMatrix, oap::cuda::CopyDeviceBufferToDeviceReMatrix);
  }
  else if (containerType == ArgType::HOST)
  {
    createExpectedOutput (network, handler, container2D, containerType, oap::cuda::NewDeviceReMatrix, oap::cuda::CopyHostBufferToDeviceReMatrix);
  }
  debugAssertMsg (containerType != ArgType::DEVICE_COPY, "DEVICE_COPY mode is not supported");
}

template<typename LayerT, typename Container2D>
void copyToInputs_oneMatrix (LayerT* ilayer, const Container2D& container2D, ArgType containerType)
{
  if (containerType == ArgType::DEVICE)
  {
    copyToInputs_oneMatrix (ilayer, container2D, oap::cuda::CopyDeviceBufferToDeviceReMatrix);
  }
  else if (containerType == ArgType::HOST)
  {
    copyToInputs_oneMatrix (ilayer, container2D, oap::cuda::CopyHostBufferToDeviceReMatrix);
  }
  debugAssertMsg (containerType != ArgType::DEVICE_COPY, "DEVICE_COPY mode is not supported");
}

template<typename LayerT, typename Container2D>
void copyToInputs_oneMatrix (Network* network, LHandler handler, const Container2D& container2D, ArgType containerType)
{
  LayerT* layer = network->getLayer (0, handler);
  copyToInputs_oneMatrix (layer, container2D, containerType);
}

template<typename LayerT, typename Container2D>
void copyToInputs_multiMatrices (LayerT* ilayer, const Container2D& container2D, ArgType containerType)
{
  if (containerType == ArgType::DEVICE)
  {
    copyToInputs_multiMatrices (ilayer, container2D, oap::cuda::CopyDeviceBufferToDeviceReMatrix);
  }
  else if (containerType == ArgType::HOST)
  {
    copyToInputs_multiMatrices (ilayer, container2D, oap::cuda::CopyHostBufferToDeviceReMatrix);
  }
  debugAssertMsg (containerType != ArgType::DEVICE_COPY, "DEVICE_COPY mode is not supported");
}

 template<typename LayerT, typename Container2D>
void copyToInputs_multiMatrices (Network* network, LHandler handler, const Container2D& container2D, ArgType containerType)
{
  LayerT* layer = network->getLayer (0, handler);
  copyToInputs_multiMatrices (layer, container2D, containerType);
}

}
}

#endif
