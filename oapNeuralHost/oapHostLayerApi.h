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

#ifndef OAP_NEURAL_HOST_LAYER_API_H
#define OAP_NEURAL_HOST_LAYER_API_H

#include "oapLayer.h"
#include "oapHostNeuralApi.h"
#include "oapHostMatrixUtils.h"

namespace
{
void _setReValue (math::Matrix* matrix, uintt c, uintt r, floatt v)
{
  oap::host::SetReValue(matrix, c, r, v);
}

}

class HostLayerApi
{
  public:
  inline math::MatrixInfo getOutputsInfo (const Layer<HostLayerApi>* layer) const;

  inline math::MatrixInfo getInputsInfo (const Layer<HostLayerApi>* layer) const;

  inline void getOutputs (const Layer<HostLayerApi>* layer, math::Matrix* matrix, ArgType type) const;

  inline void getHostWeights (math::Matrix* output, Layer<HostLayerApi>* layer);

  inline void setHostInputs(Layer<HostLayerApi>* layer, const math::Matrix* hInputs);

  inline void setDeviceInputs(Layer<HostLayerApi>* layer, const math::Matrix* dInputs);

  inline math::MatrixInfo getWeightsInfo (const Layer<HostLayerApi>* layer) const;

  inline void printHostWeights (const Layer<HostLayerApi>* layer, bool newLine) const;

  inline void allocate(Layer<HostLayerApi>* layer);
  inline void deallocate(Layer<HostLayerApi>* layer);

  inline void setHostWeights (Layer<HostLayerApi>* layer, math::Matrix* weights);

  inline void setDeviceWeights (Layer<HostLayerApi>* layer, math::Matrix* weights);

  inline math::MatrixInfo getMatrixInfo (math::Matrix* matrix) const;
};

inline math::MatrixInfo HostLayerApi::getOutputsInfo (const Layer<HostLayerApi>* layer) const
{
  return oap::generic::getOutputsInfo (*layer, oap::host::GetMatrixInfo);
}

inline math::MatrixInfo HostLayerApi::getInputsInfo (const Layer<HostLayerApi>* layer) const
{
  return oap::device::getInputsInfo (*layer);
}

inline void HostLayerApi::getOutputs (const Layer<HostLayerApi>* layer, math::Matrix* matrix, ArgType type) const
{
  return oap::device::getOutputs (*layer, matrix, type);
}

inline void HostLayerApi::getHostWeights (math::Matrix* output, Layer<HostLayerApi>* layer)
{
  oap::host::CopyHostMatrixToHostMatrix (output, layer->getBPMatrices()->m_weights);
}

inline void HostLayerApi::setHostInputs(Layer<HostLayerApi>* layer, const math::Matrix* hInputs)
{
  oap::device::setHostInputs (*layer, hInputs);
}

inline void HostLayerApi::setDeviceInputs(Layer<HostLayerApi>* layer, const math::Matrix* dInputs)
{
  oap::device::setDeviceInputs (*layer, dInputs);
}

inline math::MatrixInfo HostLayerApi::getWeightsInfo (const Layer<HostLayerApi>* layer) const
{
  return oap::host::GetMatrixInfo (layer->getBPMatrices()->m_weights);
}

inline void HostLayerApi::printHostWeights (const Layer<HostLayerApi>* layer, bool newLine) const
{
  oap::generic::printHostWeights (*layer, newLine, oap::host::CopyHostMatrixToHostMatrix);
}

inline void HostLayerApi::allocate(Layer<HostLayerApi>* layer)
{}

inline void HostLayerApi::deallocate(Layer<HostLayerApi>* layer)
{
  //oap::generic::deallocate<Layer<HostLayerApi>, oap::alloc::cuda::DeallocLayerApi>(*layer);
}

inline void HostLayerApi::setHostWeights (Layer<HostLayerApi>* layer, math::Matrix* weights)
{
  oap::device::setHostWeights (*layer, weights);
}

inline void HostLayerApi::setDeviceWeights (Layer<HostLayerApi>* layer, math::Matrix* weights)
{
  oap::device::setDeviceWeights (*layer, weights);
}

inline math::MatrixInfo HostLayerApi::getMatrixInfo (math::Matrix* matrix) const
{
  return oap::host::GetMatrixInfo (matrix);
}

#endif
