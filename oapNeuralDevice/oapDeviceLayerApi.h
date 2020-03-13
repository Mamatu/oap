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

#ifndef OAP_NEURAL_DEVICE_LAYER_API_H
#define OAP_NEURAL_DEVICE_LAYER_API_H

#include "oapLayer.h"
#include "oapDeviceNeuralApi.h"
#include "oapCudaMatrixUtils.h"

namespace
{
void _setReValue (math::Matrix* matrix, uintt c, uintt r, floatt v)
{
  oap::cuda::SetReValue(matrix, c, r, v);
}

}

class DeviceLayerApi
{
  public:
  inline math::MatrixInfo getOutputsInfo (const Layer<DeviceLayerApi>* layer) const;

  inline math::MatrixInfo getInputsInfo (const Layer<DeviceLayerApi>* layer) const;

  inline void getOutputs (const Layer<DeviceLayerApi>* layer, math::Matrix* matrix, ArgType type) const;

  inline void getHostWeights (Layer<DeviceLayerApi>* layer, math::Matrix* output);

  inline void setHostInputs(Layer<DeviceLayerApi>* layer, const math::Matrix* hInputs);

  inline void setDeviceInputs(Layer<DeviceLayerApi>* layer, const math::Matrix* dInputs);

  inline math::MatrixInfo getWeightsInfo (const Layer<DeviceLayerApi>* layer) const;

  inline void printHostWeights (const Layer<DeviceLayerApi>* layer, bool newLine) const;

  inline void allocate(Layer<DeviceLayerApi>* layer);
  inline void deallocate(Layer<DeviceLayerApi>* layer);

  inline void setHostWeights (Layer<DeviceLayerApi>* layer, math::Matrix* weights);

  inline void setDeviceWeights (Layer<DeviceLayerApi>* layer, math::Matrix* weights);

  inline math::MatrixInfo getMatrixInfo (math::Matrix* matrix) const;
};

inline math::MatrixInfo DeviceLayerApi::getOutputsInfo (const Layer<DeviceLayerApi>* layer) const
{
  return oap::generic::getOutputsInfo (*layer, oap::cuda::GetMatrixInfo);
}

inline math::MatrixInfo DeviceLayerApi::getInputsInfo (const Layer<DeviceLayerApi>* layer) const
{
  return oap::device::getInputsInfo (*layer);
}

inline void DeviceLayerApi::getOutputs (const Layer<DeviceLayerApi>* layer, math::Matrix* matrix, ArgType type) const
{
  return oap::device::getOutputs (*layer, matrix, type);
}

inline void DeviceLayerApi::getHostWeights (Layer<DeviceLayerApi>* layer, math::Matrix* output)
{
  oap::cuda::CopyDeviceMatrixToHostMatrix (output, layer->getBPMatrices()->m_weights);
}

inline void DeviceLayerApi::setHostInputs(Layer<DeviceLayerApi>* layer, const math::Matrix* hInputs)
{
  oap::device::setHostInputs (*layer, hInputs);
}

inline void DeviceLayerApi::setDeviceInputs(Layer<DeviceLayerApi>* layer, const math::Matrix* dInputs)
{
  oap::device::setDeviceInputs (*layer, dInputs);
}

inline math::MatrixInfo DeviceLayerApi::getWeightsInfo (const Layer<DeviceLayerApi>* layer) const
{
  return oap::cuda::GetMatrixInfo (layer->getBPMatrices()->m_weights);
}

inline void DeviceLayerApi::printHostWeights (const Layer<DeviceLayerApi>* layer, bool newLine) const
{
  oap::generic::printHostWeights (*layer, newLine, oap::cuda::CopyDeviceMatrixToHostMatrix);
}

inline void DeviceLayerApi::allocate(Layer<DeviceLayerApi>* layer)
{}

inline void DeviceLayerApi::deallocate(Layer<DeviceLayerApi>* layer)
{
  //oap::generic::deallocate<Layer<DeviceLayerApi>, oap::alloc::cuda::DeallocLayerApi>(*layer);
}

inline void DeviceLayerApi::setHostWeights (Layer<DeviceLayerApi>* layer, math::Matrix* weights)
{
  oap::device::setHostWeights (*layer, weights);
}

inline void DeviceLayerApi::setDeviceWeights (Layer<DeviceLayerApi>* layer, math::Matrix* weights)
{
  oap::device::setDeviceWeights (*layer, weights);
}

inline math::MatrixInfo DeviceLayerApi::getMatrixInfo (math::Matrix* matrix) const
{
  return oap::cuda::GetMatrixInfo (matrix);
}

#endif
