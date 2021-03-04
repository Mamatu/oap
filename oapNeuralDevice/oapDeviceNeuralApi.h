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

#ifndef OAP_DEVICE_NEURAL_API_H
#define OAP_DEVICE_NEURAL_API_H

#include <stdexcept>
#include <random>

#include "oapGenericNeuralApi.h"
#include "oapDeviceAllocApi.h"
#include "oapLayer.h"

#include "oapCudaMatrixUtils.h"
#include "oapMatrixRandomGenerator.h"

namespace
{

template<typename LayerT>
void checkHostInputs (LayerT& layer, const math::ComplexMatrix* const hostInputs)
{
  if (gColumns (hostInputs) != 1)
  {
    debugAssert ("Columns of hostInputs matrix must be equal 1" == nullptr);
  }

  if (gRows (hostInputs) != layer.getRowsCount())
  {
    debugAssert ("Rows of hostInputs matrix must be equal neurons count (or neurons count + 1 if is bias neuron)" == nullptr);
  }
}

template<typename LayerT, typename Matrices>
void checkHostInputsMatrices (LayerT& layer, const Matrices& hostInputs)
{

  uintt rows = 0;
  for (uintt idx = 0; idx < hostInputs.size(); ++idx)
  {
    if (gColumns (hostInputs[idx]) != 1)
    {
      debugAssert ("Columns of hostInputs matrix must be equal 1" == nullptr);
    }
    rows += gRows (hostInputs[idx]);
  }

  if (rows != layer.getRowsCount())
  {
    debugAssert ("Rows of hostInputs matrix must be equal neurons count (or neurons count + 1 if is bias neuron)" == nullptr);
  }
}

void _setReValue (math::ComplexMatrix* matrix, uintt c, uintt r, floatt v)
{
  oap::cuda::SetReValue(matrix, c, r, v);
}
}

namespace oap
{
namespace device
{

template<typename LayerT>
void setHostInputs (LayerT& layer, const math::ComplexMatrix* hInputs)
{
  checkHostInputs (layer, hInputs);

  oap::generic::setInputs (layer, hInputs, oap::cuda::CopyHostMatrixToDeviceMatrix, _setReValue);
}

template<typename LayerT>
void setDeviceInputs (LayerT& layer, const math::ComplexMatrix* dInputs)
{
  oap::generic::setInputs (layer, dInputs, oap::cuda::CopyDeviceMatrixToDeviceMatrix, _setReValue);
}

template<typename LayerT, typename Matrices>
void setHostInputs (LayerT& layer, const Matrices& hInputs)
{
  checkHostInputsMatrices (layer, hInputs);

  oap::generic::setInputs (layer, hInputs, oap::cuda::CopyHostMatrixToDeviceMatrix, _setReValue);
}

template<typename LayerT, typename Matrices>
void setDeviceInputs (LayerT& layer, const Matrices& dInputs)
{
  oap::generic::setInputs (layer, dInputs, oap::cuda::CopyDeviceMatrixToDeviceMatrix, _setReValue);
}

template<typename LayerT>
math::MatrixInfo getOutputsInfo (const LayerT& layer)
{
  return oap::generic::getOutputsInfo (layer, oap::cuda::GetMatrixInfo);
}

template<typename LayerT>
math::MatrixInfo getInputsInfo (LayerT& layer)
{
  return oap::cuda::GetMatrixInfo (layer.getFPMatrices()->m_inputs);
}

template<typename LayerT>
void getOutputs (const LayerT& layer, math::ComplexMatrix* matrix, ArgType type)
{
  switch (type)
  {
    case ArgType::HOST:
      oap::generic::getOutputs (matrix, layer, oap::cuda::CopyDeviceMatrixToHostMatrix);
      break;
    default:
      oap::generic::getOutputs (matrix, layer, oap::cuda::CopyDeviceMatrixToDeviceMatrix);
      break;
  }
}

template<typename LayerT>
void setHostWeights (LayerT& layer, math::ComplexMatrix* weights)
{
  oap::generic::setHostWeights (layer, weights, oap::cuda::CopyHostMatrixToDeviceMatrix, oap::cuda::GetMatrixInfo, oap::host::GetMatrixInfo);
}

template<typename LayerT>
void setDeviceWeights (LayerT& layer, math::ComplexMatrix* weights)
{
  oap::cuda::CopyDeviceMatrixToDeviceMatrix (layer.getBPMatrices()->m_weights, weights);
}

template<typename LayerT>
void setWeights (const LayerT& layer, const math::ComplexMatrix* hmatrix)
{
  oap::generic::setWeights (layer, hmatrix, oap::cuda::CopyHostMatrixToDeviceMatrix);
}

inline math::MatrixInfo GetWeightsInfo (const oap::Layer& layer)
{
  return oap::generic::getWeightsInfo (layer, oap::cuda::GetMatrixInfo);
}

}
}
#endif
