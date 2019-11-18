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

#ifndef OAP_DEVICE_NEURAL_API_H
#define OAP_DEVICE_NEURAL_API_H

#include <stdexcept>
#include <random>

#include "oapGenericNeuralApi.h"
#include "oapDeviceAllocApi.h"

#include "oapCudaMatrixUtils.h"

namespace oap
{
namespace device
{

namespace
{

template<typename LayerT>
void checkHostInputs (LayerT& layer, const math::Matrix* hostInputs)
{
  if (hostInputs->columns != 1)
  {
    debugAssert ("Columns of hostInputs matrix must be equal 1" == nullptr);
  }

  if (hostInputs->rows != layer.getRowsCount())
  {
    debugAssert ("Rows of hostInputs matrix must be equal neurons count (or neurons count + 1 if is bias neuron)" == nullptr);
  }
}

void _setReValue (math::Matrix* matrix, floatt v, uintt c, uintt r)
{
  oap::cuda::SetReValue(matrix, v, c, r);
}
}

template<typename LayerT>
void setHostInputs (LayerT& layer, const math::Matrix* hInputs)
{
  checkHostInputs (layer, hInputs);

  oap::generic::setInputs (layer, hInputs, oap::cuda::CopyHostMatrixToDeviceMatrix, _setReValue);
}

template<typename LayerT>
void setDeviceInputs (LayerT& layer, const math::Matrix* dInputs)
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
void getOutputs (const LayerT& layer, math::Matrix* matrix, ArgType type)
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
void setHostWeights (LayerT& layer, math::Matrix* weights)
{
  oap::generic::setHostWeights (layer, weights, oap::cuda::CopyHostMatrixToDeviceMatrix, oap::cuda::GetMatrixInfo, oap::host::GetMatrixInfo);
}

template<typename LayerT>
void setDeviceWeights (LayerT& layer, math::Matrix* weights)
{
  oap::cuda::CopyDeviceMatrixToDeviceMatrix (layer.getBPMatrices()->m_weights, weights);
}

using RandCallback = std::function<floatt(uintt c, uintt r, floatt value)>;

template<typename LayerT>
std::unique_ptr<math::Matrix, std::function<void(const math::Matrix*)>> createRandomMatrix (LayerT& layer, uintt columns, uintt rows, RandCallback&& randCallback)
{
  std::unique_ptr<math::Matrix, std::function<void(const math::Matrix*)>> randomMatrix(oap::host::NewReMatrix(columns, rows),
                  [](const math::Matrix* m){oap::host::DeleteMatrix(m);});

  std::random_device rd;
  std::default_random_engine dre (rd());
  std::uniform_real_distribution<> dis(-0.5, 0.5);

  for (uintt c = 0; c < columns; ++c)
  {
    for (uintt r = 0; r < rows; ++r)
    {
      SetRe (randomMatrix.get(), c, r, randCallback(c, r, dis(dre)));
    }
  }

  return std::move (randomMatrix);
}

template<typename LayerT, typename GetMatrixInfo>
void initRandomWeights (LayerT& layer, const LayerT* nextLayer, GetMatrixInfo&& getMatrixInfo)
{
  math::Matrix* weights = layer.getBPMatrices()->m_weights;

  debugAssert(weights != nullptr);

  auto winfo = getMatrixInfo (layer.getBPMatrices()->m_weights);

  auto randomMatrix = createRandomMatrix (layer, winfo.columns(), winfo.rows(), [&layer, &nextLayer, &winfo](uintt c, uintt r, floatt v)
  {
    if (nextLayer->getBiasesCount() == 1 && winfo.rows() - 1 == r)
    {
      return 0.;
    }
    return v;
  });

  oap::cuda::CopyHostMatrixToDeviceMatrix (weights, randomMatrix.get());
}

template<typename LayerT>
math::MatrixInfo getWeightsInfo (const LayerT& layer)
{
  return oap::cuda::GetMatrixInfo (layer.getBPMatrices()->m_weights);
}

}
}
#endif
