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

#ifndef OAP_HOST_NEURAL_API_H
#define OAP_HOST_NEURAL_API_H

#include <stdexcept>
#include <random>

#include "oapGenericNeuralApi.h"
#include "oapHostAllocApi.h"

#include "oapHostMatrixUtils.h"
#include "oapMatrixRandomGenerator.h"

namespace oap
{
namespace device
{

namespace
{

template<typename LayerT>
void checkHostInputs (LayerT& layer, const math::Matrix* const hostInputs)
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

void _setReValue (math::Matrix* matrix, uintt c, uintt r, floatt v)
{
  oap::host::SetReValue(matrix, c, r, v);
}
}

template<typename LayerT>
void setHostInputs (LayerT& layer, const math::Matrix* hInputs)
{
  checkHostInputs (layer, hInputs);

  oap::generic::setInputs (layer, hInputs, oap::host::CopyHostMatrixToHostMatrix, _setReValue);
}

template<typename LayerT>
void setDeviceInputs (LayerT& layer, const math::Matrix* dInputs)
{
  oap::generic::setInputs (layer, dInputs, oap::host::CopyHostMatrixToHostMatrix, _setReValue);
}

template<typename LayerT, typename Matrices>
void setHostInputs (LayerT& layer, const Matrices& hInputs)
{
  checkHostInputsMatrices (layer, hInputs);

  oap::generic::setInputs (layer, hInputs, oap::host::CopyHostMatrixToHostMatrix, _setReValue);
}

template<typename LayerT, typename Matrices>
void setDeviceInputs (LayerT& layer, const Matrices& dInputs)
{
  oap::generic::setInputs (layer, dInputs, oap::host::CopyHostMatrixToHostMatrix, _setReValue);
}

template<typename LayerT>
math::MatrixInfo getOutputsInfo (const LayerT& layer)
{
  return oap::generic::getOutputsInfo (layer, oap::host::GetMatrixInfo);
}

template<typename LayerT>
math::MatrixInfo getInputsInfo (LayerT& layer)
{
  return oap::host::GetMatrixInfo (layer.getFPMatrices()->m_inputs);
}

template<typename LayerT>
void getOutputs (const LayerT& layer, math::Matrix* matrix, ArgType type)
{
  switch (type)
  {
    case ArgType::HOST:
      oap::generic::getOutputs (matrix, layer, oap::host::CopyHostMatrixToHostMatrix);
      break;
    default:
      oap::generic::getOutputs (matrix, layer, oap::host::CopyHostMatrixToHostMatrix);
      break;
  }
}

template<typename LayerT>
void setHostWeights (LayerT& layer, math::Matrix* weights)
{
  oap::generic::setHostWeights (layer, weights, oap::host::CopyHostMatrixToHostMatrix, oap::host::GetMatrixInfo, oap::host::GetMatrixInfo);
}

template<typename LayerT>
void setDeviceWeights (LayerT& layer, math::Matrix* weights)
{
  oap::host::CopyHostMatrixToHostMatrix (layer.getBPMatrices()->m_weights, weights);
}

 template<typename LayerT>
math::Matrix* getWeights (const LayerT& layer)
{
  debugAssert (layer.getBPMatrices()->m_weights != nullptr);
  return layer.getBPMatrices()->m_weights;
}

template<typename LayerT, typename GetMatrixInfo>
math::MatrixInfo getWeightsInfo (const LayerT& layer, GetMatrixInfo&& getMatrixInfo)
{
  math::Matrix* weights = getWeights (layer);
  return getMatrixInfo (weights);
}

template<typename LayerT>
void setWeights (const LayerT& layer, const math::Matrix* hmatrix)
{
  math::Matrix* weights = getWeights (layer);
  oap::host::CopyHostMatrixToHostMatrix (weights, hmatrix);
}

template<typename LayerT>
class BiasesFilter final
{
  public:
    BiasesFilter (const math::MatrixInfo& layerWeightsInfo, const LayerT& nextLayerT) :
      m_winfo (layerWeightsInfo), m_nextLayerT (nextLayerT)
    {}

    BiasesFilter (const LayerT& layer, const LayerT& nextLayerT) :
      m_winfo (oap::device::getWeightsInfo(layer, oap::host::GetMatrixInfo)), m_nextLayerT (nextLayerT)
    {}

    floatt operator()(uintt c, uintt r, floatt v) const
    {
      if (m_nextLayerT.getBiasesCount() == 1 && m_winfo.rows() - 1 == r)
      {
        return 0.;
      }
      return v;
    }

  private:
    math::MatrixInfo m_winfo;
    const LayerT& m_nextLayerT;
};

template<typename LayerT, typename MatrixRandomGenerator>
oap::HostMatrixUPtr createRandomMatrix (LayerT& layer, const math::MatrixInfo& minfo, MatrixRandomGenerator&& mrg)
{
  oap::HostMatrixUPtr randomMatrix = oap::host::NewReMatrix (minfo.columns(), minfo.rows());

  for (uintt c = 0; c < minfo.columns(); ++c)
  {
    for (uintt r = 0; r < minfo.rows(); ++r)
    {
      SetRe (randomMatrix.get(), c, r, mrg(c, r));
    }
  }

  //rg (randomMatrix.get(), ArgType::HOST);

  return std::move (randomMatrix);
}

template<typename LayerT, typename GetMatrixInfo, typename MatrixRandomGenerator>
void initRandomWeights (LayerT& layer, const LayerT& nextLayer, GetMatrixInfo&& getMatrixInfo, MatrixRandomGenerator&& mrg)
{
  math::MatrixInfo winfo = getWeightsInfo (layer, getMatrixInfo);

  auto randomMatrix = createRandomMatrix (layer, winfo, mrg);

  setWeights (layer, randomMatrix.get ());
}

template<typename LayerT, typename GetMatrixInfo, typename Range = std::pair<floatt, floatt>>
void initRandomWeightsByRange (LayerT& layer, const LayerT& nextLayer, GetMatrixInfo&& getMatrixInfo, Range&& range = std::pair<floatt, floatt>(-0.5, 0.5))
{
  math::MatrixInfo winfo = getWeightsInfo (layer, getMatrixInfo);

  oap::utils::MatrixRandomGenerator rg (range.first, range.second);
  rg.setFilter (oap::device::BiasesFilter<LayerT> (winfo, nextLayer));

  auto randomMatrix = createRandomMatrix (layer, winfo, rg);

  setWeights (layer, randomMatrix.get ());
  //rg (getWeights (layer), ArgType::DEVICE);
}

template<typename NetworkT, typename Callback>
void iterateNetwork (NetworkT& network, Callback&& callback)
{
  for (size_t idx = 0; idx < network.getLayersCount() - 1; ++idx)
  {
    auto* clayer = network.getLayer (idx);
    auto* nlayer = network.getLayer (idx + 1);

    callback (*clayer, *nlayer);
  }
}

}
}
#endif
