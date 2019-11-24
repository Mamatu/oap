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

namespace
{
template<typename LayerT>
class BiasesFilter final
{
  public:
    BiasesFilter (const math::MatrixInfo& layerWeightsInfo, const LayerT& nextLayerT) :
      m_winfo (layerWeightsInfo), m_nextLayerT (nextLayerT)
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

class RandomGenerator final
{
  public:

    using ValueCallback = std::function<floatt(uintt, uintt, floatt)>;
    using MatrixCallback = std::function<void(math::Matrix*)>;

    RandomGenerator (floatt min, floatt max) :
      m_min(min), m_max(max), m_rd(), m_dre (m_rd()), m_dis (m_min, m_max)
    {}

    void setValueCallback (ValueCallback&& vc)
    {
      m_valueCallback = std::move (vc);
    }

    void setValueCallback (const ValueCallback& vc)
    {
      m_valueCallback = vc;
    }

    void setMatrixCallback (MatrixCallback&& mc)
    {
      m_matrixCallback = std::move (mc);
    }

    void setMatrixCallback (const MatrixCallback& mc)
    {
      m_matrixCallback = mc;
    }

    floatt operator()(uintt column, uintt row)
    {
      floatt v = m_dis(m_dre);

      if (m_valueCallback)
      {
        return m_valueCallback (column, row, v);
      }
      return v;
    }

    void operator()(math::Matrix* matrix)
    {
      if (m_matrixCallback)
      {
        m_matrixCallback (matrix);
      }
    }

  private:
    floatt m_min, m_max;
    std::random_device m_rd;
    std::default_random_engine m_dre;
    std::uniform_real_distribution<floatt> m_dis;
    ValueCallback m_valueCallback;
    MatrixCallback m_matrixCallback;
};
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
  oap::cuda::CopyHostMatrixToDeviceMatrix (weights, hmatrix);
}

template<typename LayerT, typename RandomGenerator>
oap::HostMatrixUPtr createRandomMatrix (LayerT& layer, const math::MatrixInfo& minfo, RandomGenerator&& rg)
{
  oap::HostMatrixUPtr randomMatrix = oap::host::NewReMatrix (minfo.columns(), minfo.rows());

  for (uintt c = 0; c < minfo.columns(); ++c)
  {
    for (uintt r = 0; r < minfo.rows(); ++r)
    {
      SetRe (randomMatrix.get(), c, r, rg(c, r));
    }
  }

  rg (randomMatrix.get());

  return std::move (randomMatrix);
}

template<typename LayerT, typename GetMatrixInfo, typename Range = std::pair<floatt, floatt>>
void initRandomWeights (LayerT& layer, const LayerT& nextLayer, GetMatrixInfo&& getMatrixInfo, Range&& range = std::pair<floatt, floatt>(-0.5, 0.5))
{
  math::MatrixInfo winfo = getWeightsInfo (layer, getMatrixInfo);

  BiasesFilter<LayerT> bfilter (winfo, nextLayer);
  RandomGenerator rg (range.first, range.second, bfilter);

  auto randomMatrix = createRandomMatrix (layer, winfo, rg);

  setWeights (layer, randomMatrix.get ());
}

}
}
#endif
