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

#include "oapGenericNeuralApi.h"
#include "oapDeviceAllocApi.h"

#include "oapCudaMatrixUtils.h"

namespace oap
{
namespace generic
{

namespace
{
inline void checkHostInputs(LayerS& ls, const math::Matrix* hostInputs)
{
  if (hostInputs->columns != 1)
  {
    throw std::runtime_error ("Columns of hostInputs matrix must be equal 1");
  }

  if (hostInputs->rows != ls.getTotalNeuronsCount())
  {
    throw std::runtime_error ("Rows of hostInputs matrix must be equal neurons count (or neurons count + 1 if is bias neuron)");
  }
}

inline void _setReValue (math::Matrix* matrix, floatt v, uintt c, uintt r)
{
  oap::cuda::SetReValue(matrix, v, c, r);
}
}

inline void setHostInputs(LayerS& ls, const math::Matrix* hInputs)
{
  checkHostInputs (ls, hInputs);

  oap::generic::setInputs (ls, hInputs, oap::cuda::CopyHostMatrixToDeviceMatrix, _setReValue);
}

inline void setDeviceInputs(LayerS& ls, const math::Matrix* dInputs)
{
  oap::generic::setInputs (ls, dInputs, oap::cuda::CopyDeviceMatrixToDeviceMatrix, _setReValue);
}

inline math::MatrixInfo getOutputsInfo (const LayerS& ls)
{
  return oap::generic::getOutputsInfo (ls, oap::cuda::GetMatrixInfo);
}

inline math::MatrixInfo getInputsInfo (LayerS& ls)
{
  return oap::cuda::GetMatrixInfo (ls.m_inputs);
}

inline void getOutputs (const LayerS& ls, math::Matrix* matrix, ArgType type)
{
  if (type == ArgType::HOST)
  {
    oap::generic::getOutputs (matrix, ls, oap::cuda::CopyDeviceMatrixToHostMatrix);
  }
  else
  {
    oap::generic::getOutputs (matrix, ls, oap::cuda::CopyDeviceMatrixToDeviceMatrix);
  }
}

inline void setHostWeights (LayerS& ls, math::Matrix* weights)
{
  oap::generic::setHostWeights (ls, weights, oap::cuda::CopyHostMatrixToDeviceMatrix, oap::cuda::GetMatrixInfo, oap::host::GetMatrixInfo);
}

inline void setDeviceWeights (LayerS& ls, math::Matrix* weights)
{
  oap::cuda::CopyDeviceMatrixToDeviceMatrix (ls.m_weights, weights);
}

inline void getHostWeights (math::Matrix* output, const LayerS& ls)
{
  oap::cuda::CopyDeviceMatrixToHostMatrix (output, ls.m_weights);
}

inline void printHostWeights (const LayerS& ls, bool newLine)
{
  std::stringstream sstream;
  sstream << "Layer (" << &ls << ") weights = ";
  std::string matrixStr;

  if (ls.m_weights == nullptr)
  {
    oap::host::ToString (matrixStr, nullptr);
  }
  else
  {
    oap::HostMatrixUPtr matrix = oap::host::NewReMatrix (ls.getTotalNeuronsCount(), ls.m_nextLayer->getTotalNeuronsCount());
    oap::generic::getHostWeights (matrix.get(), ls);

    oap::host::ToString (matrixStr, matrix.get());
  }

  logInfo ("%s %s", sstream.str().c_str(), matrixStr.c_str());
}

using RandCallback = std::function<floatt(uintt c, uintt r, floatt value)>;

inline std::unique_ptr<math::Matrix, std::function<void(const math::Matrix*)>> createRandomMatrix (LayerS& ls, uintt columns, uintt rows, RandCallback&& randCallback)
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

inline void initRandomWeights (LayerS& ls, const LayerS* nextLayer)
{
  if (ls.m_weights == nullptr)
  {
    throw std::runtime_error("m_weights == nullptr");
  }

  auto randomMatrix = createRandomMatrix (ls, ls.m_weightsDim.first, ls. m_weightsDim.second, [&ls, &nextLayer](uintt c, uintt r, floatt v)
  {
    if (nextLayer->m_biasCount == 1 && ls.m_weightsDim.second - 1 == r)
    {
      return 0.;
    }
    return v;
  });

  oap::cuda::CopyHostMatrixToDeviceMatrix (ls.m_weights, randomMatrix.get());
}

inline math::MatrixInfo getWeightsInfo (const LayerS& ls)
{
  return oap::cuda::GetMatrixInfo (ls.m_weights);
}

}
}
#endif
