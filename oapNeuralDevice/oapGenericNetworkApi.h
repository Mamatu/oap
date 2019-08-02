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

#ifndef OAP_GENERIC_NETWORK_API_H
#define OAP_GENERIC_NETWORK_API_H

#include <stdexcept>

#include "oapLayerStructure.h"
#include "oapNetworkStructure.h"


#include "oapCudaMatrixUtils.h"
#include "oapHostMatrixUtils.h"

#include "oapHostMatrixUPtr.h"

namespace oap
{
namespace generic
{

template<typename Api>
void activateFunc (math::Matrix* output, math::Matrix* input, Activation activation, Api& api)
{
  switch (activation)
  {
    case Activation::SIGMOID:
      api.sigmoid (output, input);
    break;
    case Activation::LINEAR:
      api.identity (output, input);
    break;
    case Activation::TANH:
      api.tanh (output, input);
    break;
    case Activation::SIN:
      api.sin (output, input);
    break;
  };
}

template<typename Api>
void derivativeFunc (math::Matrix* output, math::Matrix* input, Activation activation, Api& api)
{
  switch (activation)
  {
    case Activation::SIGMOID:
      api.sigmoidDerivative (output, input);
    break;
    case Activation::LINEAR:
      api.identityDerivative (output, input);
    break;
    case Activation::TANH:
      api.tanhDerivative (output, input);
    break;
    case Activation::SIN:
      api.sinDerivative (output, input);
    break;
  };
}

template<typename Layers, typename Api, typename SetReValue>
void forwardPropagation (const Layers& layers, Api& api, SetReValue&& setReValue)
{
  if (layers.size() < 2)
  {
    throw std::runtime_error ("m_layerSs.size() is lower than 2");
  }

  LayerS* previous = nullptr;
  LayerS* current = layers[0];

  for (size_t idx = 1; idx < layers.size(); ++idx)
  {
    previous = current;
    current = layers[idx];

    if (previous->m_biasCount == 1)
    {
      setReValue (previous->m_inputs, 1.f, 0, previous->getTotalNeuronsCount() - 1);
    }

    //PRINT_CUMATRIX(current->m_sums);
    //PRINT_CUMATRIX(previous->m_weights);
    //PRINT_CUMATRIX(previous->m_inputs);
    api.dotProduct (current->m_sums, previous->m_weights, previous->m_inputs);

    activateFunc (current->m_inputs, current->m_sums, current->m_activation, api);
  }
}

inline void allocateNeurons (LayerS& ls, size_t neuronsCount, size_t biasCount)
{
  logInfo ("Layer %p allocates %lu neurons (neurons : %lu, bias : %lu)", &ls, neuronsCount + biasCount, neuronsCount, biasCount);
  ls.m_neuronsCount = neuronsCount;

  ls.m_inputs = oap::cuda::NewDeviceReMatrix (1, ls.getTotalNeuronsCount());
  ls.m_sums = oap::cuda::NewDeviceMatrixDeviceRef (ls.m_inputs);
  ls.m_errors = oap::cuda::NewDeviceMatrixDeviceRef (ls.m_inputs);
  ls.m_errorsAcc = oap::cuda::NewDeviceMatrixDeviceRef (ls.m_inputs);
  ls.m_errorsAux = oap::cuda::NewDeviceMatrixDeviceRef (ls.m_inputs);
  ls.m_errorsHost = oap::host::NewReMatrix (1, ls.getTotalNeuronsCount());
  ls.m_tinputs = oap::cuda::NewDeviceReMatrix (ls.getTotalNeuronsCount(), 1); //todo: use transpose
}

inline void allocateWeights(LayerS& ls, const LayerS* nextLayer)
{
  ls.m_weights = oap::cuda::NewDeviceReMatrix (ls.getTotalNeuronsCount(), nextLayer->getTotalNeuronsCount());
  ls.m_tweights = oap::cuda::NewDeviceReMatrix (nextLayer->getTotalNeuronsCount(), ls.getTotalNeuronsCount());
  ls.m_weights1 = oap::cuda::NewDeviceMatrixDeviceRef (ls.m_weights);
  ls.m_weights2 = oap::cuda::NewDeviceMatrixDeviceRef (ls.m_weights);
  ls.m_weightsDim = std::make_pair (ls.getTotalNeuronsCount(), nextLayer->getTotalNeuronsCount());


  oap::HostMatrixUPtr c = oap::host::NewReMatrix (ls.getTotalNeuronsCount(), 1, 0.);
  ls.m_vec = oap::cuda::NewDeviceReMatrix (ls.getTotalNeuronsCount(), 1);
  oap::cuda::CopyHostMatrixToDeviceMatrix (ls.m_vec, c.get());

  ls.m_nextLayer = nextLayer;
}

inline void deallocate (LayerS& ls)
{
  auto del = [](math::Matrix** matrix)
  {
    if (matrix != nullptr)
    {
      oap::cuda::DeleteDeviceMatrix (*matrix);
      matrix = nullptr;
    }
  };

  del (&ls.m_inputs);
  del (&ls.m_tinputs);
  del (&ls.m_sums);
  del (&ls.m_errors);
  del (&ls.m_errorsAcc);
  del (&ls.m_errorsAux);
  del (&ls.m_weights);
  del (&ls.m_tweights);
  del (&ls.m_weights1);
  del (&ls.m_weights2);
  del (&ls.m_vec);
  oap::host::DeleteMatrix (ls.m_errorsHost);
}

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
}

inline void setHostInputs(LayerS& ls, const math::Matrix* hInputs)
{
  checkHostInputs (ls, hInputs);

  oap::cuda::CopyHostMatrixToDeviceMatrix (ls.m_inputs, hInputs);
}

inline void setDeviceInputs(LayerS& ls, const math::Matrix* dInputs)
{
  oap::cuda::CopyDeviceMatrixToDeviceMatrix (ls.m_inputs, dInputs);
}

inline math::MatrixInfo getOutputsInfo (const LayerS& ls)
{
  return oap::cuda::GetMatrixInfo (ls.m_inputs);
}

inline math::MatrixInfo getInputsInfo (LayerS& ls)
{
  return oap::cuda::GetMatrixInfo (ls.m_inputs);
}

inline void getOutputs (const LayerS& ls, math::Matrix* matrix, ArgType type)
{
  if (type == ArgType::HOST)
  {
    oap::cuda::CopyDeviceMatrixToHostMatrix (matrix, ls.m_inputs);
  }
  else
  {
    oap::cuda::CopyDeviceMatrixToDeviceMatrix (matrix, ls.m_inputs);
  }
}

inline void setHostWeights (LayerS& ls, math::Matrix* weights)
{
  oap::cuda::CopyHostMatrixToDeviceMatrix (ls.m_weights, weights);
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

using RandCallback = std::function<floatt(size_t c, size_t r, floatt value)>;

inline std::unique_ptr<math::Matrix, std::function<void(const math::Matrix*)>> createRandomMatrix (LayerS& ls, size_t columns, size_t rows, RandCallback&& randCallback)
{
  std::unique_ptr<math::Matrix, std::function<void(const math::Matrix*)>> randomMatrix(oap::host::NewReMatrix(columns, rows),
                  [](const math::Matrix* m){oap::host::DeleteMatrix(m);});

  std::random_device rd;
  std::default_random_engine dre (rd());
  std::uniform_real_distribution<> dis(-0.5, 0.5);

  for (size_t c = 0; c < columns; ++c)
  {
    for (size_t r = 0; r < rows; ++r)
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

  auto randomMatrix = createRandomMatrix (ls, ls.m_weightsDim.first, ls. m_weightsDim.second, [&ls, &nextLayer](size_t c, size_t r, floatt v)
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
