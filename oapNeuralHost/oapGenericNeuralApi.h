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

#ifndef OAP_GENERIC_NEURAL_API_H
#define OAP_GENERIC_NEURAL_API_H

#include <stdexcept>

#include "oapLayerStructure.h"
#include "oapNetworkStructure.h"

#include "oapGenericAllocApi.h"

#include "oapHostMatrixUtils.h"
#include "oapHostMatrixUPtr.h"

#ifdef OAP_CUDA_BUILD
  #include "oapCudaMatrixUtils.h"
#endif

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
      api.linear (output, input);
    break;
    case Activation::TANH:
      api.tanh (output, input);
    break;
    case Activation::SIN:
      api.sin (output, input);
    break;
    case Activation::RELU:
      api.relu (output, input);
    break;
    case Activation::SOFTPLUS:
      api.softplus (output, input);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename Api>
void activateFunc (math::Matrix* output, math::Matrix* input, Activation activation, Api& api, uintt dims[2])
{
  switch (activation)
  {
    case Activation::SIGMOID:
      api.sigmoid (output, input, dims);
    break;
    case Activation::LINEAR:
      api.linear (output, input, dims);
    break;
    case Activation::TANH:
      api.tanh (output, input, dims);
    break;
    case Activation::SIN:
      api.sin (output, input, dims);
    break;
    case Activation::RELU:
      api.relu (output, input, dims);
    break;
    case Activation::SOFTPLUS:
      api.softplus (output, input, dims);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename Api>
void activateFunc (math::Matrix* output, math::Matrix* input, Activation activation, Api& api, uintt dims[2][2])
{
  switch (activation)
  {
    case Activation::SIGMOID:
      api.sigmoid (output, input, dims);
    break;
    case Activation::LINEAR:
      api.linear (output, input, dims);
    break;
    case Activation::TANH:
      api.tanh (output, input, dims);
    break;
    case Activation::SIN:
      api.sin (output, input, dims);
    break;
    case Activation::RELU:
      api.relu (output, input, dims);
    break;
    case Activation::SOFTPLUS:
      api.softplus (output, input, dims);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename Api>
void derivativeFunc (math::Matrix* output, math::Matrix* input, Activation activation, Api& api)
{
  switch (activation)
  {
    case Activation::SIGMOID:
      api.dsigmoid (output, input);
    break;
    case Activation::LINEAR:
      api.dlinear (output, input);
    break;
    case Activation::TANH:
      api.dtanh (output, input);
    break;
    case Activation::SIN:
      api.dsin (output, input);
    break;
    case Activation::RELU:
      api.drelu (output, input);
    break;
    case Activation::SOFTPLUS:
      api.dsoftplus (output, input);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename Api>
void derivativeFunc (math::Matrix* output, math::Matrix* input, Activation activation, Api& api, uintt dims[2])
{
  switch (activation)
  {
    case Activation::SIGMOID:
      api.dsigmoid (output, input, dims);
    break;
    case Activation::LINEAR:
      api.dlinear (output, input, dims);
    break;
    case Activation::TANH:
      api.dtanh (output, input, dims);
    break;
    case Activation::SIN:
      api.dsin (output, input, dims);
    break;
    case Activation::RELU:
      api.drelu (output, input, dims);
    break;
    case Activation::SOFTPLUS:
      api.dsoftplus (output, input, dims);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename Api>
void derivativeFunc (math::Matrix* output, math::Matrix* input, Activation activation, Api& api, uintt dims[2][2])
{
  switch (activation)
  {
    case Activation::SIGMOID:
      api.dsigmoid (output, input, dims);
    break;
    case Activation::LINEAR:
      api.dlinear (output, input, dims);
    break;
    case Activation::TANH:
      api.dtanh (output, input, dims);
    break;
    case Activation::SIN:
      api.dsin (output, input, dims);
    break;
    case Activation::RELU:
      api.drelu (output, input, dims);
    break;
    case Activation::SOFTPLUS:
      api.dsoftplus (output, input, dims);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename LayerT, typename SetReValue>
void initLayerBiases (LayerT& layer, SetReValue&& setReValue, uintt samples = 1)
{
  if (layer.getBiasesCount() == 1)
  {
    for (uintt idx = 0; idx <= samples * layer.getTotalNeuronsCount(); idx += layer.getTotalNeuronsCount())
    {
      setReValue (layer.getFPMatrices()->m_inputs, 1.f, 0, idx - 1);
    }
  }
  else if (layer.getBiasesCount() > 1)
  {
    debugAssert ("Not supported yet" == nullptr);
  }
}

template<typename Layers, typename SetReValue>
void initNetworkBiases (const Layers& layers, SetReValue&& setReValue)
{
  for (uintt idx = 0; idx < layers.size(); ++idx)
  {
    initLayerBiases (*layers[idx], setReValue);
  }
}

template<typename LayerT, typename CopyMatrixToMatrix, typename SetReValue>
void setInputs(LayerT& layer, const math::Matrix* inputs, CopyMatrixToMatrix&& copyMatrixToMatrix, SetReValue&& setReValue)
{
  copyMatrixToMatrix (layer.getFPMatrices()->m_inputs, inputs);

  initLayerBiases (layer, setReValue);
}

template<typename LayerT, typename CopyKernelMatrixToMatrix>
void getHostWeights (math::Matrix* output, const LayerT& layer, CopyKernelMatrixToMatrix&& copyKernelMatrixToMatrix)
{
  copyKernelMatrixToMatrix (output, layer.getBPMatrices()->m_weights);
}

template<typename LayerT, typename CopyKernelMatrixToMatrix>
void printHostWeights (const LayerT& layer, bool newLine, CopyKernelMatrixToMatrix&& copyKernelMatrixToMatrix)
{
  //debugAssertMsg (layer.getNextLayer() != nullptr, "Provided layer does not contain next layer. Weights matrices are not assigned into last layer.");

  std::stringstream sstream;
  sstream << "Layer (" << &layer << ") weights = ";
  std::string matrixStr;

  if (layer.getBPMatrices() == nullptr || layer.getBPMatrices()->m_weights == nullptr)
  {
    oap::host::ToString (matrixStr, nullptr);
  }
  else
  {
    oap::HostMatrixUPtr matrix = oap::host::NewReMatrix (layer.getTotalNeuronsCount(), layer.getNextLayer()->getTotalNeuronsCount());
    copyKernelMatrixToMatrix (matrix.get(), layer.getBPMatrices()->m_weights);

    oap::host::ToString (matrixStr, matrix.get());
  }

  logInfo ("%s %s", sstream.str().c_str(), matrixStr.c_str());
}

template<typename LayerT, typename Layers, typename Api>
void forwardPropagation_oneSample (const Layers& layers, Api& api)
{
  //debugAssertMsg (layers.getSamplesCount() == 1, "For samples higher than 1 please use forwardPropagationExtended method");

  LayerT* previous = nullptr;
  LayerT* current = layers[0];

  for (uintt idx = 1; idx < layers.size(); ++idx)
  {
    previous = current;
    current = layers[idx];

    uintt dims[3][2] =
    {
      {1, current->getNeuronsCount()},
      {previous->getTotalNeuronsCount(), current->getNeuronsCount()},
      {1, previous->getTotalNeuronsCount()}
    };

    FPMatrices& current_fp = *current->getFPMatrices ();
    FPMatrices& previous_fp = *previous->getFPMatrices ();
    BPMatrices& previous_bp = *previous->getBPMatrices ();

    api.dotProduct (current_fp.m_sums, previous_bp.m_weights, previous_fp.m_inputs, dims);

    activateFunc (current_fp.m_inputs, current_fp.m_sums, current->getActivation(), api, dims[0]);
  }
}

template<typename LayerT, typename Layers, typename Api>
void forwardPropagation_multiSamples (const Layers& layers, Api& api)
{
  //debugAssertMsg (layers.getSamplesCount() > 1, "For samples count equals to 1 please use forwardPropagation method");

  LayerT* previous = nullptr;
  LayerT* current = layers[0];

  for (uintt idx = 1; idx < layers.size(); ++idx)
  {
    previous = current;
    current = layers[idx];

    FPMatrices& current_fp = *current->getFPMatrices ();
    FPMatrices& previous_fp = *previous->getFPMatrices ();
    BPMatrices& previous_bp = *previous->getBPMatrices ();

    uintt dims[3][2] =
    {
      {1, current->getNeuronsCount()},
      {previous->getTotalNeuronsCount(), current->getNeuronsCount()},
      {1, previous->getTotalNeuronsCount()}
    };

    uintt periodicRows = current->getTotalNeuronsCount(); 

    api.dotProductDimPeriodic (current_fp.m_sums, previous_bp.m_weights, previous_fp.m_inputs, dims, periodicRows);

    uintt dims1[2][2] =
    {
      {1, current->getNeuronsCount()},
      {1, current->getTotalNeuronsCount()}
    };

    activateFunc (current_fp.m_inputs, current_fp.m_sums, current->getActivation(), api, dims1);
  }
}

template<typename LayerT, typename Api, typename CopyKernelMatrixToHostMatrix>
void getErrors (math::Matrix* errorsOutput, LayerT& layer, Api& api, math::Matrix* expectedDeviceOutputs,
                oap::ErrorType errorType, CopyKernelMatrixToHostMatrix&& copyKernelMatrixToHostMatrix)
{
  debugAssert (expectedDeviceOutputs != nullptr);

  if (errorType == oap::ErrorType::CROSS_ENTROPY)
  {
    api.crossEntropy (layer.getFPMatrices()->m_errorsAux, expectedDeviceOutputs, layer.getFPMatrices()->m_inputs);
  }
  else
  {
    api.substract (layer.getFPMatrices()->m_errorsAux, layer.getFPMatrices()->m_inputs, expectedDeviceOutputs);
  }
  copyKernelMatrixToHostMatrix (errorsOutput, layer.getFPMatrices()->m_errorsAux);
}

template<typename LayerT, typename Layers, typename Api, typename CopyMatrixToMatrix>
void backPropagation (const Layers& layers, Api& api, CopyMatrixToMatrix&& copyMatrixToMatrix)
{
  auto calcErrors = [&layers, &api]()
  {
    int idx = layers.size () - 1;
    LayerT* next = nullptr;
    LayerT* current = layers[idx];

    auto calculateCurrentErrors = [&api] (LayerT* current)
    {
      FPMatrices& current_fp = *current->getFPMatrices ();

      uintt dims[2] = {1, current->getNeuronsCount()};
      oap::generic::derivativeFunc (current_fp.m_sums, current_fp.m_sums, current->getActivation(), api, dims);
      api.hadamardProductVec (current_fp.m_errors, current_fp.m_errors, current_fp.m_sums);
    };

    calculateCurrentErrors (current);

    do
    {
      next = current;
      --idx;
      current = layers[idx];

      BPMatrices& current_bp = *current->getBPMatrices ();
 
      FPMatrices& current_fp = *current->getFPMatrices ();
      FPMatrices& next_fp = *next->getFPMatrices ();

      api.transpose (current_bp.m_tweights, current_bp.m_weights);

      uintt dims[3][2] =
      {
        {1, current->getTotalNeuronsCount()},
        {next->getNeuronsCount(), current->getTotalNeuronsCount()},
        {1, next->getNeuronsCount()}
      };

      api.dotProduct (current_fp.m_errors, current_bp.m_tweights, next_fp.m_errors, dims);
      calculateCurrentErrors (current);
    }
    while (idx > 1);
  };

  auto calcNablaWeights = [&layers, &api]()
  {
    LayerT* current = nullptr;
    LayerT* next = layers[0];

    for (uintt idx = 1; idx < layers.size(); ++idx)
    {
      current = next;
      next = layers[idx];

      BPMatrices& current_bp = *current->getBPMatrices ();
 
      FPMatrices& current_fp = *current->getFPMatrices ();
      FPMatrices& next_fp = *next->getFPMatrices ();

      api.transpose (current_bp.m_tinputs, current_fp.m_inputs);
#ifdef OAP_CUDA_BUILD
      //PRINT_CUMATRIX(current_bp.m_tinputs);
      //PRINT_CUMATRIX(current_fp.m_inputs);
#endif
      {
        uintt dims[3][2] =
        {
          {current->getTotalNeuronsCount(), next->getNeuronsCount()},
          {current->getTotalNeuronsCount(), 1},
          {1, next->getNeuronsCount()},
        };
        api.tensorProduct (current_bp.m_weights1, current_bp.m_tinputs, next_fp.m_errors, dims);
      }

      api.add (current_bp.m_weights2, current_bp.m_weights2, current_bp.m_weights1);
    }
  };

  LayerT* current = layers.back ();
  FPMatrices& current_fp = *current->getFPMatrices ();

  copyMatrixToMatrix (current_fp.m_errors, current_fp.m_errorsAux);

  calcErrors ();

  calcNablaWeights ();
}

template<typename LayerT, typename Layers, typename Api, typename PostCallback>
void updateWeights(const Layers& layers, Api& api, PostCallback&& postCallback, floatt learningRate, uintt normalizationFactor)
{
  LayerT* current = nullptr;
  LayerT* next = layers[0];

  for (uintt idx = 1; idx < layers.size(); ++idx)
  {
    current = next;
    next = layers[idx];

    floatt lr = learningRate / static_cast<floatt>(normalizationFactor);
    api.multiplyReConstant (current->getBPMatrices()->m_weights2, current->getBPMatrices()->m_weights2, lr);

    api.substract (current->getBPMatrices()->m_weights, current->getBPMatrices()->m_weights, current->getBPMatrices()->m_weights2);
  }

  postCallback ();
}

template<typename AllocNeuronsApi, typename Matrices, typename LayerT>
void allocateFPMatrices (Matrices& matrices, const LayerT& layerRef, uintt samplesCount = 1)
{
  const uintt unitsCount = layerRef.getTotalNeuronsCount ();

  AllocNeuronsApi alloc;

  matrices.m_inputs = alloc.newDeviceReMatrix (1, unitsCount * samplesCount);
  matrices.m_sums = alloc.newDeviceMatrixDeviceRef (matrices.m_inputs);
  matrices.m_errors = alloc.newDeviceMatrixDeviceRef (matrices.m_inputs);
  matrices.m_errorsAux = alloc.newDeviceMatrixDeviceRef (matrices.m_inputs);
}

template<typename DeallocMatrixApi>
void deallocateFPMatrices (FPMatrices& fp)
{
  DeallocMatrixApi dealloc;

  auto del = [&dealloc](math::Matrix** matrix)
  {
    if (matrix != nullptr)
    {
      dealloc.deleteMatrix (*matrix);
      matrix = nullptr;
    }
  };

  del (&fp.m_inputs);
  del (&fp.m_sums);
  del (&fp.m_errors);
  del (&fp.m_errorsAux);
}

template<typename DeallocMatrixApi>
void deallocateBPMatrices (BPMatrices& bp)
{
  DeallocMatrixApi dealloc;

  auto del = [&dealloc](math::Matrix** matrix)
  {
    if (matrix != nullptr)
    {
      dealloc.deleteMatrix (*matrix);
      matrix = nullptr;
    }
  };

  del (&bp.m_tinputs);
  del (&bp.m_weights);
  del (&bp.m_tweights);
  del (&bp.m_weights1);
  del (&bp.m_weights2);
}

template<typename DeallocMatrixApi, typename LayerT>
void deallocateFPMatricesInLayer (LayerT& layer)
{
  deallocateFPMatrices<DeallocMatrixApi> (*layer.getFPMatrices());
}

template<typename DeallocMatrixApi, typename LayerT>
void deallocateBPMatricesInLayer (LayerT& layer)
{
  deallocateBPMatrices<DeallocMatrixApi> (*layer.getBPMatrices());
}

template<typename AllocApi, typename Matrices, typename LayerT>
void allocateBPMatrices (Matrices& matrices, LayerT& layer, const LayerT& nextLayer)
{
  const uintt cUCount = layer.getTotalNeuronsCount ();
  const uintt nUCount = nextLayer.getNeuronsCount ();

  AllocApi alloc;

  matrices.m_tinputs = alloc.newDeviceReMatrix (cUCount, 1); //todo: use transpose
  matrices.m_weights = alloc.newDeviceReMatrix (cUCount, nUCount);
  matrices.m_tweights = alloc.newDeviceReMatrix (nUCount, cUCount);
  matrices.m_weights1 = alloc.newDeviceMatrixDeviceRef (matrices.m_weights);
  matrices.m_weights2 = alloc.newDeviceMatrixDeviceRef (matrices.m_weights);
}

template<typename LayerT, typename DeallocMatrixApi>
void deallocate (LayerT& layer)
{
  deallocateFPMatricesInLayer<DeallocMatrixApi> (layer);
  deallocateBPMatricesInLayer<DeallocMatrixApi> (layer);
}

template<typename LayerT, typename AllocNeuronsApi>
LayerT* createLayer (uintt neurons, bool hasBias, uintt samplesCount, Activation activation, bool bAllocateFPMatrices = true)
{
  LayerT* layer = new LayerT (neurons, hasBias ? 1 : 0, samplesCount, activation);

  logInfo ("Layer %p allocates %u neurons (neurons : %u, bias : %u)", layer, layer->getTotalNeuronsCount(), layer->getNeuronsCount(), layer->getBiasesCount());

  if (bAllocateFPMatrices)
  {
    FPMatrices* fpMatrices = new FPMatrices();

    layer->setFPMatrices (fpMatrices);
    allocateFPMatrices<AllocNeuronsApi> (*layer->getFPMatrices(), *layer, samplesCount);
  }

  return layer;
}

template<typename LayerT, typename AllocWeightsApi>
void connectLayers (LayerT* previous, LayerT* next)
{
  previous->setNextLayer (next);
  oap::generic::allocateBPMatrices<AllocWeightsApi> (*previous->getBPMatrices(), *previous, *next);
}

template<typename LayerT, typename CopyHostMatrixToMatrix, typename GetMatrixInfo>
void setHostWeights (LayerT& layer, math::Matrix* weights, CopyHostMatrixToMatrix&& copyHostMatrixToMatrix, GetMatrixInfo&& getLayerMatrixInfo, GetMatrixInfo&& getArgMatrixInfo)
{
  auto linfo = getLayerMatrixInfo (layer.getBPMatrices()->m_weights);
  auto ainfo = getArgMatrixInfo (weights);

  debugAssert (linfo.columns() == ainfo.columns() && linfo.rows() == ainfo.rows());

  copyHostMatrixToMatrix (layer.getBPMatrices()->m_weights, weights);
}

template<typename LayerT, typename GetMatrixInfo>
math::MatrixInfo getOutputsInfo (const LayerT& layer, GetMatrixInfo&& getMatrixInfo)
{
  return getMatrixInfo (layer.getFPMatrices()->m_inputs);
}

template<typename LayerT, typename CopyMatrixToMatrix>
void getOutputs (math::Matrix* matrix, const LayerT& layer, CopyMatrixToMatrix&& copyMatrixToMatrix)
{
  copyMatrixToMatrix (matrix, layer.getFPMatrices()->m_inputs);
}

}
}
#endif
