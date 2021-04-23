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

#ifndef OAP_GENERIC_NEURAL_API_H
#define OAP_GENERIC_NEURAL_API_H

#include <stdexcept>

#include "oapLayerStructure.h"

#include "oapGenericAllocApi.h"

#include "oapHostMatrixUtils.h"
#include "oapHostComplexMatrixUPtr.h"
#include "oapProcedures.h"

#include "oapNetworkGenericApi.h"

namespace oap
{
namespace generic
{

template<typename MT, typename Api>
void activateFunc (MT output, MT input, Activation activation, Api& api)
{
  logTrace("");
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
    case Activation::PRELU:
      api.prelu (output, input);
    break;
    case Activation::SOFTPLUS:
      api.softplus (output, input);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename MT, typename Api>
void activateFunc (MT output, MT input, Activation activation, Api& api, oap::generic::Dim2 dims)
{
  logTrace("");
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
    case Activation::PRELU:
      api.prelu (output, input, dims);
    break;
    case Activation::SOFTPLUS:
      api.softplus (output, input, dims);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename MT, typename Api>
void activateFunc (MT output, MT input, Activation activation, Api& api, oap::generic::Dim22 dims)
{
  logTrace("");
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
    case Activation::PRELU:
      api.prelu (output, input, dims);
    break;
    case Activation::SOFTPLUS:
      api.softplus (output, input, dims);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename MT, typename Api>
void derivativeFunc (MT output, MT input, Activation activation, Api& api)
{
  logTrace("");
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
    case Activation::PRELU:
      api.prelu (output, input);
    break;
    case Activation::SOFTPLUS:
      api.dsoftplus (output, input);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename MT, typename Api>
void derivativeFunc (MT output, MT input, Activation activation, Api& api, oap::generic::Dim2 dims)
{
  logTrace("");
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
    case Activation::PRELU:
      api.prelu (output, input, dims);
    break;
    case Activation::SOFTPLUS:
      api.dsoftplus (output, input, dims);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename MT, typename Api>
void derivativeFunc (MT output, MT input, Activation activation, Api& api, oap::generic::Dim22 dims)
{
  logTrace("");
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
    case Activation::PRELU:
      api.prelu (output, input, dims);
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
  logTrace("");
  if (layer.getBiasesCount() == 1)
  {
    uintt size = layer.getInputs().size();
    for (uintt idx1 = 0; idx1 < size; ++idx1)
    {
      for (uintt idx = layer.getTotalNeuronsCount(); idx <= samples * layer.getTotalNeuronsCount(); idx += layer.getTotalNeuronsCount())
      {
        setReValue (layer.getFPMatrices(idx1)->m_inputs, 0, idx - 1, 1.f);
      }
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
  logTrace("");
  for (uintt idx = 0; idx < layers.size(); ++idx)
  {
    initLayerBiases (*layers[idx], setReValue);
  }
}

template<typename LayerT, typename Matrices, typename CopyMatrixToMatrix, typename SetReValue>
void setInputs (LayerT& layer, const Matrices& inputs, CopyMatrixToMatrix&& copyMatrixToMatrix, SetReValue&& setReValue)
{
  logTrace("");
  for (uintt idx = 0; idx < inputs.size(); ++idx)
  {
    copyMatrixToMatrix (layer.getFPMatrices(idx)->m_inputs, inputs[idx]);
  }

  initLayerBiases (layer, setReValue);
}

template<typename LayerT, typename CopyMatrixToMatrix, typename SetReValue>
void setInputs(LayerT& layer, const math::ComplexMatrix* input, CopyMatrixToMatrix&& copyMatrixToMatrix, SetReValue&& setReValue)
{
  std::vector<const math::ComplexMatrix*> inputs = {input};
  setInputs (layer, inputs, copyMatrixToMatrix, setReValue);
}

template<typename LayerT, typename CopyKernelMatrixToMatrix>
void getHostWeights (math::ComplexMatrix* output, const LayerT& layer, CopyKernelMatrixToMatrix&& copyKernelMatrixToMatrix)
{
  logTrace("");
  copyKernelMatrixToMatrix (output, layer.getBPMatrices()->m_weights);
}

template<typename LayerT, typename CopyKernelMatrixToMatrix>
void printHostWeights (const LayerT& layer, bool newLine, CopyKernelMatrixToMatrix&& copyKernelMatrixToMatrix)
{
  //debugAssertMsg (layer.getNextLayer() != nullptr, "Provided layer does not contain next layer. Weights matrices are not assigned into last layer.");

  logTrace("");
  std::stringstream sstream;
  sstream << "Layer (" << &layer << ") weights = ";
  std::string matrixStr;

  if (layer.getBPMatrices() == nullptr || layer.getBPMatrices()->m_weights == nullptr)
  {
    oap::host::ToString (matrixStr, nullptr);
  }
  else
  {
    oap::HostComplexMatrixUPtr matrix = oap::host::NewHostMatrixFromMatrixInfo (layer.getWeightsInfo());
    copyKernelMatrixToMatrix (matrix.get(), layer.getBPMatrices()->m_weights);

    oap::host::ToString (matrixStr, matrix.get());
  }

  debugInfo ("%s %s", sstream.str().c_str(), matrixStr.c_str());
}

template<typename LayerT, typename Layers, typename Api>
void forwardPropagation_oneSample (const Layers& layers, Api& api)
{
  //debugAssertMsg (layers.getSamplesCount() == 1, "For samples higher than 1 please use forwardPropagationExtended method");
  logTrace("");

  LayerT* previous = nullptr;
  LayerT* current = layers[0];

  for (uintt idx = 1; idx < layers.size(); ++idx)
  {
    previous = current;
    current = layers[idx];

    oap::generic::Dim32 dims
    {{
      {1, current->getNeuronsCount()},
      {previous->getTotalNeuronsCount(), current->getNeuronsCount()},
      {1, previous->getTotalNeuronsCount()}
    }};

    FPMatrices& current_fp = *current->getFPMatrices ();
    FPMatrices& previous_fp = *previous->getFPMatrices ();
    BPMatrices& previous_bp = *previous->getBPMatrices ();

    api.dotProduct (current_fp.m_sums, previous_bp.m_weights, previous_fp.m_inputs, dims);

    activateFunc (current_fp.m_inputs, current_fp.m_sums, previous->getActivation(), api, dims[0]);
  }
}

template<typename LayerT, typename Layers, typename Api>
void forwardPropagation_multiSamples (const Layers& layers, Api& api)
{
  //debugAssertMsg (layers.getSamplesCount() > 1, "For samples count equals to 1 please use forwardPropagation method");
  logTrace("");

  LayerT* previous = nullptr;
  LayerT* current = layers[0];

  for (uintt idx = 1; idx < layers.size(); ++idx)
  {
    previous = current;
    current = layers[idx];

    FPMatrices& current_fp = *current->getFPMatrices ();
    FPMatrices& previous_fp = *previous->getFPMatrices ();
    BPMatrices& previous_bp = *previous->getBPMatrices ();

    oap::generic::Dim32 dims
    {{
      {1, current->getNeuronsCount()},
      {previous->getTotalNeuronsCount(), current->getNeuronsCount()},
      {1, previous->getTotalNeuronsCount()}
    }};

    uintt periodicRows = current->getTotalNeuronsCount(); 

    api.dotProductDimPeriodic (current_fp.m_sums, previous_bp.m_weights, previous_fp.m_inputs, dims, periodicRows);

    oap::generic::Dim22 dims1
    {{
      {1, current->getNeuronsCount()},
      {1, current->getTotalNeuronsCount()}
    }};

    activateFunc (current_fp.m_inputs, current_fp.m_sums, previous->getActivation(), api, dims1);
  }
}

template<typename LayerT, typename Layers, typename Api>
void forwardPropagation_multiMatrices (const Layers& layers, Api& api)
{
  //debugAssertMsg (layers.getSamplesCount() > 1, "For samples count equals to 1 please use forwardPropagation method");

  logTrace("");
  LayerT* previous = nullptr;
  LayerT* current = layers[0];

  for (uintt idx = 1; idx < layers.size(); ++idx)
  {
    previous = current;
    current = layers[idx];

    auto& current_inputs = current->getInputs ();
    auto& current_sums = current->getSums ();
    auto& current_sums_wb = current->getSumsWB ();
    auto& current_inputs_wb = current->getInputsWB ();
    auto& previous_inputs = previous->getInputs ();
    auto& previous_weights = previous->getWeights ();

    api.dotProduct (current_sums_wb, previous_weights, previous_inputs);
    activateFunc (current_inputs_wb, current_sums_wb, previous->getActivation(), api);
  }
}

template<typename LayerT, typename Api, typename CopyKernelMatrixToHostMatrix>
void getErrors (math::ComplexMatrix* errorsOutput, LayerT& layer, Api& api, math::ComplexMatrix* expectedDeviceOutputs, oap::ErrorType errorType, CopyKernelMatrixToHostMatrix&& copyKernelMatrixToHostMatrix)
{
  logTrace("");
  debugAssert (expectedDeviceOutputs != nullptr);

  if (errorType == oap::ErrorType::CROSS_ENTROPY)
  {
    api.crossEntropy (layer.getFPMatrices()->m_errorsAux, expectedDeviceOutputs, layer.getFPMatrices()->m_inputs);
  }
  else
  {
    api.subtract (layer.getFPMatrices()->m_errorsAux, layer.getFPMatrices()->m_inputs, expectedDeviceOutputs);
  }
  copyKernelMatrixToHostMatrix (errorsOutput, layer.getFPMatrices()->m_errorsAux);
}

template<typename Matrices, typename LayerT, typename Api, typename CopyKernelMatrixToHostMatrix>
void getErrors_multiMatrices (const Matrices& errorsOutput, LayerT& layer, Api& api, const Matrices& expectedDeviceOutputs, oap::ErrorType errorType, CopyKernelMatrixToHostMatrix&& copyKernelMatrixToHostMatrix)
{
  debugAssert (!expectedDeviceOutputs.empty());
  debugAssert (!errorsOutput.empty());
  debugAssert (errorsOutput.size() == layer.getFPMatricesCount());

  logTrace("");
  if (errorType == oap::ErrorType::CROSS_ENTROPY)
  {
    //api.crossEntropy (layer.getFPMatrices()->m_errorsAux, expectedDeviceOutputs, layer.getFPMatrices()->m_inputs);
  }
  else
  {
    api.subtract (layer.getErrorsAux(), layer.getInputs(), expectedDeviceOutputs);
  }
  for (uintt idx = 0; idx < errorsOutput.size(); ++idx)
  {
    copyKernelMatrixToHostMatrix (errorsOutput[idx], layer.getErrorsAux()[idx]);
  }
}

template<typename LayerT, typename Layers, typename Api, typename CopyMatrixToMatrix>
void backPropagation (const Layers& layers, Api& api, CopyMatrixToMatrix&& copyMatrixToMatrix)
{
  logTrace("");
  auto calcErrors = [&layers, &api]()
  {
    int idx = layers.size () - 1;
    LayerT* next = nullptr;
    LayerT* current = layers[idx];
    LayerT* previous = layers[idx - 1];

    auto calculateCurrentErrors = [&api] (LayerT* current, LayerT* previous)
    {
      FPMatrices& current_fp = *current->getFPMatrices ();

      oap::generic::Dim2 dims {{1, current->getNeuronsCount()}};
      oap::generic::derivativeFunc (current_fp.m_sums, current_fp.m_sums, previous->getActivation(), api, dims);
      api.hadamardProductVec (current_fp.m_errors, current_fp.m_errors, current_fp.m_sums);
    };

    calculateCurrentErrors (current, previous);

    while (idx > 1)
    {
      next = current;
      --idx;
      current = layers[idx];
      previous = layers[idx - 1];

      BPMatrices& current_bp = *current->getBPMatrices ();
 
      FPMatrices& current_fp = *current->getFPMatrices ();
      FPMatrices& next_fp = *next->getFPMatrices ();

      api.transpose (current_bp.m_tweights, current_bp.m_weights);

      oap::generic::Dim32 dims
      {{
        {1, current->getTotalNeuronsCount()},
        {next->getNeuronsCount(), current->getTotalNeuronsCount()},
        {1, next->getNeuronsCount()}
      }};

      api.dotProduct (current_fp.m_errors, current_bp.m_tweights, next_fp.m_errors, dims);
      calculateCurrentErrors (current, previous);
    }
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
      {
        oap::generic::Dim32 dims
        {{
          {current->getTotalNeuronsCount(), next->getNeuronsCount()},
          {current->getTotalNeuronsCount(), 1},
          {1, next->getNeuronsCount()},
        }};
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

template<typename LayerT, typename Layers, typename Api, typename Api2, typename CopyMatrixToMatrix>
void backPropagation_multiMatrices (const Layers& layers, Api& api, Api2& api2, CopyMatrixToMatrix&& copyMatrixToMatrix)
{
  logTrace("");
  auto calcErrors = [&layers, &api]()
  {
    int idx = layers.size () - 1;
    LayerT* next = nullptr;
    LayerT* current = layers[idx];
    LayerT* previous = layers[idx - 1];

    auto calculateCurrentErrors = [&api] (LayerT* current, LayerT* previous)
    {
      auto& current_sums_wb = current->getSumsWB ();
      auto& current_sums = current->getSums ();
      auto& current_errors = current->getErrors ();

      oap::generic::derivativeFunc (current_sums_wb, current_sums_wb, previous->getActivation(), api);
      api.hadamardProductVec (current_errors, current_errors, current_sums);
    };

    calculateCurrentErrors (current, previous);

    while (idx > 1)
    {
      next = current;
      --idx;
      current = layers[idx];
      previous = layers[idx - 1];

      auto& current_tweights = current->getTWeights ();
      auto& current_weights = current->getWeights ();
      auto& current_errors = current->getErrors();
      auto& current_errors_wb = current->getErrorsWB();
      auto& next_errors = next->getErrors();
      auto& next_errors_wb = next->getErrorsWB();

      api.transpose (current_tweights, current_weights);
      //PRINT_CUMATRIX_CARRAY(current_tweights);
      //PRINT_CUMATRIX_CARRAY(current_weights);


      api.dotProduct (current_errors, current_tweights, next_errors_wb);
      //PRINT_CUMATRIX_CARRAY(current_errors);
      //PRINT_CUMATRIX_CARRAY(current_tweights);
      //PRINT_CUMATRIX_CARRAY(next_errors);
      calculateCurrentErrors (current, previous);
    }
  };
#if 0
  auto calcNablaWeights = [&layers, &api]()
  {
    LayerT* current = nullptr;
    LayerT* next = layers[0];

    for (uintt idx = 1; idx < layers.size(); ++idx)
    {
      current = next;
      next = layers[idx];

      auto& current_weights1 = current->getWeights1 ();
      auto& current_weights2 = current->getWeights2 ();
      auto& current_inputs = current->getInputs ();
      auto& current_tinputs = current->getTInputs ();
 
      auto& next_errors = next->getErrors ();
      auto& next_errors_wb = next->getErrorsWB ();

      api.transpose (current_tinputs, current_inputs);
      //PRINT_CUMATRIX_CARRAY(current_tinputs);
      //PRINT_CUMATRIX_CARRAY(current_inputs);

      api.tensorProduct (current_weights1, current_tinputs, next_errors_wb);
      //PRINT_CUMATRIX_CARRAY(current_weights1);
      //PRINT_CUMATRIX_CARRAY(current_tinputs);
      //PRINT_CUMATRIX_CARRAY(next_errors);

      api.add (current_weights2, current_weights2, current_weights1);
      //PRINT_CUMATRIX_CARRAY(current_weights2);
      //PRINT_CUMATRIX_CARRAY(current_weights2);
      //PRINT_CUMATRIX_CARRAY(current_weights1);
    }
  };
#endif
  auto calcNablaWeights = [&layers, &api2]()
  {
    LayerT* current = nullptr;
    LayerT* next = layers[0];

    for (uintt idx = 1; idx < layers.size(); ++idx)
    {
      current = next;
      next = layers[idx];
  
      const uintt fpMatricesCount = current->getFPMatricesCount();

      BPMatrices& current_bp = *current->getBPMatrices ();
      for (uintt fpidx = 0; fpidx < fpMatricesCount; ++fpidx)
      {
        FPMatrices& current_fp = *current->getFPMatrices (fpidx);
        FPMatrices& next_fp = *next->getFPMatrices (fpidx);

        api2.transpose (current_bp.m_tinputs, current_fp.m_inputs);
        {
          oap::generic::Dim32 dims
          {{
            {current->getTotalNeuronsCount(), next->getNeuronsCount()},
            {current->getTotalNeuronsCount(), 1},
            {1, next->getNeuronsCount()},
          }};
          api2.tensorProduct (current_bp.m_weights1, current_bp.m_tinputs, next_fp.m_errors, dims);
        }
        api2.add (current_bp.m_weights2, current_bp.m_weights2, current_bp.m_weights1);
      }
    }
  };

  LayerT* current = layers.back ();

  uintt fpcount = current->getFPMatricesCount();
  for (uintt fpidx = 0; fpidx < fpcount; ++fpidx)
  {
    FPMatrices& current_fp = *current->getFPMatrices (fpidx);
    copyMatrixToMatrix (current_fp.m_errors, current_fp.m_errorsAux);
  }

  calcErrors ();

  calcNablaWeights ();
}

template<typename LayerT, typename Layers, typename Api>
void updateWeights(const Layers& layers, Api& api, floatt learningRate, uintt normalizationFactor)
{
  logTrace("");
  LayerT* current = nullptr;
  LayerT* next = layers[0];

  for (uintt idx = 1; idx < layers.size(); ++idx)
  {
    current = next;
    next = layers[idx];

    floatt lr = learningRate / static_cast<floatt>(normalizationFactor);
    api.multiplyReConstant (current->getBPMatrices()->m_weights2, current->getBPMatrices()->m_weights2, lr);
    api.subtract (current->getBPMatrices()->m_weights, current->getBPMatrices()->m_weights, current->getBPMatrices()->m_weights2);
  }
}

template<typename LayerT>
math::ComplexMatrix* allocateCommonErrMatrix (const LayerT& layerRef, uintt samplesCount1, uintt samplesCount2, oap::NetworkGenericApi* nga)
{
  logAssert (samplesCount1 > 0);
  logAssert (samplesCount2 > 0);
  const uintt unitsCountWithBiases = layerRef.getTotalNeuronsCount ();
  const uintt unitsCount = layerRef.getNeuronsCount ();

  auto matrixInfo = math::MatrixInfo (true, false, 3 * samplesCount1, unitsCountWithBiases * samplesCount2);
  return nga->newKernelMatrixFromMatrixInfo (matrixInfo);
}

template<typename LayerT>
void allocateFPMatrices (FPMatrices& fp, const LayerT& layerRef, uintt samplesCount, oap::NetworkGenericApi* nga)
{
  logTrace("");
  logTraceS ("%s %p", __func__, &fp);

  const uintt unitsCountWithBiases = layerRef.getTotalNeuronsCount ();
  const uintt unitsCount = layerRef.getNeuronsCount ();

  fp.m_matricesInfo = math::MatrixInfo (true, false, 1, unitsCountWithBiases * samplesCount);
  fp.m_matricesInfo_wb = math::MatrixInfo (true, false, 1, unitsCount * samplesCount);
  const math::MatrixDim mdim = {fp.m_matricesInfo.columns(), fp.m_matricesInfo.rows()};
  const math::MatrixDim mdim_wb = {fp.m_matricesInfo_wb.columns(), fp.m_matricesInfo_wb.rows()};

  fp.m_inputs = nga->newKernelMatrixFromMatrixInfo (fp.m_matricesInfo);
  fp.m_inputs_wb = nga->newKernelSharedSubMatrix (mdim_wb, fp.m_inputs);
  fp.m_sums = nga->newKernelMatrixFromMatrixInfo (fp.m_matricesInfo);
  fp.m_sums_wb = nga->newKernelSharedSubMatrix (mdim_wb, fp.m_sums);

  //math::Matrix* errorsMatrix = getErrorsMatrix ();
  //const math::MatrixLoc loc = getErrorsMatrixLoc ();
  //const math::MatrixLoc loc1 = getErrorsMatrixLoc ();
  //const math::MatrixLoc loc2 = getErrorsMatrixLoc ();

  //if (errorsMatrix != nullptr)
  //{
  //  fp.m_errors = nga->newKernelSharedSubMatrix (loc, mdim, errorsMatrix);
  //  fp.m_errorsAux = nga->newKernelSharedSubMatrix (loc1, mdim, errorsMatrix);
  //  fp.m_errorsAcc = nga->newKernelSharedSubMatrix (loc2, mdim, errorsMatrix);
  //}
  //else
  //{
    fp.m_errors = nga->newKernelMatrixFromMatrixInfo (fp.m_matricesInfo);
    fp.m_errorsAux = nga->newKernelMatrixFromMatrixInfo (fp.m_matricesInfo);
    fp.m_errorsAcc = nga->newKernelMatrixFromMatrixInfo (fp.m_matricesInfo);
  //}

  fp.m_errors_wb = nga->newKernelSharedSubMatrix (mdim_wb, fp.m_errors);

  logTrace ("minfo = %s", std::to_string(fp.m_matricesInfo).c_str());
  logTrace ("fp.m_inputs = %p", fp.m_inputs);
  logTrace ("fp.m_inputs_wb = %p", fp.m_inputs_wb);
  logTrace ("fp.m_sums = %p", fp.m_sums);
  logTrace ("fp.m_sums_wb = %p", fp.m_sums_wb);
  logTrace ("fp.m_errors = %p", fp.m_errors);
  logTrace ("fp.m_errorsAux = %p", fp.m_errorsAux);
  logTrace ("fp.m_errorsAcc = %p", fp.m_errorsAcc);
  logTraceE ("%s %p", __func__, &fp);
}

template<typename LayerT>
void allocateSharedFPMatrices (FPMatrices& fp, const LayerT& layerRef, FPMatrices* orig, oap::NetworkGenericApi* nga)
{
  logTrace("");
  logTraceS ("%s %p", __func__, &fp);
  const uintt samplesCount = 1;

  const uintt unitsCountWithBiases = layerRef.getTotalNeuronsCount ();
  const uintt unitsCount = layerRef.getNeuronsCount ();

  const auto noneMemory = oap::common::OAP_NONE_MEMORY();

  fp.m_matricesInfo = math::MatrixInfo (true, false, 1, unitsCountWithBiases * samplesCount);
  fp.m_matricesInfo_wb = math::MatrixInfo (true, false, 1, unitsCount * samplesCount);
  const math::MatrixDim mdim = {fp.m_matricesInfo.columns(), fp.m_matricesInfo.rows()};
  const math::MatrixDim mdim_wb = {fp.m_matricesInfo_wb.columns(), fp.m_matricesInfo_wb.rows()};

  fp.m_inputs = nga->newKernelSharedSubMatrix (mdim, orig->m_inputs);
  fp.m_inputs_wb = nga->newKernelSharedSubMatrix (mdim_wb, fp.m_inputs);
  fp.m_sums = nga->newKernelSharedSubMatrix (mdim, orig->m_sums);
  fp.m_sums_wb = nga->newKernelSharedSubMatrix (mdim_wb, fp.m_sums);

  fp.m_errors = nga->newKernelSharedSubMatrix (mdim, orig->m_errors);
  fp.m_errors_wb = nga->newKernelSharedSubMatrix (mdim_wb, fp.m_errors);
  fp.m_errorsAux = nga->newKernelSharedSubMatrix (mdim, orig->m_errorsAux);
  fp.m_errorsAcc = nga->newKernelSharedSubMatrix (mdim, orig->m_errorsAcc);

  logTrace ("minfo = %s", std::to_string(fp.m_matricesInfo).c_str());
  logTrace ("fp.m_inputs = %p", fp.m_inputs);
  logTrace ("fp.m_inputs_wb = %p", fp.m_inputs_wb);
  logTrace ("fp.m_sums = %p", fp.m_sums);
  logTrace ("fp.m_sums_wb = %p", fp.m_sums_wb);
  logTrace ("fp.m_errors = %p", fp.m_errors);
  logTrace ("fp.m_errorsAux = %p", fp.m_errorsAux);
  logTrace ("fp.m_errorsAcc = %p", fp.m_errorsAcc);
  logTraceE ("%s %p", __func__, &fp);
}

template<typename LayerT>
FPMatrices* allocateFPMatrices (const LayerT& layerRef, uintt samplesCount, oap::NetworkGenericApi* nga)
{
  FPMatrices* fpMatrices = new FPMatrices ();
  allocateFPMatrices (*fpMatrices, layerRef, samplesCount, nga);
  return fpMatrices;
}

template<typename LayerT>
FPMatrices* allocateSharedFPMatrices (const LayerT& layerRef, FPMatrices* orig, oap::NetworkGenericApi* nga)
{
  FPMatrices* fpMatrices = new FPMatrices ();
  allocateSharedFPMatrices (*fpMatrices, layerRef, orig, nga);
  return fpMatrices;
}

inline void deallocateFPMatrices (FPMatrices* fp, oap::NetworkGenericApi* nga)
{
  logTrace("");
  logTrace ("%s %p", __func__, &fp);

  auto delk = [&nga](math::ComplexMatrix** matrix)
  {
    if (matrix != nullptr)
    {
      nga->deleteKernelMatrix (*matrix);
      matrix = nullptr;
    }
  };

  delk (&fp->m_inputs);
  delk (&fp->m_inputs_wb);
  delk (&fp->m_sums);
  delk (&fp->m_sums_wb);
  delk (&fp->m_errors);
  delk (&fp->m_errors_wb);
  delk (&fp->m_errorsAcc);
  delk (&fp->m_errorsAux);

  oap::host::DeleteMatrix (fp->m_errorsHost);
  delete fp;
}

inline void deallocateBPMatrices (BPMatrices* bp, oap::NetworkGenericApi* nga)
{
  logTrace("");
  logTraceS ("%s %p", __func__, &bp);

  auto delk = [&nga](math::ComplexMatrix** matrix)
  {
    if (matrix != nullptr)
    {
      nga->deleteKernelMatrix (*matrix);
      matrix = nullptr;
    }
  };

  delk (&bp->m_tinputs);
  delk (&bp->m_weights);
  delk (&bp->m_tweights);
  delk (&bp->m_weights1);
  delk (&bp->m_weights2);
  delete bp;
  logTraceE ("%s %p", __func__, &bp);
}

template<typename LayerT>
void deallocateFPMatricesInLayer (LayerT& layer, oap::NetworkGenericApi* nga)
{
  logTrace("");
  if (layer.getFPMatrices() == nullptr)
  {
    return;
  }
  deallocateFPMatrices (*layer.getFPMatrices(), nga);
}

template<typename LayerT>
void deallocateBPMatricesInLayer (LayerT& layer, oap::NetworkGenericApi* nga)
{
  logTrace("");
  if (layer.getBPMatrices() == nullptr)
  {
    return;
  }
  deallocateBPMatrices (*layer.getBPMatrices(), nga);
}

inline void allocateBPMatrices (BPMatrices& bp, const NBPair& neuronsCount1, const NBPair& neuronsCount2, oap::NetworkGenericApi* nga)
{
  logTrace("");
  logTraceS ("%s %p", __func__, &bp);
  const uintt cUCount = neuronsCount1.first + neuronsCount1.second;
  const uintt nUCount = neuronsCount2.first;

  math::MatrixInfo tinputsInfo (true, false, cUCount, 1);
  bp.m_tinputs = nga->newKernelMatrixFromMatrixInfo (tinputsInfo); //todo: use transpose

  math::MatrixInfo weightsInfo (true, false, cUCount, nUCount);
  bp.m_weights = nga->newKernelMatrixFromMatrixInfo (weightsInfo);
  bp.m_weights1 = nga->newKernelMatrixFromMatrixInfo (weightsInfo);
  bp.m_weights2 = nga->newKernelMatrixFromMatrixInfo (weightsInfo);

  math::MatrixInfo tweightsInfo (true, false, nUCount, cUCount);
  bp.m_tweights = nga->newKernelMatrixFromMatrixInfo (tweightsInfo);

  logTrace ("bp.m_tinputs = %p", bp.m_tinputs);
  logTrace ("bp.m_weights = %p", bp.m_weights);
  logTrace ("bp.m_weights1 = %p", bp.m_weights1);
  logTrace ("bp.m_weights2 = %p", bp.m_weights2);
  logTrace ("bp.m_tweights = %p", bp.m_tweights);
  logTraceE ("%s %p", __func__, &bp);
}

inline BPMatrices* allocateBPMatrices (const NBPair& neuronsCount1, const NBPair& neuronsCount2, oap::NetworkGenericApi* nga)
{
  BPMatrices* bpMatrices = new BPMatrices ();
  allocateBPMatrices (*bpMatrices, neuronsCount1, neuronsCount2, nga);
  return bpMatrices;
}

template<typename LayerT>
void allocateBPMatrices (BPMatrices& bp, const LayerT& layer, const LayerT& nextLayer)
{
  allocateBPMatrices (bp, layer.getNBPair(), nextLayer.getNBPair());
}

template<typename LayerT>
BPMatrices* allocateBPMatrices (const LayerT& layer, const LayerT& nextLayer, oap::NetworkGenericApi* nga)
{
  return allocateBPMatrices (layer.getNBPair(), nextLayer.getNBPair(), nga);
}

template<typename LayerT>
void deallocate (LayerT& layer, oap::NetworkGenericApi* nga)
{
  logTrace("");
  deallocateFPMatricesInLayer (layer, nga);
  deallocateBPMatricesInLayer (layer, nga);
}

template<typename AllocNeuronsApi, typename LayerT>
void createFPMatrices (LayerT& layer)
{
  logTrace("");
  FPMatrices* fpMatrices = new FPMatrices();

  layer.addFPMatrices (fpMatrices);
  allocateFPMatrices<AllocNeuronsApi> (*layer.getFPMatrices(), layer, layer.getSamplesCount());
}

template<typename LayerT>
void createBPMatrices (LayerT& layer, LayerT& nextLayer, oap::NetworkGenericApi* nga)
{
  logTrace("");
  BPMatrices* bpMatrices = allocateBPMatrices (layer, nextLayer, nga);
  layer.addBPMatrices (bpMatrices);
}

template<typename LayerT>
LayerT* createLayer (uintt neurons, bool hasBias, uintt samplesCount, Activation activation)
{
  logTrace("");
  LayerT* layer = new LayerT (neurons, hasBias ? 1 : 0, samplesCount, activation);

  debugInfo ("Layer %p allocates %u neurons (neurons : %u, bias : %u)", layer, layer->getTotalNeuronsCount(), layer->getNeuronsCount(), layer->getBiasesCount());

  return layer;
}

template<typename LayerT>
void connectLayers (LayerT* previous, LayerT* next, oap::NetworkGenericApi* nga)
{
  logTrace("");
  previous->setNextLayer (next);
  oap::generic::createBPMatrices (*previous, *next, nga);
}

template<typename LayerT, typename CopyHostMatrixToMatrix, typename GetMatrixInfo>
void setHostWeights (LayerT& layer, math::ComplexMatrix* weights, CopyHostMatrixToMatrix&& copyHostMatrixToMatrix, GetMatrixInfo&& getLayerMatrixInfo, GetMatrixInfo&& getArgMatrixInfo)
{
  logTrace("");
  auto linfo = getLayerMatrixInfo (layer.getBPMatrices()->m_weights);
  auto ainfo = getArgMatrixInfo (weights);

  debugAssert (linfo.columns() == ainfo.columns() && linfo.rows() == ainfo.rows());

  copyHostMatrixToMatrix (layer.getBPMatrices()->m_weights, weights);
}

template<typename LayerT, typename GetMatrixInfo>
math::MatrixInfo getOutputsInfo (const LayerT& layer, GetMatrixInfo&& getMatrixInfo)
{
  logTrace("");
  return getMatrixInfo (layer.getFPMatrices()->m_inputs);
}

template<typename LayerT, typename CopyMatrixToMatrix>
void getOutputs (math::ComplexMatrix* matrix, const LayerT& layer, CopyMatrixToMatrix&& copyMatrixToMatrix)
{
  logTrace("");
  copyMatrixToMatrix (matrix, layer.getFPMatrices()->m_inputs);
}

template<typename LayerT>
math::ComplexMatrix* getWeights (const LayerT& layer)
{
  debugAssert (layer.getBPMatrices()->m_weights != nullptr);
  return layer.getBPMatrices()->m_weights;
}

template<typename LayerT, typename GetMatrixInfo>
math::MatrixInfo getWeightsInfo (const LayerT& layer, GetMatrixInfo&& getMatrixInfo)
{
  math::ComplexMatrix* weights = getWeights (layer);
  return getMatrixInfo (weights);
}

template<typename LayerT, typename CopyHostMatrixToKernelMatrix>
void setWeights (const LayerT& layer, const math::ComplexMatrix* hmatrix, CopyHostMatrixToKernelMatrix&& copyHostMatrixToKernelMatrix)
{
  math::ComplexMatrix* weights = getWeights (layer);
  copyHostMatrixToKernelMatrix (weights, hmatrix);
}

}
}
#endif
