/*
 * Copyright 2016 - 2018 Marcin Matula
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

#ifndef OAP_NEURAL_NETWORK_H
#define OAP_NEURAL_NETWORK_H

#include "NeuralTypes.h"

#include "oapLayer.h"
#include "oapDeviceMatrixPtr.h"

class Network
{
public: // types

  enum Type
  {
    HOST,
    DEVICE,
    DEVICE_COPY,
  };

  class IController
  {
    public:
      virtual ~IController() {}
      virtual bool shouldCalculateError(size_t step) = 0;
      virtual void setError (floatt value, oap::ErrorType type) = 0;
      virtual bool shouldContinue() = 0;
  };

public:

  Network();
  virtual ~Network();

  Network (const Network&) = delete;
  Network (Network&&) = delete;
  Network& operator= (const Network&) = delete;
  Network& operator= (Network&&) = delete;

  Layer* createLayer (size_t neurons, const Activation& activation = Activation::SIGMOID);

  oap::HostMatrixUPtr run (math::Matrix* hostInputs, Type argsType, oap::ErrorType errorType);

  void setInputs (math::Matrix* inputs, Type argsType);
  void setExpected (math::Matrix* expected, Type argsType);

  math::Matrix* getOutputs (math::Matrix* outputs, Type argsType) const;
  math::Matrix* getHostOutputs () const;

  void forwardPropagation ();
  void calculateErrors (oap::ErrorType errorType);

  math::Matrix* getErrors (Type type) const;

  floatt calculateMSE ();
  floatt calculateRMSE ();
  floatt calculateSum ();
  floatt calculateSumMean ();
  floatt calculateCrossEntropy ();

  floatt calculateError (oap::ErrorType errorType);

  void backwardPropagation ();

  bool train (math::Matrix* hostInputs, math::Matrix* expectedHostOutputs, Type argsType, oap::ErrorType errorType);

  void createLevel (Layer* layer);

  void addLayer (Layer* layer);

  void setController (IController* icontroller);

  void setHostWeights (math::Matrix* weights, size_t layerIndex);

  void getHostWeights (math::Matrix* weights, size_t layerIndex);

  void setDeviceWeights (math::Matrix* weights, size_t layerIndex);

  void setLearningRate (floatt lr);
  floatt getLearningRate () const;

  void save (utils::ByteBuffer& buffer) const;

  static Network* load (const utils::ByteBuffer& buffer);

  size_t getLayersCount () const
  {
    return m_layers.size ();
  }
  
  Layer* getLayer(size_t layerIndex) const;

  bool operator== (const Network& network) const;
  bool operator!= (const Network& network) const;

  void printLayersWeights ();

protected:
  void setHostInputs (math::Matrix* inputs, size_t layerIndex);

  inline void transpose (math::Matrix* m_tinputs, math::Matrix* m_inputs)
  {
    m_cuApi.transpose (m_tinputs, m_inputs);
  }

  inline void tensorProduct (math::Matrix* m_weights1, math::Matrix* m_tinputs, math::Matrix* next_errors)
  {
    m_cuApi.tensorProduct (m_weights1, m_tinputs, next_errors);
  }

  inline void multiplyReConstant (math::Matrix* m_weights1Output, math::Matrix* m_weights1Input, floatt m_learningRate)
  {
    m_cuApi.multiplyReConstant (m_weights1Output, m_weights1Input, m_learningRate);
  }

  inline void sigmoidDerivative (math::Matrix* next_sums, math::Matrix* next_sums1)
  {
    m_cuApi.sigmoidDerivative (next_sums, next_sums1);
  }

  inline void hadamardProductVec (math::Matrix* m_weights2, math::Matrix* m_weights1, math::Matrix* next_sums)
  {
    m_cuApi.hadamardProductVec (m_weights2, m_weights1, next_sums);
  }

  inline void add (math::Matrix* m_weightsOutput, math::Matrix* m_weightsInput, math::Matrix* m_weights2)
  {
    m_cuApi.add (m_weightsOutput, m_weightsInput, m_weights2);
  }

  inline void activateFunc (math::Matrix* output, math::Matrix* input, const Activation& activation)
  {
    switch (activation)
    {
      case Activation::SIGMOID:
        m_cuApi.sigmoid (output, input);
      break;
      case Activation::IDENTITY:
        m_cuApi.identity (output, input);
      break;
      case Activation::TANH:
        m_cuApi.tanh (output, input);
      break;
      case Activation::SIN:
        m_cuApi.sin (output, input);
      break;
    };
  }

  inline void derivativeFunc (math::Matrix* output, math::Matrix* input, const Activation& activation)
  {
    switch (activation)
    {
      case Activation::SIGMOID:
        m_cuApi.sigmoidDerivative (output, input);
      break;
      case Activation::IDENTITY:
        m_cuApi.identityDerivative (output, input);
      break;
      case Activation::TANH:
        m_cuApi.tanhDerivative (output, input);
      break;
      case Activation::SIN:
        m_cuApi.sinDerivative (output, input);
      break;
    };
  }

private:
  std::vector<Layer*> m_layers;

  void destroyLayers();

  void updateWeights();

  inline void calculateError();

  bool shouldContinue (oap::ErrorType errorType);

  oap::CuProceduresApi m_cuApi;

  floatt m_learningRate = 0.1f;
  size_t m_step = 1;

  oap::DeviceMatrixPtr m_expectedDeviceOutputs = nullptr;
  IController* m_icontroller = nullptr;

  std::ostream& log()
  {
    //std::cout << "[network] ";
    return std::cout;
  }
};

#endif
