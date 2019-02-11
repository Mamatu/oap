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

  enum MatrixType
  {
    HOST,
    DEVICE
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

  Layer* createLayer (size_t neurons, bool bias = false);

  oap::HostMatrixUPtr run (math::Matrix* hostInputs, MatrixType argsType, oap::ErrorType errorType);
  void train (math::Matrix* hostInputs, math::Matrix* expectedHostOutputs, MatrixType argsType, oap::ErrorType errorType);

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

protected:
  void setHostInputs (math::Matrix* inputs, size_t layerIndex);

  void backwardPropagation (math::Matrix* deviceExpected, oap::ErrorType errorType);

  void forwardPropagation ();

private:
  std::vector<Layer*> m_layers;

  void destroyLayers();

  void updateWeights();

  inline void calculateError();

  bool shouldContinue (oap::ErrorType errorType);

  oap::CuProceduresApi m_cuApi;

  floatt m_learningRate = 0.1f;
  floatt m_serror = 0;
  size_t m_step = 1;

  oap::DeviceMatrixPtr m_expectedDeviceOutputs = nullptr;
  IController* m_icontroller = nullptr;
};

#endif
