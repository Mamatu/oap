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

#include "oapLayer.h"
#include "oapDeviceMatrixPtr.h"

class Network
{
  oap::CuProceduresApi m_cuApi;
  floatt m_learningRate;
        
public:
  enum MatrixType
  {
    HOST,
    DEVICE
  };

  Network();

  class IController
  {
    public:
      virtual ~IController() {}
      virtual bool shouldCalculateError(size_t step) = 0;
      virtual void setSquareError (floatt sqe) = 0;
      virtual bool shouldContinue() = 0;
  };

  virtual ~Network();

  Layer* createLayer(size_t neurons, bool bias = false);

  void runTest (math::Matrix* hostInputs, math::Matrix* expectedHostOutputs, MatrixType argsType);
  oap::HostMatrixUPtr run (math::Matrix* hostInputs, MatrixType argsType);

  void setController(IController* icontroller);

  void setHostWeights (math::Matrix* weights, size_t layerIndex);

  void getHostWeights (math::Matrix* weights, size_t layerIndex);

  void setDeviceWeights (math::Matrix* weights, size_t layerIndex);

  void setLearningRate (floatt lr);

  void save(const std::string& filepath);

  void load(const std::string& filepath);

protected:
  enum class AlgoType
  {
    TEST_MODE,
    NORMAL_MODE
  };

  void setHostInputs (math::Matrix* inputs, size_t layerIndex);

  void executeLearning(math::Matrix* deviceExpected);

  oap::HostMatrixUPtr executeAlgo(AlgoType algoType, math::Matrix* deviceExpected);

private:
  std::vector<Layer*> m_layers;

  Layer* getLayer(size_t layerIndex) const;

  void destroyLayers();

  void updateWeights();

  inline void calculateError();

  bool shouldContinue();

  floatt m_serror;

  oap::DeviceMatrixPtr m_expectedDeviceOutputs;

  IController* m_icontroller;

  size_t m_step;
};

#endif
