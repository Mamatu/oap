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

#include <string>
#include "gtest/gtest.h"
#include "CuProceduresApi.h"
#include "KernelExecutor.h"
#include "MatchersUtils.h"
#include "MathOperationsCpu.h"

#include "oapCudaMatrixUtils.h"
#include "oapHostMatrixUtils.h"
#include "oapNetwork.h"
#include "oapFunctions.h"
#include "PyPlot.h"
#include "Config.h"

namespace
{
class NetworkT : public Network
{
  public:
    void setHostInput (math::Matrix* inputs, size_t index)
    {
      Network::setHostInputs (inputs, index);
    }
};
}

class OapNeuralTests_SimpleForwardPropagation : public testing::Test
{
 public:
  CUresult status;
  NetworkT* network;

  virtual void SetUp()
  {
    oap::cuda::Context::Instance().create();
    network = nullptr;
    network = new NetworkT();
  }

  virtual void TearDown()
  {
    delete network;
    network = nullptr;
    oap::cuda::Context::Instance().destroy();
  }
};

TEST_F(OapNeuralTests_SimpleForwardPropagation, SimpleForwardPropagation_1)
{
  using namespace oap::math;

  DeviceLayer* l1 = network->createLayer(2);
  DeviceLayer* l2 = network->createLayer(2);
  DeviceLayer* l3 = network->createLayer(1);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 2);
  weights1to2->reValues[0] = 1;
  weights1to2->reValues[2] = 1;

  weights1to2->reValues[1] = 1;
  weights1to2->reValues[3] = 1;

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (2, 1);
  weights2to3->reValues[0] = 1;
  weights2to3->reValues[1] = 1;

  l1->setHostWeights (weights1to2);
  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 2);
  inputs->reValues[0] = 1;
  inputs->reValues[1] = 1;

  network->setInputs (inputs, ArgType::HOST);

  network->forwardPropagation ();

  auto minfo = l3->getOutputsInfo ();
  oap::HostMatrixPtr outputsL3 = oap::host::NewReMatrix (minfo.m_matrixDim.columns, minfo.m_matrixDim.rows);
  l3->getOutputs (outputsL3, ArgType::HOST);

  EXPECT_DOUBLE_EQ (sigmoid (sigmoid(2) + sigmoid(2)), outputsL3->reValues[0]);
}

TEST_F(OapNeuralTests_SimpleForwardPropagation, SimpleForwardPropagation_2)
{
  using namespace oap::math;

  DeviceLayer* l1 = network->createLayer(3);
  DeviceLayer* l2 = network->createLayer(3);
  DeviceLayer* l3 = network->createLayer(1);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (3, 3);
  weights1to2->reValues[0] = 1;
  weights1to2->reValues[3] = 1;
  weights1to2->reValues[6] = 1;

  weights1to2->reValues[1] = 1;
  weights1to2->reValues[4] = 1;
  weights1to2->reValues[7] = 1;

  weights1to2->reValues[2] = 1;
  weights1to2->reValues[5] = 1;
  weights1to2->reValues[8] = 1;

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);
  weights2to3->reValues[0] = 1;
  weights2to3->reValues[1] = 1;
  weights2to3->reValues[2] = 1;

  l1->setHostWeights (weights1to2);
  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 3);
  inputs->reValues[0] = 1;
  inputs->reValues[1] = 1;
  inputs->reValues[2] = 1;

  network->setInputs (inputs, ArgType::HOST);

  network->forwardPropagation ();

  auto minfo = l3->getOutputsInfo ();
  oap::HostMatrixPtr outputsL3 = oap::host::NewReMatrix (minfo.m_matrixDim.columns, minfo.m_matrixDim.rows);
  l3->getOutputs (outputsL3, ArgType::HOST);

  EXPECT_DOUBLE_EQ (sigmoid (sigmoid(3) + sigmoid(3) + sigmoid(3)), outputsL3->reValues[0]);
}

TEST_F(OapNeuralTests_SimpleForwardPropagation, SimpleForwardPropagation_3)
{
  using namespace oap::math;

  DeviceLayer* l1 = network->createLayer(3);
  DeviceLayer* l2 = network->createLayer(3);
  DeviceLayer* l3 = network->createLayer(1);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (3, 3);
  weights1to2->reValues[0] = 2;
  weights1to2->reValues[3] = 2;
  weights1to2->reValues[6] = 2;

  weights1to2->reValues[1] = 1;
  weights1to2->reValues[4] = 1;
  weights1to2->reValues[7] = 1;

  weights1to2->reValues[2] = 1;
  weights1to2->reValues[5] = 1;
  weights1to2->reValues[8] = 1;

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);
  weights2to3->reValues[0] = 1;
  weights2to3->reValues[1] = 1;
  weights2to3->reValues[2] = 1;

  l1->setHostWeights (weights1to2);
  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 3);
  inputs->reValues[0] = 1;
  inputs->reValues[1] = 1;
  inputs->reValues[2] = 1;

  network->setInputs (inputs, ArgType::HOST);

  network->forwardPropagation ();

  auto minfo = l3->getOutputsInfo ();
  oap::HostMatrixPtr outputsL3 = oap::host::NewReMatrix (minfo.m_matrixDim.columns, minfo.m_matrixDim.rows);
  l3->getOutputs (outputsL3, ArgType::HOST);

  EXPECT_DOUBLE_EQ (sigmoid (sigmoid(4) + sigmoid(4) + sigmoid(4)), outputsL3->reValues[0]);
}

TEST_F(OapNeuralTests_SimpleForwardPropagation, SimpleForwardPropagation_4)
{
  using namespace oap::math;

  DeviceLayer* l1 = network->createLayer(3);
  DeviceLayer* l2 = network->createLayer(3);
  DeviceLayer* l3 = network->createLayer(1);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (3, 3);
  weights1to2->reValues[0] = 4;
  weights1to2->reValues[3] = 3;
  weights1to2->reValues[6] = 2;

  weights1to2->reValues[1] = 1;
  weights1to2->reValues[4] = 1;
  weights1to2->reValues[7] = 1;

  weights1to2->reValues[2] = 1;
  weights1to2->reValues[5] = 1;
  weights1to2->reValues[8] = 1;

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);
  weights2to3->reValues[0] = 1;
  weights2to3->reValues[1] = 1;
  weights2to3->reValues[2] = 1;

  l1->setHostWeights (weights1to2);
  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 3);
  inputs->reValues[0] = 1;
  inputs->reValues[1] = 1;
  inputs->reValues[2] = 1;

  network->setInputs (inputs, ArgType::HOST);

  network->forwardPropagation ();

  auto getLayerOutput = [](DeviceLayer* layer)
  {
    auto minfo = layer->getOutputsInfo ();
    oap::HostMatrixPtr outputsL = oap::host::NewReMatrix (minfo.m_matrixDim.columns, minfo.m_matrixDim.rows);
    layer->getOutputs (outputsL, ArgType::HOST);
    return outputsL;
  };

  auto outputsL2 = getLayerOutput(l2);
  auto outputsL3 = getLayerOutput(l3);

  logInfo ("FP o2 %p %s", l2, oap::host::to_string(outputsL2).c_str());
  EXPECT_DOUBLE_EQ (sigmoid(6), outputsL2->reValues[0]);
  EXPECT_DOUBLE_EQ (sigmoid(5), outputsL2->reValues[1]);
  EXPECT_DOUBLE_EQ (sigmoid(4), outputsL2->reValues[2]);
  EXPECT_DOUBLE_EQ (sigmoid (sigmoid(6) + sigmoid(5) + sigmoid(4)), outputsL3->reValues[0]);
}
