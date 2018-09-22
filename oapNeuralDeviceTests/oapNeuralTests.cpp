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

#include <string>
#include "gtest/gtest.h"
#include "CuProceduresApi.h"
#include "KernelExecutor.h"
#include "MatchersUtils.h"
#include "MathOperationsCpu.h"

#include "oapCudaMatrixUtils.h"
#include "oapHostMatrixUtils.h"
#include "oapNetwork.h"

class NetworkT : public Network
{
  public:
    void executeLearning(math::Matrix* expected)
    {
      Network::executeLearning (expected);
    }

    void setHostInput (math::Matrix* inputs, size_t index)
    {
      Network::setHostInputs (inputs, index);
    }

    oap::HostMatrixUPtr executeTest(math::Matrix* deviceExpected)
    {
      return Network::executeAlgo(Network::AlgoType::TEST_MODE, deviceExpected);
    }
};

class OapNeuralTests : public testing::Test
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

  class Runner
  {
    bool m_hasBias;
    OapNeuralTests* m_ont;
    floatt m_bvalue;
  public:
    Runner(bool hasBias, OapNeuralTests* ont, floatt bvalue = 1.f): m_hasBias(hasBias), m_ont(ont), m_bvalue(bvalue)
    {}

    void runTest(floatt a1, floatt a2, floatt e1)
    {
      size_t neurons = 2;

      if (m_hasBias)
      {
        neurons = neurons + 1;
      }

      oap::HostMatrixUPtr inputs = oap::host::NewReMatrix(1, neurons);
      oap::HostMatrixUPtr expected = oap::host::NewReMatrix(1, 1);
      inputs->reValues[0] = a1;
      inputs->reValues[1] = a2;

      if (m_hasBias)
      {
        inputs->reValues[2] = m_bvalue;
      }

      expected->reValues[0] = e1;

      m_ont->network->runHostArgsTest(inputs, expected);
    }

    floatt run(floatt a1, floatt a2)
    {
      size_t neurons = 2;

      if (m_hasBias)
      {
        neurons = neurons + 1;
      }

      oap::HostMatrixUPtr inputs = oap::host::NewReMatrix(1, neurons);
      inputs->reValues[0] = a1;
      inputs->reValues[1] = a2;

      if (m_hasBias)
      {
        inputs->reValues[2] = m_bvalue;
      }

      auto output = m_ont->network->runHostArgs(inputs);
      return m_ont->is(output->reValues[0]);
    }
  };

  floatt sigmoid(floatt x)
  {
    return 1.f / (1.f + exp (-x));
  }

  floatt dsigmoid(floatt x)
  {
    return sigmoid(x) * (1.f - sigmoid(x));
  }

  floatt is(floatt a)
  {
    debug("arg is %f", a);
    if (a > 0.5f)
    {
      return 1;
    }
    return 0;
  }
};

TEST_F(OapNeuralTests, LogicalOr)
{
  Layer* l1 = network->createLayer(2);
  network->createLayer(1);

  network->setLearningRate (1);

  Runner r(false, this);

  r.runTest(1, 1, 1);
  l1->printHostWeights();
  r.runTest(1, 0, 1);
  l1->printHostWeights();
  r.runTest(0, 1, 1);
  l1->printHostWeights();
  r.runTest(0, 0, 0);
  l1->printHostWeights();

  EXPECT_EQ(1, r.run(1, 1));
  EXPECT_EQ(1, r.run(1, 0));
  EXPECT_EQ(0, r.run(0, 0));
  EXPECT_EQ(1, r.run(1, 0));
}

TEST_F(OapNeuralTests, LogicalAnd)
{
  bool isbias = true;

  Layer* l1 = network->createLayer(2, isbias);
  network->createLayer(1);

  Runner r(isbias, this, 1);
  network->setLearningRate (0.01);

  for (size_t value = 1; value < 10000; ++value)
  {
    floatt fvalue = static_cast<floatt>(1);
    r.runTest(fvalue, fvalue, 1);
    l1->printHostWeights();
    r.runTest(fvalue, 0, 0);
    l1->printHostWeights();
    r.runTest(0, fvalue, 0);
    l1->printHostWeights();
    r.runTest(0, 0, 0);
    l1->printHostWeights();
  }

  EXPECT_EQ(1, r.run(1, 1));
  EXPECT_EQ(0, r.run(1, 0));
  EXPECT_EQ(0, r.run(0, 0));
  EXPECT_EQ(0, r.run(0, 1));
}

