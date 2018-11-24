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
#include "Controllers.h"

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
  size_t m_learningSteps;

  virtual void SetUp()
  {
    oap::cuda::Context::Instance().create();
    network = new NetworkT();
    m_learningSteps = 100000;
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

      m_ont->network->runTest(inputs, expected, Network::HOST);
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

      auto output = m_ont->network->run (inputs, Network::HOST);
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

TEST_F(OapNeuralTests, LogicalOr_Binary)
{
  Layer* l1 = network->createLayer(2);
  network->createLayer(1);

  network->setLearningRate (0.01);
  
  std::shared_ptr<SquareErrorLimitController> controller = std::make_shared<SquareErrorLimitController>(0.062, 4);
  network->setController (controller.get());
  
  Runner r(false, this);

  for (size_t idx = 0; idx < m_learningSteps && controller->shouldContinue(); ++idx)
  {
    r.runTest(1, 1, 1);
    l1->printHostWeights();
    r.runTest(1, 0, 1);
    l1->printHostWeights();
    r.runTest(0, 1, 1);
    l1->printHostWeights();
    r.runTest(0, 0, 0);
    l1->printHostWeights();
  }

  EXPECT_EQ(1, r.run(1, 1));
  EXPECT_EQ(1, r.run(1, 0));
  EXPECT_EQ(0, r.run(0, 0));
  EXPECT_EQ(1, r.run(1, 0));
}

TEST_F(OapNeuralTests, LogicalAnd_Binary)
{
  bool isbias = true;

  Layer* l1 = network->createLayer(2, isbias);
  network->createLayer(1);

  Runner r(isbias, this, 1);
  network->setLearningRate (0.01);

  std::shared_ptr<SquareErrorLimitController> controller = std::make_shared<SquareErrorLimitController>(0.05, 4);
  network->setController (controller.get());

  for (size_t idx = 0; idx < m_learningSteps && controller->shouldContinue(); ++idx)
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

TEST_F(OapNeuralTests, LogicalOr)
{
  Layer* l1 = network->createLayer(2);
  network->createLayer(1);

  network->setLearningRate (0.1);

  Runner r(false, this);

  std::shared_ptr<SquareErrorLimitController> controller = std::make_shared<SquareErrorLimitController>(0.006, 40);
  network->setController (controller.get());

  for (size_t idx = 0; idx < m_learningSteps && controller->shouldContinue(); ++idx)
  {
    for (size_t value = 1; value <= 10 && controller->shouldContinue(); ++value)
    {
      floatt fvalue = static_cast<floatt>(value);
      r.runTest(fvalue, fvalue, 1);
      l1->printHostWeights();
      r.runTest(fvalue, 0, 1);
      l1->printHostWeights();
      r.runTest(0, fvalue, 1);
      l1->printHostWeights();
      r.runTest(0, 0, 0);
      l1->printHostWeights();
    }
  }

  for (size_t value = 1; value <= 10; ++value)
  {
    floatt fvalue = static_cast<floatt>(value);
    EXPECT_EQ(1, r.run(fvalue, fvalue));
    EXPECT_EQ(1, r.run(fvalue, 0));
    EXPECT_EQ(0, r.run(0, 0));
    EXPECT_EQ(1, r.run(0, fvalue));
  }
}

TEST_F(OapNeuralTests, LogicalAnd)
{
  bool isbias = true;

  Layer* l1 = network->createLayer(2, isbias);
  Layer* l2 = network->createLayer(6);
  Layer* l3 = network->createLayer(1);

  Runner r(isbias, this, 1);
  network->setLearningRate (0.001);

  std::shared_ptr<SquareErrorLimitController> controller = std::make_shared<SquareErrorLimitController>(0.01, 1000);
  network->setController (controller.get());

  std::random_device rd;
  std::default_random_engine dre (rd());
  std::uniform_real_distribution<> dis_0_1(0., 1.);
  std::uniform_real_distribution<> dis_1_2(1., 2.);

  auto for_test = [&](std::uniform_real_distribution<>& dis1, std::uniform_real_distribution<>& dis2)
  {
    for (size_t idx1 = 0; idx1 < 250 && controller->shouldContinue(); ++idx1)
    {
      floatt fvalue = dis1(dre);
      floatt fvalue1 = dis2(dre);
      floatt output = (fvalue >= 1. && fvalue1 >= 1.) ? 1. : 0.;
      r.runTest(fvalue, fvalue1, output);
      l1->printHostWeights();
      l2->printHostWeights();
    }
  };

  for (size_t idx = 0; idx < m_learningSteps && controller->shouldContinue(); ++idx)
  {
    for_test(dis_0_1, dis_0_1);
    for_test(dis_0_1, dis_1_2);
    for_test(dis_1_2, dis_1_2);
    for_test(dis_1_2, dis_0_1);
  }

  for (floatt fv = 0; fv <= 2; fv += 0.1)
  {
    for (floatt fv1 = 0; fv1 <= 2; fv1 += 0.1)
    {
      floatt output = (fv >= 1. && fv1 >= 1.) ? 1. : 0.;
      EXPECT_EQ(output, r.run(fv, fv1)) << " for " << fv << " and " << fv1;
    }
  }
}
/*
TEST_F(OapNeuralTests, LogicalOr_Optimalized)
{
  Layer* l1 = network->createLayer(2);
  network->createLayer(1);

  network->setLearningRate (0.01);

  Runner r(false, this);

  for (size_t idx = 0; idx < m_learningSteps; ++idx)
  {
    for (size_t value = 1; value <= 10; ++value)
    {
      floatt fvalue = static_cast<floatt>(value);
      r.runTest(fvalue, fvalue, 1);
      l1->printHostWeights();
      r.runTest(fvalue, 0, 1);
      l1->printHostWeights();
      r.runTest(0, fvalue, 1);
      l1->printHostWeights();
      r.runTest(0, 0, 0);
      l1->printHostWeights();
    }
  }

  for (size_t value = 1; value <= 10; ++value)
  {
    floatt fvalue = static_cast<floatt>(value);
    EXPECT_EQ(1, r.run(fvalue, fvalue));
    EXPECT_EQ(1, r.run(fvalue, 0));
    EXPECT_EQ(0, r.run(0, 0));
    EXPECT_EQ(1, r.run(0, fvalue));
  }
}

TEST_F(OapNeuralTests, LogicalAnd_Optimalized)
{
  bool isbias = true;

  Layer* l1 = network->createLayer(2, isbias);
  network->createLayer(1);

  Runner r(isbias, this, 1);
  network->setLearningRate (0.01);

  for (size_t idx = 0; idx < m_learningSteps; ++idx)
  {
    for (size_t value = 1; value <= 10; ++value)
    {
      floatt fvalue = static_cast<floatt>(value);
      r.runTest(fvalue, fvalue, 1);
      l1->printHostWeights();
      r.runTest(fvalue, 0, 0);
      l1->printHostWeights();
      r.runTest(0, fvalue, 0);
      l1->printHostWeights();
      r.runTest(0, 0, 0);
      l1->printHostWeights();
    }
  }

  for (size_t value = 1; value <= 10; ++value)
  {
    floatt fvalue = static_cast<floatt>(value);
    EXPECT_EQ(1, r.run(fvalue, fvalue)) << " for " << fvalue;
    EXPECT_EQ(0, r.run(fvalue, 0)) << " for " << fvalue;
    EXPECT_EQ(0, r.run(0, 0)) << " for " << fvalue;
    EXPECT_EQ(0, r.run(0, fvalue)) << " for " << fvalue;
  }
}
*/
