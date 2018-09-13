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

  void runTest(floatt a1, floatt a2, floatt e1)
  {
    oap::HostMatrixUPtr inputs = oap::host::NewReMatrix(2, 1);
    oap::HostMatrixUPtr expected = oap::host::NewReMatrix(1, 1);
    inputs->reValues[0] = a1;
    inputs->reValues[1] = a2;
    expected->reValues[0] = e1;

    network->runHostArgsTest(inputs, expected);
  }

  floatt run(floatt a1, floatt a2)
  {
    oap::HostMatrixUPtr inputs = oap::host::NewReMatrix(2, 1);
    inputs->reValues[0] = a1;
    inputs->reValues[1] = a2;

    auto output = network->runHostArgs(inputs);
    return is(output->reValues[0]);
  }

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
    if (a > 0.5)
    {
      return 1;
    }
    return 0;
  }

  void testForwardPropagation_2_to_1(floatt w_1, floatt w_2, floatt i_1, floatt i_2)
  {
    Layer* l1 = network->createLayer(2);
    network->createLayer(1);

    network->setLearningRate (1);

    oap::HostMatrixUPtr hw = oap::host::NewReMatrix (2, 1);
    oap::HostMatrixUPtr hw1 = oap::host::NewReMatrix (2, 1);

    floatt hw_1 = w_1;
    floatt hw_2 = w_2;
    floatt hw1_1 = i_1;
    floatt hw1_2 = i_2;

    hw->reValues[0] = hw_1;
    hw->reValues[1] = hw_2;

    hw1->reValues[0] = hw1_1;
    hw1->reValues[1] = hw1_2;

    l1->setHostWeights (hw.get ());

    auto output = network->runHostArgs (hw1);

    EXPECT_THAT(output->reValues[0], testing::DoubleNear(sigmoid(hw_1 * hw1_1 + hw_2 * hw1_2), 0.0001));
    EXPECT_EQ(1, output->columns);
    EXPECT_EQ(1, output->rows);
  }

  void testBackPropagation_1_to_2(floatt w_1, floatt w_2, floatt i_1, floatt i_2, floatt e_1)
  {
    Layer* l1 = network->createLayer(2);
    network->createLayer(1);

    network->setLearningRate (1);

    oap::HostMatrixUPtr hw = oap::host::NewReMatrix (2, 1);
    oap::HostMatrixUPtr io = oap::host::NewReMatrix (2, 1);
    oap::HostMatrixUPtr io1 = oap::host::NewReMatrix (1, 1);
    oap::HostMatrixUPtr e1 = oap::host::NewReMatrix (1, 1);
    oap::DeviceMatrixUPtr de1 = oap::cuda::NewDeviceReMatrix(1, 1);

    floatt hw_1 = w_1;
    floatt hw_2 = w_2;

    hw->reValues[0] = hw_1;
    hw->reValues[1] = hw_2;

    io->reValues[0] = i_1;
    io->reValues[1] = i_2;

    floatt i1_1 = sigmoid(i_1 * hw_1 + i_2 * hw_2);
    e1->reValues[0] = e_1;

    oap::cuda::CopyHostMatrixToDeviceMatrix (de1, e1);

    l1->setHostWeights (hw.get ());

    network->setHostInput (io, 0);
    network->runHostArgsTest (io, e1);

    hw->reValues[0] = 0;
    hw->reValues[1] = 0;

    network->getHostWeights(hw, 0);

    floatt sigma = e_1 - i1_1;
    floatt ds = dsigmoid(i_1 * hw_1 + i_2 * hw_2);

    floatt c1 = ds * sigma * i_1;
    floatt c2 = ds * sigma * i_2;

    EXPECT_THAT(hw->reValues[0] - hw_1, testing::DoubleNear(c1, 0.0001));
    EXPECT_THAT(hw->reValues[1] - hw_2, testing::DoubleNear(c2, 0.0001));
  }
};

TEST_F(OapNeuralTests, ForwardPropagation_1)
{
  testForwardPropagation_2_to_1 (1, 1, 1, 1);
}

TEST_F(OapNeuralTests, ForwardPropagation_2)
{
  testForwardPropagation_2_to_1 (1, 2, 3, 4);
}

TEST_F(OapNeuralTests, ForwardPropagation_3)
{
  testForwardPropagation_2_to_1 (11, 22, 33, 44);
}

TEST_F(OapNeuralTests, ForwardPropagation_4)
{
  testForwardPropagation_2_to_1 (111, 221, 331, 441);
}

TEST_F(OapNeuralTests, BackPropagation_1)
{
  testBackPropagation_1_to_2 (1, 1, 1, 1, 1);
}

TEST_F(OapNeuralTests, BackPropagation_2)
{
  testBackPropagation_1_to_2 (2, 1, 1, 1, 0);
}

TEST_F(OapNeuralTests, BackPropagation_3)
{
  testBackPropagation_1_to_2 (2, 1, 3, 2, 1);
}

TEST_F(OapNeuralTests, BackPropagation_4)
{
  testBackPropagation_1_to_2 (1, 2, 3, 4, 5);
}

TEST_F(OapNeuralTests, LogicalOr)
{
  Layer* l1 = network->createLayer(2);
  network->createLayer(1);

  network->setLearningRate (1);

  runTest(1, 1, 1);
  l1->printHostWeights();
  runTest(1, 0, 1);
  l1->printHostWeights();
  runTest(0, 1, 1);
  l1->printHostWeights();
  runTest(0, 0, 0);
  l1->printHostWeights();

  EXPECT_EQ(1, run(1, 1));
  EXPECT_EQ(1, run(1, 0));
  EXPECT_EQ(0, run(0, 0));
  EXPECT_EQ(1, run(1, 0));
}

TEST_F(OapNeuralTests, LogicalAnd)
{
  Layer* l1 = network->createLayer(2);
  network->createLayer(1);

  network->setLearningRate (1);

  size_t value = 1;
  floatt fvalue = static_cast<floatt>(value);
  runTest(fvalue, fvalue, 1);
  l1->printHostWeights();
  runTest(fvalue, 0, 0);
  l1->printHostWeights();
  runTest(0, fvalue, 0);
  l1->printHostWeights();
  runTest(0, 0, 0);
  l1->printHostWeights();

  EXPECT_EQ(1, run(1, 1));
  EXPECT_EQ(0, run(1, 0));
  EXPECT_EQ(0, run(0, 0));
  EXPECT_EQ(0, run(0, 1));
}

TEST_F(OapNeuralTests, LogicalAnd_LargeValues)
{
  Layer* l1 = network->createLayer(2);
  network->createLayer(1);

  network->setLearningRate (0.01);

  for (size_t value = 1; value < 1000; ++value)
  {
    floatt fvalue = static_cast<floatt>(value);
    runTest(fvalue, fvalue, 1);
    l1->printHostWeights();
    runTest(fvalue, 0, 0);
    l1->printHostWeights();
    runTest(0, fvalue, 0);
    l1->printHostWeights();
    runTest(0, 0, 0);
    l1->printHostWeights();
  }

  EXPECT_EQ(1, run(1, 1));
  EXPECT_EQ(0, run(1, 0));
  EXPECT_EQ(0, run(0, 0));
  EXPECT_EQ(0, run(0, 1));
}

