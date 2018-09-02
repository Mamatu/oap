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

class OapNeuralTests : public testing::Test {
 public:
  CUresult status;
  Network* network;

  virtual void SetUp()
  {
    oap::cuda::Context::Instance().create();
    network = new Network();
  }

  virtual void TearDown()
  {
    delete network;
    oap::cuda::Context::Instance().destroy();
  }

  void runTest(floatt a1, floatt a2, floatt e1)
  {
    oap::HostMatrixUPtr inputs = oap::host::NewReMatrix(2, 1);
    oap::HostMatrixUPtr expected = oap::host::NewReMatrix(1, 1);
    inputs->reValues[0] = a1;
    inputs->reValues[1] = a2;
    expected->reValues[0] = e1;

    network->runHostArgsTest(inputs.get(), expected.get());
  }

  floatt run(floatt a1, floatt a2)
  {
    oap::HostMatrixUPtr inputs = oap::host::NewReMatrix(2, 1);
    inputs->reValues[0] = a1;
    inputs->reValues[1] = a2;

    auto output = network->runHostArgs(inputs.get());
    return is(output->reValues[0]);
  }

  floatt sigmoid(floatt x)
  {
    return 1.f / (1.f + exp (-x));
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
};

TEST_F(OapNeuralTests, LogicalOr)
{
  Layer* l1 = network->createLayer(2);
  network->createLayer(1);

  network->setLearningRate (0.1);

  runTest(1, 1, 1);
  runTest(1, 0, 1);
  runTest(0, 1, 1);
  runTest(0, 0, 0);

  EXPECT_EQ(1, run(1, 1));
  EXPECT_EQ(1, run(1, 0));
  EXPECT_EQ(0, run(0, 0));
  EXPECT_EQ(1, run(1, 0));
}

TEST_F(OapNeuralTests, LogicalAnd)
{
  Layer* l1 = network->createLayer(2);
  network->createLayer(1);

  network->setLearningRate (0.1);

  runTest(1, 1, 1);
  runTest(1, 0, 0);
  runTest(0, 1, 0);
  runTest(0, 0, 0);

  EXPECT_EQ(1, run(1, 1));
  EXPECT_EQ(0, run(1, 0));
  EXPECT_EQ(0, run(0, 0));
  EXPECT_EQ(0, run(0, 1));
}

