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

#include "oapExpectedOutputs_ForwardPropagation_PyPlotCoords.h"

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

class OapNeuralTests_Forwardpropagation : public testing::Test
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

TEST_F(OapNeuralTests_Forwardpropagation, ForwardPropagation)
{
  DeviceLayer* l1 = network->createLayer(2, true, Activation::TANH);
  DeviceLayer* l2 = network->createLayer(3, true, Activation::TANH);
  DeviceLayer* l3 = network->createLayer(1, Activation::TANH);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (3, 3);
  *GetRePtrIndex (weights1to2, 0) = -1;
  *GetRePtrIndex (weights1to2, 3) = 0.53;
  *GetRePtrIndex (weights1to2, 6) = 0.33;

  *GetRePtrIndex (weights1to2, 1) = -0.1;
  *GetRePtrIndex (weights1to2, 4) = -0.81;
  *GetRePtrIndex (weights1to2, 7) = 0.92;

  *GetRePtrIndex (weights1to2, 2) = 2.2;
  *GetRePtrIndex (weights1to2, 5) = 1.8;
  *GetRePtrIndex (weights1to2, 8) = 1.8;

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (4, 1);
  *GetRePtrIndex (weights2to3, 0) = 4.7;
  *GetRePtrIndex (weights2to3, 1) = 4.7;
  *GetRePtrIndex (weights2to3, 2) = 4.7;
  *GetRePtrIndex (weights2to3, 3) = -7.08;

  l1->setHostWeights (weights1to2);
  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr hinputs = oap::host::NewReMatrix (1, 3);
  oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

  size_t idx = 0;
  auto getLabelIdx = [&hinputs, &houtput, this, &idx] (floatt x, floatt y)
  {
    *GetRePtrIndex (hinputs, 0) = x;
    *GetRePtrIndex (hinputs, 1) = y;

    network->setInputs (hinputs, ArgType::HOST);

    network->forwardPropagation ();

    network->getOutputs (houtput.get(), ArgType::HOST);

    floatt output = GetReIndex (houtput, 0);

    floatt expectedOutput = test_ForwardPropagation_PyPlotCoords::g_expected.at(idx).second;
    EXPECT_NEAR (expectedOutput, output, 0.00001) << "Failure: x: " << x << " y: " << y << " expected: " << expectedOutput << " actual: " << output;
    EXPECT_NEAR (x, test_ForwardPropagation_PyPlotCoords::g_expected.at(idx).first.first, 0.05);
    EXPECT_NEAR (y, test_ForwardPropagation_PyPlotCoords::g_expected.at(idx).first.second, 0.05);
    ++idx;

    if (output < -0.5)
    {
      return 0;
    }
    else if (output >= -0.5 && output < 0)
    {
      return 1;
    }
    else if (output >= 0 && output < 0.5)
    {
      return 2;
    }
    return 3;
  };

#ifdef OAP_TESTS_PLOT
  oap::pyplot::plotCoords2D ("/tmp/ForwardPropagation_PyPlotCoords.py", std::make_tuple(-6,6,0.1), std::make_tuple(-6,6,0.1), getLabelIdx, {"r.", "g.", "y.", "b."});
#endif
}

TEST_F(OapNeuralTests_Forwardpropagation, ForwardPropagation_PyPlotCoords_Parallel)
{
  DeviceLayer* l1 = network->createLayer(2, true, Activation::TANH);
  DeviceLayer* l2 = network->createLayer(3, true, Activation::TANH);
  DeviceLayer* l3 = network->createLayer(1, Activation::TANH);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (3, 3);
  *GetRePtrIndex (weights1to2, 0) = -1;
  *GetRePtrIndex (weights1to2, 3) = 0.53;
  *GetRePtrIndex (weights1to2, 6) = 0.33;

  *GetRePtrIndex (weights1to2, 1) = -0.1;
  *GetRePtrIndex (weights1to2, 4) = -0.81;
  *GetRePtrIndex (weights1to2, 7) = 0.92;

  *GetRePtrIndex (weights1to2, 2) = 2.2;
  *GetRePtrIndex (weights1to2, 5) = 1.8;
  *GetRePtrIndex (weights1to2, 8) = 1.8;

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (4, 1);
  *GetRePtrIndex (weights2to3, 0) = 4.7;
  *GetRePtrIndex (weights2to3, 1) = 4.7;
  *GetRePtrIndex (weights2to3, 2) = 4.7;
  *GetRePtrIndex (weights2to3, 3) = -7.08;

  l1->setHostWeights (weights1to2);
  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr hinputs = oap::host::NewReMatrix (1, 3);
  oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

  size_t idx = 0;
  auto getLabelIdx = [&hinputs, &houtput, this, &idx] (floatt x, floatt y)
  {
    *GetRePtrIndex (hinputs, 0) = x;
    *GetRePtrIndex (hinputs, 1) = y;
    *GetRePtrIndex (hinputs, 2) = 1;

    network->setInputs (hinputs, ArgType::HOST);

    network->forwardPropagation ();

    network->getOutputs (houtput.get(), ArgType::HOST);

    floatt output = GetReIndex (houtput, 0);

    floatt expectedOutput = test_ForwardPropagation_PyPlotCoords::g_expected.at(idx).second;
    EXPECT_NEAR (expectedOutput, output, 0.00001) << "Failure: x: " << x << " y: " << y << " expected: " << expectedOutput << " actual: " << output;
    EXPECT_NEAR (x, test_ForwardPropagation_PyPlotCoords::g_expected.at(idx).first.first, 0.05);
    EXPECT_NEAR (y, test_ForwardPropagation_PyPlotCoords::g_expected.at(idx).first.second, 0.05);
    ++idx;

    if (output < -0.5)
    {
      return 0;
    }
    else if (output >= -0.5 && output < 0)
    {
      return 1;
    }
    else if (output >= 0 && output < 0.5)
    {
      return 2;
    }
    return 3;
  };

#ifdef OAP_TESTS_PLOT
  oap::pyplot::plotCoords2D ("/tmp/ForwardPropagation_PyPlotCoords_Parallel.py", std::make_tuple(-6,6,0.1), std::make_tuple(-6,6,0.1), getLabelIdx, {"r.", "g.", "y.", "b."});
#endif
}
