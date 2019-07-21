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

#include "oapExpectedOutputs_ForwardPropagation_PyPlotCoords.h"

void setReValueToMatrix (math::Matrix* matrix, floatt value, size_t idx = 0)
{
  matrix->reValues[idx] = value;
}

void setReValuesToMatrix (math::Matrix* matrix, const std::vector<floatt>& vec)
{
  debugAssert (vec.size() <= matrix->columns * matrix->rows);
  memcpy (matrix->reValues, vec.data(), sizeof(floatt) * vec.size());
}

void setReValuesToMatrix (math::Matrix* matrix, floatt* array, size_t length)
{
  debugAssert (length <= matrix->columns * matrix->rows);
  memcpy (matrix->reValues, array, sizeof(floatt) * length);
}

template<typename Tuple, size_t Tidx = 0>
void setReValuesToMatrix (math::Matrix* matrix, const std::vector<Tuple>& vecl)
{
  std::vector<floatt> vec;
  vec.reserve (vecl.size());
  for (auto it = vecl.begin(); it != vecl.end(); ++it)
  {
    vec.push_back (std::get<Tidx>(*it));
  }

  setReValuesToMatrix (matrix, vec.data(), vec.size());
}

class NetworkT : public Network
{
  public:
    void setHostInput (math::Matrix* inputs, size_t index)
    {
      Network::setHostInputs (inputs, index);
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

      m_ont->network->train (inputs, expected, ArgType::HOST, oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
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

      auto output = m_ont->network->run (inputs, ArgType::HOST, oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
      return m_ont->is(output->reValues[0]);
    }
  };

  floatt is(floatt a)
  {
    debug("arg is %f", a);
    if (a > 0.5f)
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
    oap::HostMatrixUPtr hinputs = oap::host::NewReMatrix (1, 2);

    floatt hw_1 = w_1;
    floatt hw_2 = w_2;

    hw->reValues[0] = hw_1;
    hw->reValues[1] = hw_2;

    hinputs->reValues[0] = i_1;
    hinputs->reValues[1] = i_2;

    l1->setHostWeights (hw.get ());

    auto output = network->run (hinputs, ArgType::HOST, oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);

    EXPECT_THAT(output->reValues[0], testing::DoubleNear(oap::math::sigmoid(hw_1 * i_1 + hw_2 * i_2), 0.0001));
    EXPECT_EQ(1, output->columns);
    EXPECT_EQ(1, output->rows);
  }

  void testBackPropagation_1_to_2(floatt w_1, floatt w_2, floatt i_1, floatt i_2, floatt e_1)
  {
    Layer* l1 = network->createLayer(2);
    Layer* l2 = network->createLayer(1);

    network->setLearningRate (1);

    oap::HostMatrixUPtr hw = oap::host::NewReMatrix (2, 1);
    oap::HostMatrixUPtr io = oap::host::NewReMatrix (1, 2);
    oap::HostMatrixUPtr io1 = oap::host::NewReMatrix (1, 1);
    oap::HostMatrixUPtr e1 = oap::host::NewReMatrix (1, 1);
    oap::DeviceMatrixUPtr de1 = oap::cuda::NewDeviceReMatrix(1, 1);

    floatt hw_1 = w_1;
    floatt hw_2 = w_2;

    hw->reValues[0] = hw_1;
    hw->reValues[1] = hw_2;

    io->reValues[0] = i_1;
    io->reValues[1] = i_2;

    floatt i1_1 = oap::math::sigmoid(i_1 * hw_1 + i_2 * hw_2);
    e1->reValues[0] = e_1;

    oap::cuda::CopyHostMatrixToDeviceMatrix (de1, e1);

    l1->setHostWeights (hw.get ());

    network->setHostInput (io, 0);
    network->train (io, e1, ArgType::HOST, oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);

    hw->reValues[0] = 0;
    hw->reValues[1] = 0;

    network->getHostWeights(hw, 0);

    floatt sigma = e_1 - i1_1;
    floatt ds = oap::math::dsigmoid(i_1 * hw_1 + i_2 * hw_2);

    floatt c1 = ds * sigma * i_1;
    floatt c2 = ds * sigma * i_2;

    EXPECT_THAT(hw->reValues[0] - hw_1, testing::DoubleNear(c1, 0.0001));
    EXPECT_THAT(hw->reValues[1] - hw_2, testing::DoubleNear(c2, 0.0001));

    l1->printHostWeights ();
    l2->printHostWeights ();
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

TEST_F(OapNeuralTests, SaveLoadBufferTest)
{
  bool isbias = true;

  Layer* l1 = network->createLayer(isbias ? 3 : 2);
  Layer* l2 = network->createLayer(6);
  Layer* l3 = network->createLayer(1);

  Runner r(isbias, this, 1);
  network->setLearningRate (0.001);

  std::random_device rd;
  std::default_random_engine dre (rd());
  std::uniform_real_distribution<> dis_0_1(0., 1.);
  std::uniform_real_distribution<> dis_1_2(1., 2.);

  auto for_test = [&](std::uniform_real_distribution<>& dis1, std::uniform_real_distribution<>& dis2)
  {
    for (size_t idx1 = 0; idx1 < 25; ++idx1)
    {
      floatt fvalue = dis1(dre);
      floatt fvalue1 = dis2(dre);
      floatt output = (fvalue >= 1. && fvalue1 >= 1.) ? 1. : 0.;
      r.runTest(fvalue, fvalue1, output);
    }
  };

  for (size_t idx = 0; idx < 1; ++idx)
  {
    for_test(dis_0_1, dis_0_1);
    for_test(dis_0_1, dis_1_2);
    for_test(dis_1_2, dis_1_2);
    for_test(dis_1_2, dis_0_1);
  }

  utils::ByteBuffer buffer;
  network->save (buffer);

  std::unique_ptr<Network> cnetwork (Network::load (buffer));

  EXPECT_TRUE (*network == *cnetwork);
}

TEST_F(OapNeuralTests, SaveLoadFileTest)
{
  bool isbias = true;

  Layer* l1 = network->createLayer(isbias ? 3 : 2);
  Layer* l2 = network->createLayer(6);
  Layer* l3 = network->createLayer(1);

  Runner r(isbias, this, 1);
  network->setLearningRate (0.001);

  std::random_device rd;
  std::default_random_engine dre (rd());
  std::uniform_real_distribution<> dis_0_1(0., 1.);
  std::uniform_real_distribution<> dis_1_2(1., 2.);

  auto for_test = [&](std::uniform_real_distribution<>& dis1, std::uniform_real_distribution<>& dis2)
  {
    for (size_t idx1 = 0; idx1 < 25; ++idx1)
    {
      floatt fvalue = dis1(dre);
      floatt fvalue1 = dis2(dre);
      floatt output = (fvalue >= 1. && fvalue1 >= 1.) ? 1. : 0.;
      r.runTest(fvalue, fvalue1, output);
    }
  };

  for (size_t idx = 0; idx < 1; ++idx)
  {
    for_test(dis_0_1, dis_0_1);
    for_test(dis_0_1, dis_1_2);
    for_test(dis_1_2, dis_1_2);
    for_test(dis_1_2, dis_0_1);
  }

  std::string path = "device_tests/OapNeuralTests_SaveLoadFileTest.bin";
  path = utils::Config::getFileInTmp (path);

  auto save = [&]()
  {
    utils::ByteBuffer buffer;
    network->save (buffer);
    buffer.fwrite (path);
  };

  auto load = [&]() -> std::unique_ptr<Network>
  {
    utils::ByteBuffer buffer (path);
    return std::unique_ptr<Network> (Network::load (buffer));
  };

  save ();
  auto cnetwork = load ();

  EXPECT_TRUE (*network == *cnetwork);
}

TEST_F(OapNeuralTests, NeuralNetworkTest)
{
  // values come from https://www.nnwj.de/backpropagation.html
  Layer* l1 = network->createLayer(2);
  Layer* l2 = network->createLayer(2);
  Layer* l3 = network->createLayer(1);

  network->setLearningRate (0.25);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 2);
  weights1to2->reValues[0] = 0.62;
  weights1to2->reValues[2] = 0.42;

  weights1to2->reValues[1] = 0.55;
  weights1to2->reValues[3] = -0.17;

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (2, 1);
  weights2to3->reValues[0] = 0.35;
  weights2to3->reValues[1] = 0.81;

  l1->setHostWeights (weights1to2);
  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 2);
  inputs->reValues[0] = 0;
  inputs->reValues[1] = 1;
  oap::HostMatrixPtr eoutput = oap::host::NewReMatrix (1, 1, 0);

  network->train (inputs, eoutput, ArgType::HOST, oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);

  weights1to2 = oap::host::NewReMatrix (2, 2);
  weights2to3 = oap::host::NewReMatrix (2, 1);
  l1->getHostWeights (weights1to2);
  l2->getHostWeights (weights2to3);

  EXPECT_NEAR (0.326593362, weights2to3->reValues[0], 0.00001);
  EXPECT_NEAR (0.793109407, weights2to3->reValues[1], 0.00001);

  EXPECT_NEAR (0.62, weights1to2->reValues[0], 0.00001);
  EXPECT_NEAR (0.42, weights1to2->reValues[2], 0.00001);
  EXPECT_NEAR (0.512648936, weights1to2->reValues[1], 0.00001);
  EXPECT_NEAR (-0.209958271, weights1to2->reValues[3], 0.00001);
}

TEST_F(OapNeuralTests, SimpleForwardPropagation_1)
{
  using namespace oap::math;

  Layer* l1 = network->createLayer(2);
  Layer* l2 = network->createLayer(2);
  Layer* l3 = network->createLayer(1);

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

  auto minfo = l3->getOutputsDim ();
  oap::HostMatrixPtr outputsL3 = oap::host::NewReMatrix (minfo.m_matrixDim.columns, minfo.m_matrixDim.rows);
  l3->getOutputs (outputsL3, oap::HOST);

  EXPECT_DOUBLE_EQ (sigmoid (sigmoid(2) + sigmoid(2)), outputsL3->reValues[0]);
}

TEST_F(OapNeuralTests, SimpleForwardPropagation_2)
{
  using namespace oap::math;

  Layer* l1 = network->createLayer(3);
  Layer* l2 = network->createLayer(3);
  Layer* l3 = network->createLayer(1);

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

  auto minfo = l3->getOutputsDim ();
  oap::HostMatrixPtr outputsL3 = oap::host::NewReMatrix (minfo.m_matrixDim.columns, minfo.m_matrixDim.rows);
  l3->getOutputs (outputsL3, oap::HOST);

  EXPECT_DOUBLE_EQ (sigmoid (sigmoid(3) + sigmoid(3) + sigmoid(3)), outputsL3->reValues[0]);
}

TEST_F(OapNeuralTests, SimpleForwardPropagation_3)
{
  using namespace oap::math;

  Layer* l1 = network->createLayer(3);
  Layer* l2 = network->createLayer(3);
  Layer* l3 = network->createLayer(1);

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

  auto minfo = l3->getOutputsDim ();
  oap::HostMatrixPtr outputsL3 = oap::host::NewReMatrix (minfo.m_matrixDim.columns, minfo.m_matrixDim.rows);
  l3->getOutputs (outputsL3, oap::HOST);

  EXPECT_DOUBLE_EQ (sigmoid (sigmoid(4) + sigmoid(4) + sigmoid(4)), outputsL3->reValues[0]);
}

TEST_F(OapNeuralTests, SimpleForwardPropagation_4)
{
  using namespace oap::math;

  Layer* l1 = network->createLayer(3);
  Layer* l2 = network->createLayer(3);
  Layer* l3 = network->createLayer(1);

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

  auto getLayerOutput = [](Layer* layer)
  {
    auto minfo = layer->getOutputsDim ();
    oap::HostMatrixPtr outputsL = oap::host::NewReMatrix (minfo.m_matrixDim.columns, minfo.m_matrixDim.rows);
    layer->getOutputs (outputsL, oap::HOST);
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

TEST_F(OapNeuralTests, ForwardPropagation_PyPlotCoords)
{
  Layer* l1 = network->createLayer(3, false, Activation::TANH);
  Layer* l2 = network->createLayer(3, true, Activation::TANH);
  Layer* l3 = network->createLayer(1, Activation::TANH);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (3, 4);
  weights1to2->reValues[0] = -1;
  weights1to2->reValues[3] = 0.53;
  weights1to2->reValues[6] = 0.33;
  weights1to2->reValues[9] = 0;

  weights1to2->reValues[1] = -0.1;
  weights1to2->reValues[4] = -0.81;
  weights1to2->reValues[7] = 0.92;
  weights1to2->reValues[10] = 0;

  weights1to2->reValues[2] = 2.2;
  weights1to2->reValues[5] = 1.8;
  weights1to2->reValues[8] = 1.8;
  weights1to2->reValues[11] = 0;

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (4, 1);
  weights2to3->reValues[0] = 4.7;
  weights2to3->reValues[1] = 4.7;
  weights2to3->reValues[2] = 4.7;
  weights2to3->reValues[3] = -7.08;

  l1->setHostWeights (weights1to2);
  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr hinputs = oap::host::NewReMatrix (1, 3);
  oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

  size_t idx = 0;
  auto getLabelIdx = [&hinputs, &houtput, this, &idx] (floatt x, floatt y)
  {
    hinputs->reValues[0] = x;
    hinputs->reValues[1] = y;
    hinputs->reValues[2] = 1;

    network->setInputs (hinputs, ArgType::HOST);

    network->forwardPropagation ();

    network->getOutputs (houtput.get(), ArgType::HOST);

    floatt output = houtput->reValues[0];

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

TEST_F(OapNeuralTests, ForwardPropagation_PyPlotCoords_Parallel)
{
  Layer* l1 = network->createLayer(3, false, Activation::TANH);
  Layer* l2 = network->createLayer(3, true, Activation::TANH);
  Layer* l3 = network->createLayer(1, Activation::TANH);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (3, 4);
  weights1to2->reValues[0] = -1;
  weights1to2->reValues[3] = 0.53;
  weights1to2->reValues[6] = 0.33;
  weights1to2->reValues[9] = 0;

  weights1to2->reValues[1] = -0.1;
  weights1to2->reValues[4] = -0.81;
  weights1to2->reValues[7] = 0.92;
  weights1to2->reValues[10] = 0;

  weights1to2->reValues[2] = 2.2;
  weights1to2->reValues[5] = 1.8;
  weights1to2->reValues[8] = 1.8;
  weights1to2->reValues[11] = 0;

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (4, 1);
  weights2to3->reValues[0] = 4.7;
  weights2to3->reValues[1] = 4.7;
  weights2to3->reValues[2] = 4.7;
  weights2to3->reValues[3] = -7.08;

  l1->setHostWeights (weights1to2);
  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr hinputs = oap::host::NewReMatrix (1, 3);
  oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

  size_t idx = 0;
  auto getLabelIdx = [&hinputs, &houtput, this, &idx] (floatt x, floatt y)
  {
    hinputs->reValues[0] = x;
    hinputs->reValues[1] = y;
    hinputs->reValues[2] = 1;

    network->setInputs (hinputs, ArgType::HOST);

    network->forwardPropagation ();

    network->getOutputs (houtput.get(), ArgType::HOST);

    floatt output = houtput->reValues[0];

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

TEST_F(OapNeuralTests, BackwardPropagation_PyPlotCoords_1)
{
  Layer* l1 = network->createLayer(3, false, Activation::TANH);
  Layer* l2 = network->createLayer(3, true, Activation::TANH);
  Layer* l3 = network->createLayer(1, Activation::TANH);
  network->setLearningRate (0.03);

  using PConvert = std::pair<floatt, floatt>;
  std::vector<PConvert> weights1to2Vec =
  {
    {-0.03844587775473951, -0.03428101998933327},
    {-0.19482274545428657, -0.19317562257641502},
    {0.1, 0.09974779149698941},
    {-0.48058386853487256, -0.4827218451998201},
    {-0.04542142439186897, -0.04726372254435102},
    {0.1, 0.10082359690201734},
    {0.4925593825477017, 0.4923944692068602},
    {0.4154241670749974, 0.4153694370016125},
    {0.1, 0.10008534844421194},
    {0.,0.},
    {0.,0.},
    {0.,0.}
  };

  std::vector<PConvert> weights2to3Vec =
  {
    {-0.20444003164948682, -0.20193914343449443},
    {0.23061038559617397, 0.2391694840585953},
    {0.036329506711680115, 0.027658209241972043},
    {0.1, 0.09819870188084666},
  };

  std::vector<size_t> idxToCheck1 = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<size_t> idxToCheck2 = {0, 1, 2, 3};

  std::vector<std::pair<std::pair<floatt, floatt>, floatt>> points =
  {
    {{-3.4275013199113995, 3.472108546014614}, -1},
    {{-0.9129396943609743, 0.35837215256198157}, 1},
    {{-0.2716364245490655, -0.4625308393815398}, 1},
    {{0.8126092445716526, -0.4043453500882416}, 1},
    {{0.6752134587177548, 3.8164173152395726}, -1},
    {{-1.5956210634644072, -0.5000970991895521}, 1},
    {{4.377934920368058, -0.7291583904411608}, -1},
    {{-1.9835003774230164, 0.3673867240076611}, 1},
    {{3.486707415017228, -3.0465899941800587}, -1},
    {{2.173984324655197, -3.912626545077933}, -1},
  };

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (3, 4);
  for (size_t idx = 0; idx < weights1to2Vec.size(); ++idx)
  {
    weights1to2->reValues[idx] = weights1to2Vec[idx].first;
  }

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (4, 1);
  for (size_t idx = 0; idx < weights2to3Vec.size(); ++idx)
  {
    weights2to3->reValues[idx] = weights2to3Vec[idx].first;
  }

  l1->setHostWeights (weights1to2);
  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr hinputs = oap::host::NewReMatrix (1, 3);
  oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

  size_t idx = 0;
  for (const auto& p : points)
  {
    hinputs->reValues[0] = p.first.first;
    hinputs->reValues[1] = p.first.second;
    hinputs->reValues[2] = 1;

    houtput->reValues[0] = p.second;

    network->setInputs (hinputs, ArgType::HOST);
    network->setExpected (houtput, ArgType::HOST);

    network->forwardPropagation ();
    network->calculateErrors (oap::ErrorType::MEAN_SQUARE_ERROR);
  }

  network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR);
  network->backwardPropagation ();

  l1->getHostWeights (weights1to2);
  for (size_t idx = 0; idx < idxToCheck1.size(); ++idx)
  {
      size_t trueIdx = idxToCheck1[idx];
      EXPECT_NEAR (weights1to2Vec[trueIdx].second, weights1to2->reValues[trueIdx], 0.001) << "Idx: " << idx;
  }
}

TEST_F(OapNeuralTests, BackwardPropagation_PyPlotCoords_2)
{
  Layer* l1 = network->createLayer(3, false, Activation::TANH);
  Layer* l2 = network->createLayer(3, true, Activation::TANH);
  Layer* l3 = network->createLayer(1, Activation::TANH);
  network->setLearningRate (0.03);

  using PConvert = std::pair<floatt, floatt>;
  std::vector<PConvert> weights1to2Vec =
  {
    {0.2, 0.2014514943123398},
    {0.2, 0.20074015751338653},
    {0.1, 0.09889165677634874},
    {0.2, 0.2014514943123398},
    {0.2, 0.20074015751338653},
    {0.1, 0.09889165677634874},
    {0.2, 0.2014514943123398},
    {0.2, 0.20074015751338653},
    {0.1, 0.09889165677634874}
  };

  std::vector<PConvert> weights2to3Vec =
  {
    {0.2, 0.2013002870720081},
    {0.2, 0.2013002870720081},
    {0.2, 0.2013002870720081},
    {0.1, 0.08822316417432748}
  };

  std::vector<size_t> idxToCheck1 = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<size_t> idxToCheck2 = {0, 1, 2, 3};

  std::vector<std::pair<std::pair<floatt, floatt>, floatt>> points =
  {
    {{2.382377282145529, 2.5982150872108445}, -1},
    {{-2.2419128617962625, -2.8587813033619534}, -1},
    {{1.7112768352615761, -4.192861294580691}, -1},
    {{-0.001731499521305971, 0.21355281143153995}, 1},
    {{0.03304381776128395, 0.2745599342542303}, 1},
    {{-2.8107451617662207, 2.5714960848151365}, -1},
    {{-3.4045680012923976, -1.505414350405937}, -1},
    {{3.578185653682367, -0.4271641083518562}, -1},
    {{-4.727018012452029, 0.2297057556153474}, -1},
    {{1.1541123870730146, 0.27666822157065263}, 1}
  };

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (3, 4);
  for (size_t idx = 0; idx < weights1to2Vec.size(); ++idx)
  {
    weights1to2->reValues[idx] = weights1to2Vec[idx].first;
  }

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (4, 1);
  for (size_t idx = 0; idx < weights2to3Vec.size(); ++idx)
  {
    weights2to3->reValues[idx] = weights2to3Vec[idx].first;
  }

  l1->setHostWeights (weights1to2);
  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr hinputs = oap::host::NewReMatrix (1, 3);
  oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

  size_t idx = 0;
  for (const auto& p : points)
  {
    hinputs->reValues[0] = p.first.first;
    hinputs->reValues[1] = p.first.second;
    hinputs->reValues[2] = 1;

    houtput->reValues[0] = p.second;

    network->setInputs (hinputs, ArgType::HOST);
    network->setExpected (houtput, ArgType::HOST);

    network->forwardPropagation ();
    network->calculateErrors (oap::ErrorType::MEAN_SQUARE_ERROR);
  }

  network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR);
  network->backwardPropagation ();

  l1->getHostWeights (weights1to2);
  for (size_t idx = 0; idx < idxToCheck1.size(); ++idx)
  {
    size_t trueIdx = idxToCheck1[idx];
    EXPECT_NEAR (weights1to2Vec[trueIdx].second, weights1to2->reValues[trueIdx], 0.001) << "Idx: " << idx;
  }
}

TEST_F(OapNeuralTests, BackwardPropagation_PyPlotCoords_3)
{
  Layer* l1 = network->createLayer(3, false, Activation::TANH);
  Layer* l2 = network->createLayer(3, true, Activation::TANH);
  Layer* l3 = network->createLayer(1, Activation::TANH);
  network->setLearningRate (0.03);

  using PConvert = std::tuple<floatt, floatt, bool>;
  std::vector<PConvert> weights1to2Vec =
  {
    std::make_tuple(0.2, 0.19908199840345, true),
    std::make_tuple(0.2, 0.19356847603468466, true),
    std::make_tuple(0.1, 0.10580908512486277, true),
    std::make_tuple(0.2, 0.19908199840345, true),
    std::make_tuple(0.2, 0.19356847603468466, true),
    std::make_tuple(0.1, 0.10580908512486277, true),
    std::make_tuple(0.2, 0.19908199840345, true),
    std::make_tuple(0.2, 0.19356847603468466, true),
    std::make_tuple(0.1, 0.10580908512486277, true),
    std::make_tuple(0, 0, false),
    std::make_tuple(0, 0, false),
    std::make_tuple(0, 0, false),
  };

  std::vector<PConvert> weights2to3Vec =
  {
    std::make_tuple(0.2, 0.1954852905493779, true),
    std::make_tuple(0.2, 0.1954852905493779, true),
    std::make_tuple(0.2, 0.1954852905493779, true),
    std::make_tuple(0.1, 0.1297309930846901, true),
  };

  std::vector<std::pair<std::pair<floatt, floatt>, floatt>> points =
  {
    {{-0.15802860120278975, -1.1071492028561536}, 1},
  };

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (3, 4);
  setReValuesToMatrix (weights1to2, weights1to2Vec);

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (4, 1);
  setReValuesToMatrix (weights2to3, weights2to3Vec);

  l1->setHostWeights (weights1to2);
  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr hinputs = oap::host::NewReMatrix (1, 3);
  oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

  size_t idx = 0;
  for (const auto& p : points)
  {
    hinputs->reValues[0] = p.first.first;
    hinputs->reValues[1] = p.first.second;
    hinputs->reValues[2] = 1;

    houtput->reValues[0] = p.second;

    network->setInputs (hinputs, ArgType::HOST);
    network->setExpected (houtput, ArgType::HOST);

    network->forwardPropagation ();
    network->calculateErrors (oap::ErrorType::MEAN_SQUARE_ERROR);
  }

  network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR);
  network->backwardPropagation ();

  l1->getHostWeights (weights1to2);
  for (size_t idx = 0; idx < weights1to2Vec.size() ; ++idx)
  {
      if (std::get<2>(weights1to2Vec[idx]))
      {
        EXPECT_NEAR (std::get<1>(weights1to2Vec[idx]), weights1to2->reValues[idx], 0.000001) << "Idx: " << idx;
      }
  }
}

TEST_F(OapNeuralTests, BackwardPropagation_PyPlotCoords_4)
{
  Layer* l1 = network->createLayer(3, false, Activation::TANH);
  Layer* l2 = network->createLayer(3, true, Activation::TANH);
  Layer* l3 = network->createLayer(1, Activation::TANH);
  network->setLearningRate (0.03);

  using PConvert = std::pair<floatt, floatt>;
  std::vector<PConvert> weights1to2Vec =
  {
    {0.2, 0.20182628407433365},
    {0.2, 0.20093695144385168},
    {0.1, 0.10411721816404285},
    {0.2, 0.20182628407433365},
    {0.2, 0.20093695144385168},
    {0.1, 0.10411721816404285},
    {0.2, 0.20182628407433365},
    {0.2, 0.20093695144385168},
    {0.1, 0.10411721816404285},
    {0, 0},
    {0, 0},
    {0, 0},
  };

  std::vector<PConvert> weights2to3Vec =
  {
    {0.2, 0.20500015007570618},
    {0.2, 0.20500015007570618},
    {0.2, 0.20500015007570618},
    {0.1, 0.12173630913135866},
  };

  std::vector<size_t> idxToCheck1 = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<size_t> idxToCheck2 = {0, 1, 2, 3};

  std::vector<std::pair<std::pair<floatt, floatt>, floatt>> points =
  {
    {{0.44357233490399445, 0.22756905427903037}, 1},
  };

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (3, 4);
  setReValuesToMatrix (weights1to2, weights1to2Vec);

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (4, 1);
  setReValuesToMatrix (weights2to3, weights2to3Vec);

  l1->setHostWeights (weights1to2);
  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr hinputs = oap::host::NewReMatrix (1, 3);
  oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

  size_t idx = 0;
  for (const auto& p : points)
  {
    hinputs->reValues[0] = p.first.first;
    hinputs->reValues[1] = p.first.second;
    hinputs->reValues[2] = 1;

    houtput->reValues[0] = p.second;

    network->setInputs (hinputs, ArgType::HOST);
    network->setExpected (houtput, ArgType::HOST);

    network->forwardPropagation ();
    network->calculateErrors (oap::ErrorType::MEAN_SQUARE_ERROR);
  }

  network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR);
  network->backwardPropagation ();

  l1->getHostWeights (weights1to2);
  for (size_t idx = 0; idx < idxToCheck1.size(); ++idx)
  {
    size_t trueIdx = idxToCheck1[idx];
    EXPECT_NEAR (weights1to2Vec[trueIdx].second, weights1to2->reValues[trueIdx], 0.00000000001) << "Idx: " << idx;
  }

  l2->getHostWeights (weights2to3);
  for (size_t idx = 0; idx < idxToCheck2.size(); ++idx)
  {
    size_t trueIdx = idxToCheck2[idx];
    EXPECT_NEAR (weights2to3Vec[trueIdx].second, weights2to3->reValues[trueIdx], 0.00000000001) << "Idx: " << idx;
    EXPECT_DOUBLE_EQ (weights2to3Vec[trueIdx].second, weights2to3->reValues[trueIdx]) << "Idx: " << idx;
  }
}

TEST_F(OapNeuralTests, SimpleBackwardPropagation_1)
{
  using namespace oap::math;

  Layer* l1 = network->createLayer(2);
  Layer* l2 = network->createLayer(1);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 1);
  weights1to2->reValues[0] = 1;
  weights1to2->reValues[1] = 1;

  l1->setHostWeights (weights1to2);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 2);
  inputs->reValues[0] = 1;
  inputs->reValues[1] = 1;

  oap::HostMatrixPtr outputs = oap::host::NewReMatrix (1, 1);
  outputs->reValues[0] = sigmoid (2);

  network->setInputs (inputs, ArgType::HOST);
  network->setExpected (outputs, ArgType::HOST);

  network->forwardPropagation ();

  auto getLayerOutput = [](Layer* layer)
  {
    auto minfo = layer->getOutputsDim ();
    oap::HostMatrixPtr outputsL = oap::host::NewReMatrix (minfo.m_matrixDim.columns, minfo.m_matrixDim.rows);
    layer->getOutputs (outputsL, oap::HOST);
    return outputsL;
  };

  network->calculateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  EXPECT_EQ (0, network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  network->backwardPropagation ();

  oap::HostMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 1);
  l1->getHostWeights (bweights1to2);

  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[0]);
  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[1]);
}

TEST_F(OapNeuralTests, SimpleBackwardPropagation_2)
{
  using namespace oap::math;

  Layer* l1 = network->createLayer(2);
  Layer* l2 = network->createLayer(1);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 1);
  weights1to2->reValues[0] = 1;
  weights1to2->reValues[1] = 1;

  l1->setHostWeights (weights1to2);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 2);
  inputs->reValues[0] = 1;
  inputs->reValues[1] = 1;

  oap::HostMatrixPtr outputs = oap::host::NewReMatrix (1, 1);
  outputs->reValues[0] = sigmoid (2);

  network->setInputs (inputs, ArgType::HOST);
  network->setExpected (outputs, ArgType::HOST);

  network->forwardPropagation ();

  auto getLayerOutput = [](Layer* layer)
  {
    auto minfo = layer->getOutputsDim ();
    oap::HostMatrixPtr outputsL = oap::host::NewReMatrix (minfo.m_matrixDim.columns, minfo.m_matrixDim.rows);
    layer->getOutputs (outputsL, oap::HOST);
    return outputsL;
  };

  network->calculateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  EXPECT_EQ (0, network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  network->backwardPropagation ();

  oap::HostMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 1);
  l1->getHostWeights (bweights1to2);

  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[0]);
  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[1]);
}

TEST_F(OapNeuralTests, SimpleBackwardPropagation_3)
{
  using namespace oap::math;

  Layer* l1 = network->createLayer(2);
  Layer* l2 = network->createLayer(1);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 1);
  weights1to2->reValues[0] = 1;
  weights1to2->reValues[1] = 1;

  l1->setHostWeights (weights1to2);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  floatt input1 = 1;
  floatt input2 = 1;

  inputs->reValues[0] = input1;
  inputs->reValues[1] = input2;

  oap::HostMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
  floatt error = 0.001;
  expectedOutputs->reValues[0] = sigmoid (2) + error;

  floatt lr = 0.01;
  network->setLearningRate (lr);

  network->setInputs (inputs, ArgType::HOST);
  network->setExpected (expectedOutputs, ArgType::HOST);

  network->forwardPropagation ();

  auto getLayerOutput = [](Layer* layer)
  {
    auto minfo = layer->getOutputsDim ();
    oap::HostMatrixPtr outputsL = oap::host::NewReMatrix (minfo.m_matrixDim.columns, minfo.m_matrixDim.rows);
    layer->getOutputs (outputsL, oap::HOST);
    return outputsL;
  };

  network->calculateErrors (oap::ErrorType::MEAN_SQUARE_ERROR);
  logInfo ("BP %f", network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR));

  EXPECT_NEAR (error * error, network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR), 0.000000000001);
  network->backwardPropagation ();

  oap::HostMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 1);
  l1->getHostWeights (bweights1to2);

  EXPECT_DOUBLE_EQ (1 + lr * error * oap::math::dsigmoid (2) * input1, bweights1to2->reValues[0]);
  EXPECT_DOUBLE_EQ (1 + lr * error * oap::math::dsigmoid (2) * input2, bweights1to2->reValues[1]);
}

TEST_F(OapNeuralTests, SimpleBackwardPropagation_4)
{
  using namespace oap::math;

  Layer* l1 = network->createLayer(2);
  Layer* l2 = network->createLayer(1);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 1);
  weights1to2->reValues[0] = 1;
  weights1to2->reValues[1] = 1;

  l1->setHostWeights (weights1to2);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  std::vector<floatt> inputsVec = {2, 1};

  inputs->reValues[0] = inputsVec[0];
  inputs->reValues[1] = inputsVec[1];

  oap::HostMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
  floatt error = 0.001;
  expectedOutputs->reValues[0] = sigmoid (3) + error;

  floatt lr = 0.01;
  network->setLearningRate (lr);

  network->setInputs (inputs, ArgType::HOST);
  network->setExpected (expectedOutputs, ArgType::HOST);

  network->forwardPropagation ();

  network->calculateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  EXPECT_DOUBLE_EQ (error, network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  network->backwardPropagation ();

  oap::HostMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 1);
  l1->getHostWeights (bweights1to2);

  EXPECT_DOUBLE_EQ (1 + lr * error * oap::math::dsigmoid (3) * inputsVec[0], bweights1to2->reValues[0]);
  EXPECT_DOUBLE_EQ (1 + lr * error * oap::math::dsigmoid (3) * inputsVec[1], bweights1to2->reValues[1]);
}

TEST_F(OapNeuralTests, SimpleBackwardPropagation_5)
{
  using namespace oap::math;

  Layer* l1 = network->createLayer(2);
  Layer* l2 = network->createLayer(3);
  Layer* l3 = network->createLayer(1);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 3);

  setReValuesToMatrix (weights1to2, {1,1,1,1,1,1});

  l1->setHostWeights (weights1to2);

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);

  setReValuesToMatrix (weights2to3, {1,1,1});

  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  std::vector<floatt> inputsVec = {1, 1};

  inputs->reValues[0] = inputsVec[0];
  inputs->reValues[1] = inputsVec[1];

  oap::HostMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
  expectedOutputs->reValues[0] = sigmoid (sigmoid (2) + sigmoid (2) + sigmoid (2));

  floatt lr = 0.01;
  network->setLearningRate (lr);

  network->setInputs (inputs, ArgType::HOST);
  network->setExpected (expectedOutputs, ArgType::HOST);

  network->forwardPropagation ();

  network->calculateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  EXPECT_DOUBLE_EQ (0, network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  network->backwardPropagation ();

  oap::HostMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 3);
  l1->getHostWeights (bweights1to2);

  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[0]);
  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[1]);

  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[2]);
  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[3]);

  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[4]);
  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[5]);
}

TEST_F(OapNeuralTests, SimpleBackwardPropagation_5_Batch_1)
{
  using namespace oap::math;

  Layer* l1 = network->createLayer(2);
  Layer* l2 = network->createLayer(3);
  Layer* l3 = network->createLayer(1);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 3);

  setReValuesToMatrix (weights1to2, {1,1,1,1,1,1});

  l1->setHostWeights (weights1to2);

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);

  setReValuesToMatrix (weights2to3, {1,1,1});

  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  std::vector<floatt> inputsVec = {1, 1};

  inputs->reValues[0] = inputsVec[0];
  inputs->reValues[1] = inputsVec[1];

  oap::HostMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
  expectedOutputs->reValues[0] = sigmoid (sigmoid (2) + sigmoid (2) + sigmoid (2));

  floatt lr = 0.01;
  network->setLearningRate (lr);

  for (size_t i = 0; i < 100; ++i)
  {
    network->setInputs (inputs, ArgType::HOST);
    network->setExpected (expectedOutputs, ArgType::HOST);

    network->forwardPropagation ();

    network->calculateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
  }
  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  EXPECT_DOUBLE_EQ (0, network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  network->backwardPropagation ();

  oap::HostMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 3);
  l1->getHostWeights (bweights1to2);

  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[0]);
  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[1]);

  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[2]);
  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[3]);

  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[4]);
  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[5]);
}

TEST_F(OapNeuralTests, SimpleBackwardPropagation_5_Batch_2)
{
  using namespace oap::math;

  Layer* l1 = network->createLayer(2);
  Layer* l2 = network->createLayer(3);
  Layer* l3 = network->createLayer(1);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 3);

  setReValuesToMatrix (weights1to2, {1,1,1,1,1,1});

  l1->setHostWeights (weights1to2);

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);

  setReValuesToMatrix (weights2to3, {1,1,1});

  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  std::vector<std::pair<floatt, floatt>> inputsVec = {{1, 1}, {2, 2}, {3, 3}};

  oap::HostMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
  std::vector<floatt> expectedOutputsVec;
  for (const auto& pair : inputsVec)
  {
    floatt sum = pair.first + pair.second;
    expectedOutputsVec.emplace_back (sigmoid (sigmoid (sum) + sigmoid (sum) + sigmoid (sum)));
  }

  floatt lr = 0.01;
  network->setLearningRate (lr);

  for (size_t i = 0; i < inputsVec.size(); ++i)
  {
    inputs->reValues[0] = inputsVec[i].first;
    inputs->reValues[1] = inputsVec[i].second;

    network->setInputs (inputs, ArgType::HOST);

    expectedOutputs->reValues[0] = expectedOutputsVec[i];
    network->setExpected (expectedOutputs, ArgType::HOST);

    network->forwardPropagation ();

    network->calculateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
  }

  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  EXPECT_DOUBLE_EQ (0, network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  network->backwardPropagation ();

  oap::HostMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 3);
  l1->getHostWeights (bweights1to2);

  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[0]);
  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[1]);

  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[2]);
  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[3]);

  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[4]);
  EXPECT_DOUBLE_EQ (1, bweights1to2->reValues[5]);
}

TEST_F(OapNeuralTests, SimpleBackwardPropagation_5_Batch_3)
{
  using namespace oap::math;

  Layer* l1 = network->createLayer(2);
  Layer* l2 = network->createLayer(3);
  Layer* l3 = network->createLayer(1);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 3);

  setReValuesToMatrix (weights1to2, {1,1,1,1,1,1});

  l1->setHostWeights (weights1to2);

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);

  setReValuesToMatrix (weights2to3, {1,1,1});

  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  std::vector<std::pair<floatt, floatt>> inputsVec = {{1, 1}, {2, 2}, {3, 3}};

  oap::HostMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
  std::vector<floatt> expectedOutputsVec;
  for (const auto& pair : inputsVec)
  {
    floatt sum = pair.first + pair.second;
    expectedOutputsVec.emplace_back (sigmoid (sigmoid (sum) + sigmoid (sum) + sigmoid (sum)));
  }

  floatt error = 1;
  expectedOutputsVec[0] += error;

  floatt lr = 0.01;
  network->setLearningRate (lr);

  for (size_t i = 0; i < inputsVec.size(); ++i)
  {
    inputs->reValues[0] = inputsVec[i].first;
    inputs->reValues[1] = inputsVec[i].second;

    network->setInputs (inputs, ArgType::HOST);

    expectedOutputs->reValues[0] = expectedOutputsVec[i];
    network->setExpected (expectedOutputs, ArgType::HOST);

    network->forwardPropagation ();

    network->calculateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
  }

  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
}

TEST_F(OapNeuralTests, DISABLED_SimpleBackwardPropagation_6)
{
  using namespace oap::math;

  Layer* l1 = network->createLayer(2);
  Layer* l2 = network->createLayer(3);
  Layer* l3 = network->createLayer(1);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 3);

  setReValuesToMatrix (weights1to2, {1,1,1,1,1,1});

  l1->setHostWeights (weights1to2);

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);

  setReValuesToMatrix (weights2to3, {1,1,1});

  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  std::vector<floatt> inputsVec = {1, 1};

  for (size_t idx = 0; idx < inputsVec.size(); ++idx)
  {
    inputs->reValues[idx] = inputsVec[idx];
  }

  oap::HostMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
  floatt error = 0.001;

  std::vector<floatt> activs1to2 = {sigmoid(2), sigmoid(2)};
  std::vector<floatt> activs2to3 = {sigmoid(2), sigmoid(2), sigmoid(2)};

  expectedOutputs->reValues[0] = sigmoid (sum (activs2to3)) + error;

  floatt lr = 0.01;
  network->setLearningRate (lr);

  network->setInputs (inputs, ArgType::HOST);
  network->setExpected (expectedOutputs, ArgType::HOST);

  network->forwardPropagation ();

  network->calculateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  EXPECT_DOUBLE_EQ (error, network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  network->backwardPropagation ();

  oap::HostMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 3);
  l1->getHostWeights (bweights1to2);

  oap::HostMatrixPtr bweights2to3 = oap::host::NewReMatrix (3, 1);
  l2->getHostWeights (bweights2to3);

  floatt limit = 0.000001;

  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sum(activs1to2)) * inputsVec[0], bweights1to2->reValues[0], limit);
  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sum(activs1to2)) * inputsVec[1], bweights1to2->reValues[1], limit);

  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sum(activs1to2)) * inputsVec[0], bweights1to2->reValues[2], limit);
  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sum(activs1to2)) * inputsVec[1], bweights1to2->reValues[3], limit);

  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sum(activs1to2)) * inputsVec[0], bweights1to2->reValues[4], limit);
  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sum(activs1to2)) * inputsVec[1], bweights1to2->reValues[5], limit);

  EXPECT_NEAR (1 + lr * error * dsigmoid (sum(activs2to3)) * sigmoid(2), bweights2to3->reValues[0], limit);
  EXPECT_NEAR (1 + lr * error * dsigmoid (sum(activs2to3)) * sigmoid(2), bweights2to3->reValues[1], limit);
  EXPECT_NEAR (1 + lr * error * dsigmoid (sum(activs2to3)) * sigmoid(2), bweights2to3->reValues[2], limit);
}

TEST_F(OapNeuralTests, SimpleBackwardPropagation_7)
{
  using namespace oap::math;

  Layer* l1 = network->createLayer(2);
  Layer* l2 = network->createLayer(3);
  Layer* l3 = network->createLayer(1);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 3);

  setReValuesToMatrix (weights1to2, {2, 1, 1, 1, 1, 1});

  l1->setHostWeights (weights1to2);

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);

  setReValuesToMatrix(weights2to3, {1, 1, 1});

  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  std::vector<floatt> inputsVec = {1, 1};

  setReValuesToMatrix (inputs, inputsVec);

  oap::HostMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
  floatt error = 0.001;
  expectedOutputs->reValues[0] = sigmoid (sigmoid (3) + sigmoid (2) + sigmoid (2)) + error;

  floatt lr = 0.01;
  network->setLearningRate (lr);

  network->setInputs (inputs, ArgType::HOST);
  network->setExpected (expectedOutputs, ArgType::HOST);

  network->forwardPropagation ();

  network->calculateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  EXPECT_DOUBLE_EQ (error, network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  network->backwardPropagation ();

  oap::HostMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 3);
  l1->getHostWeights (bweights1to2);

  oap::HostMatrixPtr bweights2to3 = oap::host::NewReMatrix (3, 1);
  l2->getHostWeights (bweights2to3);

  floatt limit = 0.00001;

  EXPECT_NEAR (2 + lr * (error * 1) * dsigmoid (sigmoid(3) + sigmoid(2)) * inputsVec[0], bweights1to2->reValues[0], limit);
  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[1], bweights1to2->reValues[1], limit);

  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[0], bweights1to2->reValues[2], limit);
  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[1], bweights1to2->reValues[3], limit);

  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[0], bweights1to2->reValues[4], limit);
  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[1], bweights1to2->reValues[5], limit);

  EXPECT_NEAR (1 + lr * error * dsigmoid (sigmoid(3) + sigmoid(2) + sigmoid(2)) * sigmoid(3), bweights2to3->reValues[0], limit);
  EXPECT_NEAR (1 + lr * error * dsigmoid (sigmoid(3) + sigmoid(2) + sigmoid(2)) * sigmoid(2), bweights2to3->reValues[1], limit);
  EXPECT_NEAR (1 + lr * error * dsigmoid (sigmoid(3) + sigmoid(2) + sigmoid(2)) * sigmoid(2), bweights2to3->reValues[2], limit);
}

TEST_F(OapNeuralTests, SimpleBackwardPropagation_8)
{
  using namespace oap::math;

  Layer* l1 = network->createLayer(2);
  Layer* l2 = network->createLayer(3);
  Layer* l3 = network->createLayer(1);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 3);

  setReValuesToMatrix (weights1to2, {2, 1, 1, 1, 1, 1});

  l1->setHostWeights (weights1to2);

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);

  setReValuesToMatrix(weights2to3, {1, 1, 1});

  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  std::vector<floatt> inputsVec = {1, 1};

  setReValuesToMatrix (inputs, inputsVec);

  oap::HostMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
  floatt error = 0.001;
  expectedOutputs->reValues[0] = sigmoid (sigmoid (3) + sigmoid (2) + sigmoid (2)) + error;

  floatt lr = 0.01;
  network->setLearningRate (lr);

  network->setInputs (inputs, ArgType::HOST);
  network->setExpected (expectedOutputs, ArgType::HOST);

  network->forwardPropagation ();

  network->calculateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  EXPECT_DOUBLE_EQ (error, network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  network->backwardPropagation ();

  oap::HostMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 3);
  l1->getHostWeights (bweights1to2);

  oap::HostMatrixPtr bweights2to3 = oap::host::NewReMatrix (3, 1);
  l2->getHostWeights (bweights2to3);

  floatt limit = 0.00001;

  EXPECT_NEAR (2 + lr * (error * 1) * dsigmoid (sigmoid(3) + sigmoid(2)) * inputsVec[0], bweights1to2->reValues[0], limit);
  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[1], bweights1to2->reValues[1], limit);

  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[0], bweights1to2->reValues[2], limit);
  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[1], bweights1to2->reValues[3], limit);

  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[0], bweights1to2->reValues[4], limit);
  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[1], bweights1to2->reValues[5], limit);

  EXPECT_NEAR (1 + lr * error * dsigmoid (sigmoid(3) + sigmoid(2) + sigmoid(2)) * sigmoid(3), bweights2to3->reValues[0], limit);
  EXPECT_NEAR (1 + lr * error * dsigmoid (sigmoid(3) + sigmoid(2) + sigmoid(2)) * sigmoid(2), bweights2to3->reValues[1], limit);
  EXPECT_NEAR (1 + lr * error * dsigmoid (sigmoid(3) + sigmoid(2) + sigmoid(2)) * sigmoid(2), bweights2to3->reValues[2], limit);
}
