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
#include <random>
#include "gtest/gtest.h"
#include "CuProceduresApi.h"
#include "MultiMatricesCuProcedures.h"
#include "KernelExecutor.h"
#include "MatchersUtils.h"
#include "MathOperationsCpu.h"

#include "oapCudaMatrixUtils.h"
#include "oapDeviceComplexMatrixUPtr.h"
#include "oapHostMatrixUtils.h"
#include "oapNetwork.h"
#include "oapNetworkCudaApi.h"
#include "oapFunctions.h"
#include "PyPlot.h"
#include "Config.h"

namespace
{
class NetworkT : public oap::Network
{
  public:
    NetworkT(oap::CuProceduresApi* single, oap::MultiMatricesCuProcedures* multi, oap::NetworkCudaApi* nca, bool p) : oap::Network(single, multi, nca, p)  {}

    void setHostInput (math::ComplexMatrix* inputs, size_t index)
    {
      oap::Network::setHostInputs (inputs, index);
    }
};
}

class OapNeuralTests_TinyTests : public testing::Test
{
 public:
  CUresult status;
  NetworkT* network;

  virtual void SetUp()
  {
    oap::cuda::Context::Instance().create();
    network = nullptr;

    auto* singleApi = new oap::CuProceduresApi();
    auto* multiApi = new oap::MultiMatricesCuProcedures (singleApi);
    auto* nca = new oap::NetworkCudaApi ();
    network = new NetworkT(singleApi, multiApi, nca, true);
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
      OapNeuralTests_TinyTests* m_ont;
      floatt m_bvalue;
    public:
      Runner(bool hasBias, OapNeuralTests_TinyTests* ont, floatt bvalue = 1.f): m_hasBias(hasBias), m_ont(ont), m_bvalue(bvalue)
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
        *GetRePtrIndex (inputs, 0) = a1;
        *GetRePtrIndex (inputs, 1) = a2;

        if (m_hasBias)
        {
          *GetRePtrIndex (inputs, 2) = m_bvalue;
        }

        *GetRePtrIndex (expected, 0) = e1;

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
        *GetRePtrIndex (inputs, 0) = a1;
        *GetRePtrIndex (inputs, 1) = a2;

        if (m_hasBias)
        {
          *GetRePtrIndex (inputs, 2) = m_bvalue;
        }

        auto output = m_ont->network->run (inputs, ArgType::HOST, oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
        return m_ont->is(GetReIndex (output, 0));
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

  void testForwardPropagation_2_to_1 (floatt w_1, floatt w_2, floatt i_1, floatt i_2)
  {
    oap::Layer* l1 = network->createLayer(2);
    network->createLayer(1);

    network->setLearningRate (1);

    oap::HostMatrixUPtr hw = oap::host::NewReMatrix (2, 1);
    oap::HostMatrixUPtr hinputs = oap::host::NewReMatrix (1, 2);

    floatt hw_1 = w_1;
    floatt hw_2 = w_2;

    *GetRePtrIndex (hw, 0) = hw_1;
    *GetRePtrIndex (hw, 1) = hw_2;

    *GetRePtrIndex (hinputs, 0) = i_1;
    *GetRePtrIndex (hinputs, 1) = i_2;

    l1->setHostWeights (hw.get ());

    auto output = network->run (hinputs, ArgType::HOST, oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);

    EXPECT_THAT(GetReIndex (output, 0), testing::DoubleNear(oap::math::sigmoid(hw_1 * i_1 + hw_2 * i_2), 0.0001));
    EXPECT_EQ(1, gColumns (output));
    EXPECT_EQ(1, gRows (output));
  }

  void testBackPropagation_1_to_2(floatt w_1, floatt w_2, floatt i_1, floatt i_2, floatt e_1)
  {
    oap::Layer* l1 = network->createLayer(2);
    oap::Layer* l2 = network->createLayer(1);

    network->setLearningRate (1);

    oap::HostMatrixUPtr hw = oap::host::NewReMatrix (2, 1);
    oap::HostMatrixUPtr io = oap::host::NewReMatrix (1, 2);
    oap::HostMatrixUPtr io1 = oap::host::NewReMatrix (1, 1);
    oap::HostMatrixUPtr e1 = oap::host::NewReMatrix (1, 1);
    oap::DeviceMatrixUPtr de1 = oap::cuda::NewDeviceReMatrix(1, 1);

    floatt hw_1 = w_1;
    floatt hw_2 = w_2;

    *GetRePtrIndex (hw, 0) = hw_1;
    *GetRePtrIndex (hw, 1) = hw_2;

    *GetRePtrIndex (io, 0) = i_1;
    *GetRePtrIndex (io, 1) = i_2;

    floatt i1_1 = oap::math::sigmoid(i_1 * hw_1 + i_2 * hw_2);
    *GetRePtrIndex (e1, 0) = e_1;

    oap::cuda::CopyHostMatrixToDeviceMatrix (de1, e1);

    l1->setHostWeights (hw.get ());

    network->setHostInput (io, 0);
    network->train (io, e1, ArgType::HOST, oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);

    *GetRePtrIndex (hw, 0) = 0;
    *GetRePtrIndex (hw, 1) = 0;

    network->getHostWeights(hw, 0);

    floatt sigma = e_1 - i1_1;
    floatt ds = oap::math::dsigmoid(i_1 * hw_1 + i_2 * hw_2);

    floatt c1 = ds * sigma * i_1;
    floatt c2 = ds * sigma * i_2;

    EXPECT_THAT(GetReIndex (hw, 0) - hw_1, testing::DoubleNear(c1, 0.0001));
    EXPECT_THAT(GetReIndex (hw, 1) - hw_2, testing::DoubleNear(c2, 0.0001));

    l1->printHostWeights (true);
    l2->printHostWeights (true);
  }
};

TEST_F(OapNeuralTests_TinyTests, ForwardPropagation_1)
{
  testForwardPropagation_2_to_1 (1, 1, 1, 1);
}

TEST_F(OapNeuralTests_TinyTests, ForwardPropagation_2)
{
  testForwardPropagation_2_to_1 (1, 2, 3, 4);
}

TEST_F(OapNeuralTests_TinyTests, ForwardPropagation_3)
{
  testForwardPropagation_2_to_1 (11, 22, 33, 44);
}

TEST_F(OapNeuralTests_TinyTests, ForwardPropagation_4)
{
  testForwardPropagation_2_to_1 (111, 221, 331, 441);
}

TEST_F(OapNeuralTests_TinyTests, BackPropagation_1)
{
  testBackPropagation_1_to_2 (1, 1, 1, 1, 1);
}

TEST_F(OapNeuralTests_TinyTests, BackPropagation_2)
{
  testBackPropagation_1_to_2 (2, 1, 1, 1, 0);
}

TEST_F(OapNeuralTests_TinyTests, BackPropagation_3)
{
  testBackPropagation_1_to_2 (2, 1, 3, 2, 1);
}

TEST_F(OapNeuralTests_TinyTests, BackPropagation_4)
{
  testBackPropagation_1_to_2 (1, 2, 3, 4, 5);
}

TEST_F(OapNeuralTests_TinyTests, SaveLoadBufferTest)
{
  bool isbias = true;

  oap::Layer* l1 = network->createLayer(isbias ? 3 : 2);
  oap::Layer* l2 = network->createLayer(6);
  oap::Layer* l3 = network->createLayer(1);

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

  oap::utils::ByteBuffer buffer;
  //network->save (buffer);

  //std::unique_ptr<oap::Network> cnetwork (oap::Network::load (buffer));

  //EXPECT_TRUE (*network == *cnetwork);
}

TEST_F(OapNeuralTests_TinyTests, SaveLoadFileTest)
{
  bool isbias = true;

  oap::Layer* l1 = network->createLayer(isbias ? 3 : 2);
  oap::Layer* l2 = network->createLayer(6);
  oap::Layer* l3 = network->createLayer(1);

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

  std::string path = "device_tests/OapNeuralTests_TinyTests_SaveLoadFileTest.bin";
  path = oap::utils::Config::getFileInTmp (path);
/*
  auto save = [&]()
  {
    utils::ByteBuffer buffer;
    network->save (buffer);
    buffer.fwrite (path);
  };

  auto load = [&]() -> std::unique_ptr<oap::Network>
  {
    utils::ByteBuffer buffer (path);
    return std::unique_ptr<oap::Network> (oap::Network::load (buffer));
  };

  save ();
  auto cnetwork = load ();

  EXPECT_TRUE (*network == *cnetwork);
*/
}
