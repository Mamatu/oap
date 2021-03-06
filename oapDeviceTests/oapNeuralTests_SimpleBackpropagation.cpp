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
#include "MultiMatricesCuProcedures.h"
#include "KernelExecutor.h"
#include "MatchersUtils.h"
#include "MathOperationsCpu.h"

#include "oapCudaMatrixUtils.h"
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
    NetworkT (oap::CuProceduresApi* single, oap::MultiMatricesCuProcedures* multi, oap::NetworkCudaApi* nga, bool p) : oap::Network (single, multi, nga, p)
    {}

    void setHostInput (math::ComplexMatrix* inputs, size_t index)
    {
      oap::Network::setHostInputs (inputs, index);
    }
};
}

class OapNeuralTests_SimpleBackpropagation : public testing::Test
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
    auto* nga = new oap::NetworkCudaApi ();
    network = new NetworkT(singleApi, multiApi, nga, true);
  }

  virtual void TearDown()
  {
    delete network;
    network = nullptr;
    oap::cuda::Context::Instance().destroy();
  }
};

TEST_F(OapNeuralTests_SimpleBackpropagation, Test_1)
{
  using namespace oap::math;

  oap::Layer* l1 = network->createLayer(2);
  oap::Layer* l2 = network->createLayer(1);

  oap::HostComplexMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 1);
  *GetRePtrIndex (weights1to2, 0) = 1;
  *GetRePtrIndex (weights1to2, 1) = 1;

  l1->setHostWeights (weights1to2);

  oap::HostComplexMatrixPtr inputs = oap::host::NewReMatrix (1, 2);
  *GetRePtrIndex (inputs, 0) = 1;
  *GetRePtrIndex (inputs, 1) = 1;

  oap::HostComplexMatrixPtr outputs = oap::host::NewReMatrix (1, 1);
  *GetRePtrIndex (outputs, 0) = sigmoid (2);

  network->setInputs (inputs, ArgType::HOST);
  network->setExpected (outputs, ArgType::HOST);

  network->forwardPropagation ();

  auto getLayerOutput = [](oap::Layer* layer)
  {
    auto minfo = layer->getOutputsInfo ();
    oap::HostComplexMatrixPtr outputsL = oap::host::NewReMatrix (minfo.columns (), minfo.rows ());
    layer->getOutputs (outputsL, ArgType::HOST);
    return outputsL;
  };

  network->accumulateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR, CalculationType::HOST);
  network->backPropagation ();

  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  EXPECT_DOUBLE_EQ (0, network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  network->updateWeights ();

  oap::HostComplexMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 1);
  l1->getHostWeights (bweights1to2);

  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 0));
  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 1));
}

TEST_F(OapNeuralTests_SimpleBackpropagation, Test_2)
{
  using namespace oap::math;

  oap::Layer* l1 = network->createLayer(2);
  oap::Layer* l2 = network->createLayer(1);

  oap::HostComplexMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 1);
  *GetRePtrIndex (weights1to2, 0) = 1;
  *GetRePtrIndex (weights1to2, 1) = 1;

  l1->setHostWeights (weights1to2);

  oap::HostComplexMatrixPtr inputs = oap::host::NewReMatrix (1, 2);
  *GetRePtrIndex (inputs, 0) = 1;
  *GetRePtrIndex (inputs, 1) = 1;

  oap::HostComplexMatrixPtr outputs = oap::host::NewReMatrix (1, 1);
  *GetRePtrIndex (outputs, 0) = sigmoid (2);

  network->setInputs (inputs, ArgType::HOST);
  network->setExpected (outputs, ArgType::HOST);

  network->forwardPropagation ();

  auto getLayerOutput = [](oap::Layer* layer)
  {
    auto minfo = layer->getOutputsInfo ();
    oap::HostComplexMatrixPtr outputsL = oap::host::NewReMatrix (minfo.columns (), minfo.rows ());
    layer->getOutputs (outputsL, ArgType::HOST);
    return outputsL;
  };

  network->accumulateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR, CalculationType::HOST);
  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  EXPECT_DOUBLE_EQ (0, network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  network->backPropagation ();
  network->updateWeights ();

  oap::HostComplexMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 1);
  l1->getHostWeights (bweights1to2);

  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 0));
  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 1));
}

TEST_F(OapNeuralTests_SimpleBackpropagation, Test_3)
{
  using namespace oap::math;

  oap::Layer* l1 = network->createLayer(2);
  oap::Layer* l2 = network->createLayer(1);

  oap::HostComplexMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 1);
  *GetRePtrIndex (weights1to2, 0) = 1;
  *GetRePtrIndex (weights1to2, 1) = 1;

  l1->setHostWeights (weights1to2);

  oap::HostComplexMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  floatt input1 = 1;
  floatt input2 = 1;

  *GetRePtrIndex (inputs, 0) = input1;
  *GetRePtrIndex (inputs, 1) = input2;

  oap::HostComplexMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
  floatt error = 0.001;
  *GetRePtrIndex (expectedOutputs, 0) = sigmoid (2) + error;

  floatt lr = 0.01;
  network->setLearningRate (lr);

  network->setInputs (inputs, ArgType::HOST);
  network->setExpected (expectedOutputs, ArgType::HOST);

  network->forwardPropagation ();

  network->accumulateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR, CalculationType::HOST);
  network->backPropagation ();

  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  EXPECT_NEAR (error * error * 0.5, network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR), 0.000000000001);

  network->updateWeights ();

  oap::HostComplexMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 1);
  l1->getHostWeights (bweights1to2);

  EXPECT_DOUBLE_EQ (1 + lr * error * oap::math::dsigmoid (2) * input1, GetReIndex (bweights1to2, 0));
  EXPECT_DOUBLE_EQ (1 + lr * error * oap::math::dsigmoid (2) * input2, GetReIndex (bweights1to2, 1));
}

TEST_F(OapNeuralTests_SimpleBackpropagation, Test_4)
{
  using namespace oap::math;

  oap::Layer* l1 = network->createLayer(2);
  oap::Layer* l2 = network->createLayer(1);

  oap::HostComplexMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 1);
  *GetRePtrIndex (weights1to2, 0) = 1;
  *GetRePtrIndex (weights1to2, 1) = 1;

  l1->setHostWeights (weights1to2);

  oap::HostComplexMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  std::vector<floatt> inputsVec = {2, 1};

  *GetRePtrIndex (inputs, 0) = inputsVec[0];
  *GetRePtrIndex (inputs, 1) = inputsVec[1];

  oap::HostComplexMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
  floatt error = 0.001;
  *GetRePtrIndex (expectedOutputs, 0) = sigmoid (3) + error;

  floatt lr = 0.01;
  network->setLearningRate (lr);

  network->setInputs (inputs, ArgType::HOST);
  network->setExpected (expectedOutputs, ArgType::HOST);

  network->forwardPropagation ();

  network->accumulateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR, CalculationType::HOST);
  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  EXPECT_FLOAT_EQ (sqrt(error * error * 0.5), network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  network->backPropagation ();
  network->updateWeights ();

  oap::HostComplexMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 1);
  l1->getHostWeights (bweights1to2);

  EXPECT_DOUBLE_EQ (1 + lr * error * oap::math::dsigmoid (3) * inputsVec[0], GetReIndex (bweights1to2, 0));
  EXPECT_DOUBLE_EQ (1 + lr * error * oap::math::dsigmoid (3) * inputsVec[1], GetReIndex (bweights1to2, 1));
}

TEST_F(OapNeuralTests_SimpleBackpropagation, Test_5)
{
  using namespace oap::math;

  oap::Layer* l1 = network->createLayer(2);
  oap::Layer* l2 = network->createLayer(3);
  oap::Layer* l3 = network->createLayer(1);

  oap::HostComplexMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 3);

  oap::host::SetReValuesToMatrix (weights1to2, {1,1,1,1,1,1});

  l1->setHostWeights (weights1to2);

  oap::HostComplexMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);

  oap::host::SetReValuesToMatrix (weights2to3, {1,1,1});

  l2->setHostWeights (weights2to3);

  oap::HostComplexMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  std::vector<floatt> inputsVec = {1, 1};

  *GetRePtrIndex (inputs, 0) = inputsVec[0];
  *GetRePtrIndex (inputs, 1) = inputsVec[1];

  oap::HostComplexMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
  *GetRePtrIndex (expectedOutputs, 0) = sigmoid (sigmoid (2) + sigmoid (2) + sigmoid (2));

  floatt lr = 0.01;
  network->setLearningRate (lr);

  network->setInputs (inputs, ArgType::HOST);
  network->setExpected (expectedOutputs, ArgType::HOST);

  network->forwardPropagation ();

  network->accumulateErrors (oap::ErrorType::MEAN_SQUARE_ERROR, CalculationType::HOST);
  logInfo ("BP %f", network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR));

  EXPECT_DOUBLE_EQ (0, network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR));
  network->backPropagation ();
  network->updateWeights ();

  oap::HostComplexMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 3);
  l1->getHostWeights (bweights1to2);

  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 0));
  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 1));

  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 2));
  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 3));

  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 4));
  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 5));
}

TEST_F(OapNeuralTests_SimpleBackpropagation, Test_5_Batch_1)
{
  using namespace oap::math;

  oap::Layer* l1 = network->createLayer(2);
  oap::Layer* l2 = network->createLayer(3);
  oap::Layer* l3 = network->createLayer(1);

  oap::HostComplexMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 3);

  oap::host::SetReValuesToMatrix (weights1to2, {1,1,1,1,1,1});

  l1->setHostWeights (weights1to2);

  oap::HostComplexMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);

  oap::host::SetReValuesToMatrix (weights2to3, {1,1,1});

  l2->setHostWeights (weights2to3);

  oap::HostComplexMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  std::vector<floatt> inputsVec = {1, 1};

  *GetRePtrIndex (inputs, 0) = inputsVec[0];
  *GetRePtrIndex (inputs, 1) = inputsVec[1];

  oap::HostComplexMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
  *GetRePtrIndex (expectedOutputs, 0) = sigmoid (sigmoid (2) + sigmoid (2) + sigmoid (2));

  floatt lr = 0.01;
  network->setLearningRate (lr);

  for (size_t i = 0; i < 100; ++i)
  {
    network->setInputs (inputs, ArgType::HOST);
    network->setExpected (expectedOutputs, ArgType::HOST);

    network->forwardPropagation ();

    network->accumulateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR, CalculationType::HOST);

    network->backPropagation ();
  }
  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  EXPECT_DOUBLE_EQ (0, network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  network->updateWeights ();

  oap::HostComplexMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 3);
  l1->getHostWeights (bweights1to2);

  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 0));
  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 1));

  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 2));
  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 3));

  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 4));
  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 5));
}

TEST_F(OapNeuralTests_SimpleBackpropagation, Test_5_Batch_2)
{
  using namespace oap::math;

  oap::Layer* l1 = network->createLayer(2);
  oap::Layer* l2 = network->createLayer(3);
  oap::Layer* l3 = network->createLayer(1);

  oap::HostComplexMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 3);

  oap::host::SetReValuesToMatrix (weights1to2, {1,1,1,1,1,1});

  l1->setHostWeights (weights1to2);

  oap::HostComplexMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);

  oap::host::SetReValuesToMatrix (weights2to3, {1,1,1});

  l2->setHostWeights (weights2to3);

  oap::HostComplexMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  std::vector<std::pair<floatt, floatt>> inputsVec = {{1, 1}, {2, 2}, {3, 3}};

  oap::HostComplexMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
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
    *GetRePtrIndex (inputs, 0) = inputsVec[i].first;
    *GetRePtrIndex (inputs, 1) = inputsVec[i].second;

    network->setInputs (inputs, ArgType::HOST);

    *GetRePtrIndex (expectedOutputs, 0) = expectedOutputsVec[i];
    network->setExpected (expectedOutputs, ArgType::HOST);

    network->forwardPropagation ();

    network->accumulateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR, CalculationType::HOST);

    network->backPropagation ();
  }

  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  EXPECT_DOUBLE_EQ (0, network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  network->updateWeights ();

  oap::HostComplexMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 3);
  l1->getHostWeights (bweights1to2);

  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 0));
  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 1));

  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 2));
  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 3));

  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 4));
  EXPECT_DOUBLE_EQ (1, GetReIndex (bweights1to2, 5));
}

TEST_F(OapNeuralTests_SimpleBackpropagation, Test_5_Batch_3)
{
  using namespace oap::math;

  oap::Layer* l1 = network->createLayer(2);
  oap::Layer* l2 = network->createLayer(3);
  oap::Layer* l3 = network->createLayer(1);

  oap::HostComplexMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 3);

  oap::host::SetReValuesToMatrix (weights1to2, {1,1,1,1,1,1});

  l1->setHostWeights (weights1to2);

  oap::HostComplexMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);

  oap::host::SetReValuesToMatrix (weights2to3, {1,1,1});

  l2->setHostWeights (weights2to3);

  oap::HostComplexMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  std::vector<std::pair<floatt, floatt>> inputsVec = {{1, 1}, {2, 2}, {3, 3}};

  oap::HostComplexMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
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
    *GetRePtrIndex (inputs, 0) = inputsVec[i].first;
    *GetRePtrIndex (inputs, 1) = inputsVec[i].second;

    network->setInputs (inputs, ArgType::HOST);

    *GetRePtrIndex (expectedOutputs, 0) = expectedOutputsVec[i];
    network->setExpected (expectedOutputs, ArgType::HOST);

    network->forwardPropagation ();

    network->accumulateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR, CalculationType::HOST);

    network->backPropagation ();
  }

  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  network->updateWeights ();
}

TEST_F(OapNeuralTests_SimpleBackpropagation, Test_7)
{
  using namespace oap::math;

  oap::Layer* l1 = network->createLayer(2);
  oap::Layer* l2 = network->createLayer(3);
  oap::Layer* l3 = network->createLayer(1);

  oap::HostComplexMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 3);

  oap::host::SetReValuesToMatrix (weights1to2, {2, 1, 1, 1, 1, 1});

  l1->setHostWeights (weights1to2);

  oap::HostComplexMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);

  oap::host::SetReValuesToMatrix(weights2to3, {1, 1, 1});

  l2->setHostWeights (weights2to3);

  oap::HostComplexMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  std::vector<floatt> inputsVec = {1, 1};

  oap::host::SetReValuesToMatrix (inputs, inputsVec);

  oap::HostComplexMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
  floatt error = 0.001;
  *GetRePtrIndex (expectedOutputs, 0) = sigmoid (sigmoid (3) + sigmoid (2) + sigmoid (2)) + error;

  floatt lr = 0.01;
  network->setLearningRate (lr);

  network->setInputs (inputs, ArgType::HOST);
  network->setExpected (expectedOutputs, ArgType::HOST);

  network->forwardPropagation ();

  network->accumulateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR, CalculationType::HOST);
  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  EXPECT_FLOAT_EQ (sqrt (error * error * 0.5), network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  network->backPropagation ();
  network->updateWeights ();

  oap::HostComplexMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 3);
  l1->getHostWeights (bweights1to2);

  oap::HostComplexMatrixPtr bweights2to3 = oap::host::NewReMatrix (3, 1);
  l2->getHostWeights (bweights2to3);

  floatt limit = 0.00001;

  EXPECT_NEAR (2 + lr * (error * 1) * dsigmoid (sigmoid(3) + sigmoid(2)) * inputsVec[0], GetReIndex (bweights1to2, 0), limit);
  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[1], GetReIndex (bweights1to2, 1), limit);

  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[0], GetReIndex (bweights1to2, 2), limit);
  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[1], GetReIndex (bweights1to2, 3), limit);

  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[0], GetReIndex (bweights1to2, 4), limit);
  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[1], GetReIndex (bweights1to2, 5), limit);

  EXPECT_NEAR (1 + lr * error * dsigmoid (sigmoid(3) + sigmoid(2) + sigmoid(2)) * sigmoid(3), GetReIndex (bweights2to3, 0), limit);
  EXPECT_NEAR (1 + lr * error * dsigmoid (sigmoid(3) + sigmoid(2) + sigmoid(2)) * sigmoid(2), GetReIndex (bweights2to3, 1), limit);
  EXPECT_NEAR (1 + lr * error * dsigmoid (sigmoid(3) + sigmoid(2) + sigmoid(2)) * sigmoid(2), GetReIndex (bweights2to3, 2), limit);
}

TEST_F(OapNeuralTests_SimpleBackpropagation, Test_8)
{
  using namespace oap::math;

  oap::Layer* l1 = network->createLayer(2);
  oap::Layer* l2 = network->createLayer(3);
  oap::Layer* l3 = network->createLayer(1);

  oap::HostComplexMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 3);

  oap::host::SetReValuesToMatrix (weights1to2, {2, 1, 1, 1, 1, 1});

  l1->setHostWeights (weights1to2);

  oap::HostComplexMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);

  oap::host::SetReValuesToMatrix(weights2to3, {1, 1, 1});

  l2->setHostWeights (weights2to3);

  oap::HostComplexMatrixPtr inputs = oap::host::NewReMatrix (1, 2);

  std::vector<floatt> inputsVec = {1, 1};

  oap::host::SetReValuesToMatrix (inputs, inputsVec);

  oap::HostComplexMatrixPtr expectedOutputs = oap::host::NewReMatrix (1, 1);
  floatt error = 0.001;
  *GetRePtrIndex (expectedOutputs, 0) = sigmoid (sigmoid (3) + sigmoid (2) + sigmoid (2)) + error;

  floatt lr = 0.01;
  network->setLearningRate (lr);

  network->setInputs (inputs, ArgType::HOST);
  network->setExpected (expectedOutputs, ArgType::HOST);

  network->forwardPropagation ();

  network->accumulateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR, CalculationType::HOST);
  logInfo ("BP %f", network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));

  EXPECT_FLOAT_EQ (sqrt (error * error * 0.5), network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR));
  network->backPropagation ();
  network->updateWeights ();

  oap::HostComplexMatrixPtr bweights1to2 = oap::host::NewReMatrix (2, 3);
  l1->getHostWeights (bweights1to2);

  oap::HostComplexMatrixPtr bweights2to3 = oap::host::NewReMatrix (3, 1);
  l2->getHostWeights (bweights2to3);

  floatt limit = 0.00001;

  EXPECT_NEAR (2 + lr * (error * 1) * dsigmoid (sigmoid(3) + sigmoid(2)) * inputsVec[0], GetReIndex (bweights1to2, 0), limit);
  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[1], GetReIndex (bweights1to2, 1), limit);

  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[0], GetReIndex (bweights1to2, 2), limit);
  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[1], GetReIndex (bweights1to2, 3), limit);

  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[0], GetReIndex (bweights1to2, 4), limit);
  EXPECT_NEAR (1 + lr * (error * 1) * dsigmoid (sigmoid(2) + sigmoid(2)) * inputsVec[1], GetReIndex (bweights1to2, 5), limit);

  EXPECT_NEAR (1 + lr * error * dsigmoid (sigmoid(3) + sigmoid(2) + sigmoid(2)) * sigmoid(3), GetReIndex (bweights2to3, 0), limit);
  EXPECT_NEAR (1 + lr * error * dsigmoid (sigmoid(3) + sigmoid(2) + sigmoid(2)) * sigmoid(2), GetReIndex (bweights2to3, 1), limit);
  EXPECT_NEAR (1 + lr * error * dsigmoid (sigmoid(3) + sigmoid(2) + sigmoid(2)) * sigmoid(2), GetReIndex (bweights2to3, 2), limit);
}
