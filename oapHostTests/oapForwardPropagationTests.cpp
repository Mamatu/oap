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
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"


#include "oapGenericNeuralApi.h"
#include "oapGenericAllocApi.h"

#include "HostProcedures.h"
#include "oapHostComplexMatrixPtr.h"
#include "oapLayer.h"

#include "oapFunctions.h"
#if 0
class OapForwardPropagationTests : public testing::Test {
public:

    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

namespace
{
  class MockLayer
  {
    public:
      void allocate ()
      {
      }

      void deallocate ()
      {
        delete layer->getFPMatrices();
        delete layer->getBPMatrices();
      }
  };
}

std::vector<oap::HostMatrixPtr> runForwardPropagation (const std::vector<uintt> ns, const std::vector<oap::HostMatrixPtr> weights)
{
  oap::HostProcedures hostProcedures;

  auto ldeleter = [](MockLayer* layer)
  {
    oap::generic::deallocate<MockLayer, oap::alloc::host::DeallocLayerApi> (*layer);
    delete layer;
  };

  std::vector<MockLayer*> layers;
  std::vector<std::shared_ptr<MockLayer>> layersPtrs;

  for (size_t idx = 0; idx < ns.size(); ++idx)
  {
    MockLayer* l1 = oap::generic::createLayer<MockLayer> (ns[idx], false, 1, Activation::SIGMOID);
    oap::generic::createFPMatrices<oap::alloc::host::AllocNeuronsApi>(*l1);
    layersPtrs.push_back (std::shared_ptr<MockLayer>(l1, ldeleter));
    layers.push_back (l1);
  }

  for (size_t idx = 0; idx < ns.size() - 1; ++idx)
  {
    oap::generic::connectLayers<MockLayer, oap::alloc::host::AllocWeightsApi>(layers[idx], layers[idx + 1]);
    oap::generic::setHostWeights (*layers[idx], weights[idx].get(), oap::host::CopyHostMatrixToHostMatrix, oap::host::GetMatrixInfo, oap::host::GetMatrixInfo);
  }

  for (uintt idx = 0; idx < ns[0]; ++idx)
  {
    *GetRePtrIndex (layers.front()->getFPMatrices()->m_inputs, idx) = 1;
  }

  oap::generic::initNetworkBiases (layers, oap::host::SetReValue);
  oap::generic::forwardPropagation_oneSample<MockLayer> (layers, hostProcedures);

  auto getLayerOutput = [](MockLayer* layer)
  {
    auto minfo = oap::generic::getOutputsInfo (*layer, oap::host::GetMatrixInfo);
    oap::HostMatrixPtr outputsL = oap::host::NewReMatrix (minfo.columns (), minfo.rows ());
    oap::generic::getOutputs (outputsL, *layer, oap::host::CopyHostMatrixToHostMatrix);
    return outputsL;
  };

  std::vector<oap::HostMatrixPtr> outputs;

  for (size_t idx = 1; idx < layers.size(); ++idx)
  {
    auto outputsL = getLayerOutput(layers[idx]);
    outputs.push_back(outputsL);
  }

  return outputs;
}

TEST_F(OapForwardPropagationTests, Test_1)
{
  using namespace oap::math;

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (3, 3);
  *GetRePtrIndex (weights1to2, 0) = 4;
  *GetRePtrIndex (weights1to2, 3) = 3;
  *GetRePtrIndex (weights1to2, 6) = 2;

  *GetRePtrIndex (weights1to2, 1) = 1;
  *GetRePtrIndex (weights1to2, 4) = 1;
  *GetRePtrIndex (weights1to2, 7) = 1;

  *GetRePtrIndex (weights1to2, 2) = 1;
  *GetRePtrIndex (weights1to2, 5) = 1;
  *GetRePtrIndex (weights1to2, 8) = 1;

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);
  *GetRePtrIndex (weights2to3, 0) = 1;
  *GetRePtrIndex (weights2to3, 1) = 1;
  *GetRePtrIndex (weights2to3, 2) = 1;

  auto outputs = runForwardPropagation ({3, 3, 1}, {weights1to2, weights2to3});

  auto outputsL2 = outputs[0];
  auto outputsL3 = outputs[1];

  EXPECT_DOUBLE_EQ (sigmoid(6), GetReIndex (outputsL2, 0));
  EXPECT_DOUBLE_EQ (sigmoid(5), GetReIndex (outputsL2, 1));
  EXPECT_DOUBLE_EQ (sigmoid(4), GetReIndex (outputsL2, 2));
  EXPECT_DOUBLE_EQ (sigmoid (sigmoid(6) + sigmoid(5) + sigmoid(4)), GetReIndex (outputsL3, 0));
}

TEST_F(OapForwardPropagationTests, Test_2)
{
  using namespace oap::math;

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (3, 3);
  *GetRePtrIndex (weights1to2, 0) = 1;
  *GetRePtrIndex (weights1to2, 3) = 1;
  *GetRePtrIndex (weights1to2, 6) = 1;

  *GetRePtrIndex (weights1to2, 1) = 1;
  *GetRePtrIndex (weights1to2, 4) = 1;
  *GetRePtrIndex (weights1to2, 7) = 1;

  *GetRePtrIndex (weights1to2, 2) = 1;
  *GetRePtrIndex (weights1to2, 5) = 1;
  *GetRePtrIndex (weights1to2, 8) = 1;

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (3, 1);
  *GetRePtrIndex (weights2to3, 0) = 1;
  *GetRePtrIndex (weights2to3, 1) = 1;
  *GetRePtrIndex (weights2to3, 2) = 1;

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 3);
  *GetRePtrIndex (inputs, 0) = 1;
  *GetRePtrIndex (inputs, 1) = 1;
  *GetRePtrIndex (inputs, 2) = 1;

  auto outputs = runForwardPropagation ({3, 3, 1}, {weights1to2, weights2to3});

  auto outputsL3 = outputs[outputs.size() - 1];

  EXPECT_DOUBLE_EQ (sigmoid (sigmoid(3) + sigmoid(3) + sigmoid(3)), GetReIndex (outputsL3, 0));
}

TEST_F(OapForwardPropagationTests, Test_3)
{
  using namespace oap::math;

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 1);
  *GetRePtrIndex (weights1to2, 0) = 1;
  *GetRePtrIndex (weights1to2, 1) = 1;

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 2);
  *GetRePtrIndex (inputs, 0) = 1;
  *GetRePtrIndex (inputs, 1) = 1;

  auto outputs = runForwardPropagation ({2, 1}, {weights1to2});

  auto outputsL = outputs[outputs.size() - 1];

  EXPECT_DOUBLE_EQ (sigmoid (2), GetReIndex (outputsL, 0));
}

#endif
