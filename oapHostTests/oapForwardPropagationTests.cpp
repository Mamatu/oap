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
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "oapGenericNeuralApi.h"
#include "oapGenericAllocApi.h"

#include "HostProcedures.h"
#include "oapHostMatrixPtr.h"

#include "oapFunctions.h"

class OapForwardPropagationTests : public testing::Test {
public:

    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(OapForwardPropagationTests, Test_1)
{
  HostProcedures hostProcedures;
  using namespace oap::math;

  std::vector<LayerS*> layers;
  std::vector<std::shared_ptr<LayerS>> layersPtrs;

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


  LayerS* l1 = oap::generic::createLayer<LayerS, oap::alloc::host::AllocNeuronsApi> (3, false, Activation::SIGMOID);
  LayerS* l2 = oap::generic::createLayer<LayerS, oap::alloc::host::AllocNeuronsApi> (3, false, Activation::SIGMOID);
  LayerS* l3 = oap::generic::createLayer<LayerS, oap::alloc::host::AllocNeuronsApi> (1, false, Activation::SIGMOID);

  oap::generic::connectLayers<LayerS, oap::alloc::host::AllocWeightsApi>(l1, l2);
  oap::generic::connectLayers<LayerS, oap::alloc::host::AllocWeightsApi>(l2, l3);

  oap::generic::setHostWeights (*l1, weights1to2, oap::host::CopyHostMatrixToHostMatrix);
  oap::generic::setHostWeights (*l2, weights2to3, oap::host::CopyHostMatrixToHostMatrix);

  auto ldeleter = [](LayerS* layer)
  {
    oap::generic::deallocate<LayerS, oap::alloc::host::DeallocLayerApi> (*layer);
    delete layer;
  };

  layersPtrs.push_back (std::shared_ptr<LayerS>(l1, ldeleter));
  layersPtrs.push_back (std::shared_ptr<LayerS>(l2, ldeleter));
  layersPtrs.push_back (std::shared_ptr<LayerS>(l3, ldeleter));

  layers.push_back (l1);
  layers.push_back (l2);
  layers.push_back (l3);

  oap::generic::forwardPropagation (layers, hostProcedures, oap::host::SetReValue);

  auto getLayerOutput = [](LayerS* layer)
  {
    auto minfo = oap::generic::getOutputsInfo (*layer, oap::host::GetMatrixInfo);
    oap::HostMatrixPtr outputsL = oap::host::NewReMatrix (minfo.m_matrixDim.columns, minfo.m_matrixDim.rows);
    oap::generic::getOutputs (outputsL, *layer, oap::host::CopyHostMatrixToHostMatrix);
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
