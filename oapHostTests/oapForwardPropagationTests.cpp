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

  oap::alloc::host::AllocNeuronsApi allocN;
  oap::alloc::host::AllocWeightsApi allocW;
  std::vector<LayerS*> layers;
  std::vector<std::shared_ptr<LayerS>> layersPtrs;

  LayerS* l1 = oap::generic::createLayer<LayerS, oap::alloc::host::AllocNeuronsApi> (2, true, Activation::TANH);
  LayerS* l2 = oap::generic::createLayer<LayerS, oap::alloc::host::AllocNeuronsApi> (3, true, Activation::TANH);
  LayerS* l3 = oap::generic::createLayer<LayerS, oap::alloc::host::AllocNeuronsApi> (1, true, Activation::TANH);

  oap::generic::connectLayers<LayerS, oap::alloc::host::AllocWeightsApi>(l1, l2);
  oap::generic::connectLayers<LayerS, oap::alloc::host::AllocWeightsApi>(l2, l3);


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
}
