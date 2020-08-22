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

#include <gtest/gtest.h>
#include <iterator>

#include "oapNetworkStructure.h"
#include "oapGenericNeuralUtils.h"
#include "oapLayerStructure.h"
#include "oapHostMatrixPtr.h"
#include "oapHostMatrixUPtr.h"
#include "oapLayer.h"

class OapGenericNeuralUtilsTests : public testing::Test
{
public:

  virtual void SetUp()
  {}

  virtual void TearDown()
  {}

  class MockLayerApi
  {
    public:
      void allocate (Layer<MockLayerApi>* layer)
      {
        layer->setFPMatrices (new FPMatrices());
        layer->setBPMatrices (new BPMatrices());
      }

      void deallocate (Layer<MockLayerApi>* layer)
      {
        delete layer->getFPMatrices ();
        delete layer->getBPMatrices ();
      }
  };

  using MockLayer = Layer<MockLayerApi>;
};

TEST_F(OapGenericNeuralUtilsTests, CopyIntoTest_1)
{
  std::vector<std::vector<floatt>> vec = {{0}, {1}, {2}, {3}, {4}, {5}};

  uintt nc = 10;
  uintt bc = 1;


  MockLayer* layer = new MockLayer (nc, bc, vec.size(), Activation::NONE);

  size_t counts = oap::nutils::getElementsCount(vec);
  EXPECT_EQ(6, counts);

  oap::HostMatrixUPtr hmatrix = oap::host::NewReMatrix (1, counts);
  layer->getFPMatrices()->m_inputs = hmatrix.get();

  oap::nutils::copyToInputs (layer, vec, oap::nutils::copyHostBufferToHostReMatrix);

  std::vector<floatt> output (hmatrix->re.ptr, hmatrix->re.ptr + vec.size());
  std::vector<floatt> expectedOutput = {0, 1, 2, 3, 4, 5};

  EXPECT_EQ (expectedOutput, output);

  delete layer;
}

TEST_F(OapGenericNeuralUtilsTests, CopyIntoTest_2)
{
  std::vector<std::vector<floatt>> vec = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}};

  uintt nc = 10;
  uintt bc = 1;

  MockLayer* layer = new MockLayer (nc, bc, vec.size(), Activation::NONE);

  size_t counts = oap::nutils::getElementsCount(vec);
  EXPECT_EQ(12, counts);

  oap::HostMatrixUPtr hmatrix = oap::host::NewReMatrix (1, counts);
  layer->getFPMatrices()->m_inputs = hmatrix.get();

  oap::nutils::copyToInputs (layer, vec, oap::nutils::copyHostBufferToHostReMatrix);

  std::vector<floatt> output (hmatrix->re.ptr, hmatrix->re.ptr + ((counts / vec.size()) * vec.size()));
  std::vector<floatt> expectedOutput = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};

  EXPECT_EQ (expectedOutput, output);

  delete layer;
}

TEST_F(OapGenericNeuralUtilsTests, CopyIntoTest_3)
{
  std::vector<std::vector<floatt>> vec = {{1, 1, 1, 1}, {1, 1, 1, 1}, {2, 2, 3, 3}, {3, 3, 6, 7}, {4, 4, 8, 2}, {5, 5, 0, 0}};

  uintt nc = 10;
  uintt bc = 1;

  MockLayer* layer = new MockLayer (nc, bc, vec.size(), Activation::NONE);

  size_t counts = oap::nutils::getElementsCount(vec);
  EXPECT_EQ(24, counts);

  oap::HostMatrixUPtr hmatrix = oap::host::NewReMatrix (1, counts);
  layer->getFPMatrices()->m_inputs = hmatrix.get();

  oap::nutils::copyToInputs (layer, vec, oap::nutils::copyHostBufferToHostReMatrix);

  std::vector<floatt> output (hmatrix->re.ptr, hmatrix->re.ptr + ((counts / vec.size()) * vec.size()));
  std::vector<floatt> expectedOutput = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 6, 7, 4, 4, 8, 2, 5, 5, 0, 0};

  EXPECT_EQ (expectedOutput, output);

  delete layer;
}

TEST_F(OapGenericNeuralUtilsTests, CreateExpectedVector_1)
{
  class NetworkStructureImpl : public NetworkS<oap::HostMatrixPtr>
  {
    public:
      virtual void setExpectedProtected (typename ExpectedOutputs::mapped_type& holder, math::Matrix* expected, ArgType argType) override
      {
        holder = expected;
      }

      virtual math::Matrix* convertExpectedProtected (oap::HostMatrixPtr t) const override
      {
        return t.get();
      }
  };

  std::vector<std::vector<floatt>> vec = {{0}, {1}, {2}, {3}, {4}, {5}};

  uintt nc = 10;
  uintt bc = 1;

  NetworkStructureImpl network;
  size_t counts = oap::nutils::getElementsCount(vec);
  EXPECT_EQ(6, counts);

  oap::nutils::createExpectedOutput (&network, 0, vec, ArgType::HOST, oap::host::NewHostReMatrix, oap::nutils::copyHostBufferToHostReMatrix);

  std::vector<floatt> output (network.getExpected(0)->re.ptr, network.getExpected(0)->re.ptr + vec.size());
  std::vector<floatt> expectedOutput = {0, 1, 2, 3, 4, 5};

  EXPECT_EQ (expectedOutput, output);
}

TEST_F(OapGenericNeuralUtilsTests, CreateExpectedVector_2)
{
  class NetworkStructureImpl : public NetworkS<oap::HostMatrixPtr>
  {
    public:
      virtual void setExpectedProtected (typename ExpectedOutputs::mapped_type& holder, math::Matrix* expected, ArgType argType) override
      {
        holder = expected;
      }

      virtual math::Matrix* convertExpectedProtected (oap::HostMatrixPtr t) const override
      {
        return t.get();
      }
  };

  std::vector<std::vector<floatt>> vec = {{0, 1}, {1, 2}, {2, 3}, {3, 5}, {4, 8}, {5, 9}};

  uintt nc = 10;
  uintt bc = 1;

  NetworkStructureImpl network;
  size_t counts = oap::nutils::getElementsCount(vec);
  EXPECT_EQ(12, counts);

  oap::nutils::createExpectedOutput (&network, 0, vec, ArgType::HOST, oap::host::NewHostReMatrix, oap::nutils::copyHostBufferToHostReMatrix);

  std::vector<floatt> output (network.getExpected(0)->re.ptr, network.getExpected(0)->re.ptr + (vec.size() * 2));
  std::vector<floatt> expectedOutput = {0, 1, 1, 2, 2, 3, 3, 5, 4, 8, 5, 9};

  EXPECT_EQ (expectedOutput, output);
}
