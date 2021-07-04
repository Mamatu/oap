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

#include "oapDeviceComplexMatrixUPtr.hpp"

#include "oapHostLayer.hpp"
#include "KernelExecutor.hpp"
#include "oapGenericNeuralUtils.hpp"
#include "oapDeviceNeuralUtils.hpp"

class OapNeuralUtilsTests : public testing::Test
{
public:

  virtual void SetUp()
  {
    oap::cuda::Context::Instance().create();
  }

  virtual void TearDown()
  {
    oap::cuda::Context::Instance().destroy();
  }
};

namespace
{
  class MockLayer : public oap::HostLayer
  {
    public:
      MockLayer (uintt neuronsCount, uintt biasesCount, uintt samplesCount, Activation activation) : oap::HostLayer (neuronsCount, biasesCount, samplesCount, activation)
      {
        m_fpMatrices.push_back (new FPMatrices ());
      }
  
      virtual ~MockLayer()
      {
        delete m_fpMatrices[0];
      }
  };
}

TEST_F(OapNeuralUtilsTests, CopyIntoTest_1)
{
  std::vector<std::vector<floatt>> vec = {{0}, {1}, {2}, {3}, {4}, {5}};

  uintt nc = 10;
  uintt bc = 1;

  MockLayer* layer = new MockLayer (nc, bc, vec.size(), Activation::NONE);

  size_t counts = oap::nutils::getElementsCount(vec);
  EXPECT_EQ(6, counts);

  oap::DeviceComplexMatrixUPtr dmatrix = oap::cuda::NewDeviceReMatrix (1, counts);
  layer->getFPMatrices()->m_inputs = dmatrix.get();

  oap::nutils::copyToInputs_oneMatrix (layer, vec, ArgType::HOST);

  oap::HostComplexMatrixUPtr hmatrix = oap::cuda::NewHostMatrixCopyOfDeviceMatrix (dmatrix.get ());

  std::vector<floatt> output (hmatrix->re.mem.ptr, hmatrix->re.mem.ptr + vec.size());
  std::vector<floatt> expectedOutput = {0, 1, 2, 3, 4, 5};

  EXPECT_EQ (expectedOutput, output);

  delete layer;
}

TEST_F(OapNeuralUtilsTests, CopyIntoTest_2)
{
  std::vector<std::vector<floatt>> vec = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}};

  uintt nc = 10;
  uintt bc = 1;

  MockLayer* layer = new MockLayer (nc, bc, vec.size(), Activation::NONE);

  size_t counts = oap::nutils::getElementsCount(vec);
  EXPECT_EQ(12, counts);

  oap::DeviceComplexMatrixUPtr dmatrix = oap::cuda::NewDeviceReMatrix (1, counts);
  layer->getFPMatrices()->m_inputs = dmatrix.get();

  oap::nutils::copyToInputs_oneMatrix (layer, vec, ArgType::HOST);

  oap::HostComplexMatrixUPtr hmatrix = oap::cuda::NewHostMatrixCopyOfDeviceMatrix (dmatrix.get ());

  std::vector<floatt> output (hmatrix->re.mem.ptr, hmatrix->re.mem.ptr + ((counts / vec.size()) * vec.size()));
  std::vector<floatt> expectedOutput = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};

  EXPECT_EQ (expectedOutput, output);

  delete layer;
}

TEST_F(OapNeuralUtilsTests, CopyIntoTest_3)
{
  std::vector<std::vector<floatt>> vec = {{1, 1, 1, 1}, {1, 1, 1, 1}, {2, 2, 3, 3}, {3, 3, 6, 7}, {4, 4, 8, 2}, {5, 5, 0, 0}};

  uintt nc = 10;
  uintt bc = 1;

  MockLayer* layer = new MockLayer (nc, bc, vec.size(), Activation::NONE);

  size_t counts = oap::nutils::getElementsCount(vec);
  EXPECT_EQ(24, counts);

  oap::DeviceComplexMatrixUPtr dmatrix = oap::cuda::NewDeviceReMatrix (1, counts);
  layer->getFPMatrices()->m_inputs = dmatrix.get();

  oap::nutils::copyToInputs_oneMatrix (layer, vec, ArgType::HOST);

  oap::HostComplexMatrixUPtr hmatrix = oap::cuda::NewHostMatrixCopyOfDeviceMatrix (dmatrix.get ());

  std::vector<floatt> output (hmatrix->re.mem.ptr, hmatrix->re.mem.ptr + ((counts / vec.size()) * vec.size()));
  std::vector<floatt> expectedOutput = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 6, 7, 4, 4, 8, 2, 5, 5, 0, 0};

  EXPECT_EQ (expectedOutput, output);

  delete layer;
}
