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

#include "oapHostMatrixUtils.h"
#include "oapCudaMatrixUtils.h"

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
        layer->addFPMatrices (new FPMatrices());
        layer->addBPMatrices (new BPMatrices());
      }

      void deallocate (Layer<MockLayerApi>* layer)
      {
        delete layer->getFPMatrices ();
        delete layer->getBPMatrices ();
      }
  };

  using MockLayer = Layer<MockLayerApi>;
};
#if 0
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

  oap::nutils::copyToInputs (layer, vec, oap::host::CopyHostBufferToHost);

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

  oap::nutils::copyToInputs (layer, vec, oap::host::CopyHostBufferToHost);

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

  oap::nutils::copyToInputs (layer, vec, oap::host::CopyHostBufferToHost);

  std::vector<floatt> output (hmatrix->re.ptr, hmatrix->re.ptr + ((counts / vec.size()) * vec.size()));
  std::vector<floatt> expectedOutput = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 6, 7, 4, 4, 8, 2, 5, 5, 0, 0};

  EXPECT_EQ (expectedOutput, output);

  delete layer;
}

TEST_F(OapGenericNeuralUtilsTests, CreateExpectedVector_1)
{
  class NetworkStructureImpl
  {
    std::vector<math::Matrix*> m_expected;
    public:
      ~NetworkStructureImpl ()
      {
        for (math::Matrix* matrix : m_expected)
        {
          oap::host::DeleteMatrix (matrix);
        }
      }

      void setExpected (const std::vector<math::Matrix*>& expected, ArgType argType, LHandler handler = 0)
      {
        m_expected = expected;
      }

      std::vector<math::Matrix*> getExpected (LHandler handler = 0) const
      {
        return m_expected;
      }
  };

  std::vector<std::vector<floatt>> vec = {{0}, {1}, {2}, {3}, {4}, {5}};

  uintt nc = 10;
  uintt bc = 1;

  NetworkStructureImpl network;
  size_t counts = oap::nutils::getElementsCount(vec);
  EXPECT_EQ(6, counts);

  oap::nutils::createExpectedOutput (&network, 0, vec, ArgType::HOST, oap::host::NewHostReMatrix, oap::host::CopyHostBufferToReMatrix);

  std::vector<math::Matrix*> output = network.getExpected();

  std::vector<floatt> actualOutput = oap::nutils::convertToFloattBuffer (output, oap::host::GetMatrixInfo, oap::host::CopyReMatrixToHostBuffer);
  std::vector<floatt> expectedOutput = {0, 1, 2, 3, 4, 5};

  EXPECT_EQ (expectedOutput, actualOutput);
}

TEST_F(OapGenericNeuralUtilsTests, CreateExpectedVector_2)
{
  class NetworkStructureImpl
  {
    std::vector<math::Matrix*> m_expected;
    public:
      ~NetworkStructureImpl ()
      {
        for (math::Matrix* matrix : m_expected)
        {
          oap::host::DeleteMatrix (matrix);
        }
      }

      void setExpected (const std::vector<math::Matrix*>& expected, ArgType argType, LHandler handler = 0)
      {
        m_expected = expected;
      }

      std::vector<math::Matrix*> getExpected (LHandler handler = 0) const
      {
        return m_expected;
      }
  };

  std::vector<std::vector<floatt>> vec = {{0, 1}, {1, 2}, {2, 3}, {3, 5}, {4, 8}, {5, 9}};

  uintt nc = 10;
  uintt bc = 1;

  NetworkStructureImpl network;
  size_t counts = oap::nutils::getElementsCount(vec);
  EXPECT_EQ(12, counts);

  oap::nutils::createExpectedOutput (&network, 0, vec, ArgType::HOST, oap::host::NewHostReMatrix, oap::host::CopyHostBufferToReMatrix);

  std::vector<math::Matrix*> output = network.getExpected();

  std::vector<floatt> actualOutput = oap::nutils::convertToFloattBuffer (output, oap::host::GetMatrixInfo, oap::host::CopyReMatrixToHostBuffer);
  std::vector<floatt> expectedOutput = {0, 1, 1, 2, 2, 3, 3, 5, 4, 8, 5, 9};

  EXPECT_EQ (expectedOutput, actualOutput);
}

TEST_F(OapGenericNeuralUtilsTests, ConvertToMatrices_1)
{
  class NetworkStructureImpl
  {
    std::vector<math::Matrix*> m_expected;
    public:
      ~NetworkStructureImpl ()
      {
        for (math::Matrix* matrix : m_expected)
        {
          oap::host::DeleteMatrix (matrix);
        }
      }

      void setExpected (const std::vector<math::Matrix*>& expected, ArgType argType, LHandler handler = 0)
      {
        m_expected = expected;
      }

      std::vector<math::Matrix*> getExpected (LHandler handler = 0) const
      {
        return m_expected;
      }
  };

  NetworkStructureImpl network;
  std::vector<std::vector<floatt>> vec = {{0}};
  oap::nutils::createExpectedOutput (&network, 0, vec, ArgType::HOST, oap::host::NewHostReMatrix, oap::host::CopyHostBufferToReMatrix);

  auto expected = network.getExpected();

  EXPECT_EQ(1, expected.size());
  EXPECT_EQ(1, expected[0]->re.dims.width);
  EXPECT_EQ(1, expected[0]->re.dims.height);
  EXPECT_EQ(0.f, expected[0]->re.ptr[0]);
}

TEST_F(OapGenericNeuralUtilsTests, ConvertToMatrices_2)
{
  class NetworkStructureImpl
  {
    std::vector<math::Matrix*> m_expected;
    public:
      ~NetworkStructureImpl ()
      {
        for (math::Matrix* matrix : m_expected)
        {
          oap::host::DeleteMatrix (matrix);
        }
      }

      void setExpected (const std::vector<math::Matrix*>& expected, ArgType argType, LHandler handler = 0)
      {
        m_expected = expected;
      }

      std::vector<math::Matrix*> getExpected (LHandler handler = 0) const
      {
        return m_expected;
      }
  };

  NetworkStructureImpl network;
  std::vector<std::vector<floatt>> vec = {{0}, {1}, {2}};
  oap::nutils::createExpectedOutput (&network, 0, vec, ArgType::HOST, oap::host::NewHostReMatrix, oap::host::CopyHostBufferToReMatrix);
  
  auto expected = network.getExpected();

  EXPECT_EQ(3, expected.size());

  EXPECT_EQ(1, expected[0]->re.dims.width);
  EXPECT_EQ(1, expected[0]->re.dims.height);
  EXPECT_EQ(0.f, expected[0]->re.ptr[0]);

  EXPECT_EQ(1, expected[1]->re.dims.width);
  EXPECT_EQ(1, expected[1]->re.dims.height);
  EXPECT_EQ(1.f, expected[1]->re.ptr[0]);

  EXPECT_EQ(1, expected[2]->re.dims.width);
  EXPECT_EQ(1, expected[2]->re.dims.height);
  EXPECT_EQ(2.f, expected[2]->re.ptr[0]);
}

TEST_F(OapGenericNeuralUtilsTests, ConvertToMatrices_3)
{
  class NetworkStructureImpl
  {
    std::vector<math::Matrix*> m_expected;
    public:
      ~NetworkStructureImpl ()
      {
        for (math::Matrix* matrix : m_expected)
        {
          oap::host::DeleteMatrix (matrix);
        }
      }

      void setExpected (const std::vector<math::Matrix*>& expected, ArgType argType, LHandler handler = 0)
      {
        m_expected = expected;
      }

      std::vector<math::Matrix*> getExpected (LHandler handler = 0) const
      {
        return m_expected;
      }
  };

  NetworkStructureImpl network;
  std::vector<std::vector<floatt>> vec = {{0}, {1, 2}};
  oap::nutils::createExpectedOutput (&network, 0, vec, ArgType::HOST, oap::host::NewHostReMatrix, oap::host::CopyHostBufferToReMatrix);
  
  auto expected = network.getExpected();

  EXPECT_EQ(2, expected.size());

  EXPECT_EQ(1, expected[0]->re.dims.width);
  EXPECT_EQ(1, expected[0]->re.dims.height);
  EXPECT_EQ(0.f, expected[0]->re.ptr[0]);

  EXPECT_EQ(1, expected[1]->re.dims.width);
  EXPECT_EQ(2, expected[1]->re.dims.height);
  EXPECT_EQ(1.f, expected[1]->re.ptr[0]);
  EXPECT_EQ(2.f, expected[1]->re.ptr[1]);
}

TEST_F(OapGenericNeuralUtilsTests, ConvertToFloattBuffer_1)
{
  std::vector<math::Matrix*> matrices = {oap::host::NewReMatrixWithValue (1, 1, 1.f)};
  std::vector<floatt> actualOutput = oap::nutils::convertToFloattBuffer (matrices, oap::host::GetMatrixInfo, oap::host::CopyReMatrixToHostBuffer);

  EXPECT_EQ(1, actualOutput.size());
  EXPECT_EQ(1.f, actualOutput[0]);
  oap::host::DeleteMatrix (matrices[0]);
}

TEST_F(OapGenericNeuralUtilsTests, ConvertToFloattBuffer_2)
{
  std::vector<math::Matrix*> matrices = {oap::host::NewReMatrixWithValue (1, 1, 1.f), oap::host::NewReMatrixWithValue (1, 1, 2.f)};
  std::vector<floatt> actualOutput = oap::nutils::convertToFloattBuffer (matrices, oap::host::GetMatrixInfo, oap::host::CopyReMatrixToHostBuffer);

  EXPECT_EQ(2, actualOutput.size());
  EXPECT_EQ(1.f, actualOutput[0]);
  EXPECT_EQ(2.f, actualOutput[1]);

  for (auto it = matrices.begin(); it != matrices.end(); ++it)
  {
    oap::host::DeleteMatrix (*it);
  }
}

TEST_F(OapGenericNeuralUtilsTests, ConvertToFloattBuffer_3)
{
  std::vector<math::Matrix*> matrices =
  {
    oap::host::NewReMatrixWithValue (1, 1, 0.f),
    oap::host::NewReMatrixWithValue (1, 1, 1.f),
    oap::host::NewReMatrixWithValue (1, 1, 2.f),
    oap::host::NewReMatrixWithValue (1, 1, 3.f),
    oap::host::NewReMatrixWithValue (1, 1, 4.f),
    oap::host::NewReMatrixWithValue (1, 1, 5.f),
  };
  std::vector<floatt> actualOutput = oap::nutils::convertToFloattBuffer (matrices, oap::host::GetMatrixInfo, oap::host::CopyReMatrixToHostBuffer);

  EXPECT_EQ(6, actualOutput.size());
  EXPECT_EQ(0.f, actualOutput[0]);
  EXPECT_EQ(1.f, actualOutput[1]);
  EXPECT_EQ(2.f, actualOutput[2]);
  EXPECT_EQ(3.f, actualOutput[3]);
  EXPECT_EQ(4.f, actualOutput[4]);
  EXPECT_EQ(5.f, actualOutput[5]);

  for (auto it = matrices.begin(); it != matrices.end(); ++it)
  {
    oap::host::DeleteMatrix (*it);
  }
}
#endif
