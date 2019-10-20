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

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "oapMatricesContext.h"

#include "oapHostMatrixUtils.h"

class OapMatricesContextTests : public testing::Test {
 public:
  OapMatricesContextTests() {}

  virtual ~OapMatricesContextTests() {}

  virtual void SetUp()
  {}

  virtual void TearDown()
  {}
};

TEST_F(OapMatricesContextTests, ReturnUnusedMatrix_ApiTest_1)
{
  oap::generic::MatricesContext context;

  const char* HOST = "HOST";

  context.registerMemType (HOST, oap::host::NewHostMatrixFromMatrixInfo, oap::host::DeleteMatrix);
  context.registerMemType ("HOST_1", oap::host::NewHostMatrixFromMatrixInfo, oap::host::DeleteMatrix);

  math::MatrixInfo minfo (true, false, 10, 10);

  math::Matrix* m1 = context.useMatrix (minfo, HOST);
  context.unuseAllMatrices ();
  math::Matrix* m2 = context.useMatrix (minfo, HOST);

  EXPECT_EQ(m1, m2);
}

TEST_F(OapMatricesContextTests, ReturnUnusedMatrix_ApiTest_2)
{
  oap::generic::MatricesContext context;
  context.registerMemType ("HOST", oap::host::NewHostMatrixFromMatrixInfo, oap::host::DeleteMatrix);
  context.registerMemType ("HOST_1", oap::host::NewHostMatrixFromMatrixInfo, oap::host::DeleteMatrix);

  math::MatrixInfo minfo (true, false, 10, 10);

  math::Matrix* m1 = context.useMatrix (minfo, "HOST");
  context.unuseAllMatrices ();
  math::Matrix* m2 = context.useMatrix (math::MatrixInfo (true, false, 10, 10), "HOST");

  EXPECT_EQ(m1, m2);
}

TEST_F(OapMatricesContextTests, Clear_ApiTest)
{
  using namespace testing;

  oap::generic::MatricesContext context;

  const char* HOST = "HOST";

  context.registerMemType (HOST, oap::host::NewHostMatrixFromMatrixInfo, oap::host::DeleteMatrix);
  context.registerMemType ("HOST_1", oap::host::NewHostMatrixFromMatrixInfo, oap::host::DeleteMatrix);

  math::MatrixInfo minfo (true, false, 10, 10);

  math::Matrix* m1 = context.useMatrix (minfo, HOST);
  context.unuseAllMatrices ();
  math::Matrix* m2 = context.useMatrix (math::MatrixInfo (true, false, 10, 10), HOST);

  EXPECT_EQ(m1, m2);

  context.clear ();

  class Allocator
  {
    public:
      virtual math::Matrix* newMatrix (const math::MatrixInfo& minfo)
      {
        return oap::host::NewHostMatrixFromMatrixInfo (minfo);
      }
  };

  class AllocatorMock : public Allocator
  {
    public:
      MOCK_METHOD1 (newMatrix, math::Matrix*(const math::MatrixInfo&));
  };
  
  
  AllocatorMock allocatorMock;
  EXPECT_CALL (allocatorMock, newMatrix(_)).WillOnce (Invoke (oap::host::NewHostMatrixFromMatrixInfo));

  context.registerMemType (HOST, std::bind (&AllocatorMock::newMatrix, &allocatorMock, std::placeholders::_1), oap::host::DeleteMatrix);
  math::Matrix* m3 = context.useMatrix (minfo, HOST);
}

TEST_F(OapMatricesContextTests, Reuse_ApiTest)
{
  using namespace testing;

  math::MatrixInfo minfo (true, false, 10, 10);

  oap::generic::MatricesContext context;

  const char* HOST = "HOST";

  class Allocator
  {
    public:
      virtual math::Matrix* newMatrix (const math::MatrixInfo& minfo)
      {
        return oap::host::NewHostMatrixFromMatrixInfo (minfo);
      }
  };

  class AllocatorMock : public Allocator
  {
    public:
      MOCK_METHOD1 (newMatrix, math::Matrix*(const math::MatrixInfo&));
  };
  
  
  AllocatorMock allocatorMock;
  EXPECT_CALL (allocatorMock, newMatrix(_)).WillOnce (Invoke (oap::host::NewHostMatrixFromMatrixInfo));

  context.registerMemType (HOST, std::bind (&AllocatorMock::newMatrix, &allocatorMock, std::placeholders::_1), oap::host::DeleteMatrix);

  math::Matrix* m1 = context.useMatrix (minfo, HOST);
  context.unuseAllMatrices ();

  math::Matrix* m2 = context.useMatrix (minfo, HOST);

  EXPECT_EQ (m1, m2);
}

TEST_F(OapMatricesContextTests, GetterUse_ApiTest)
{
  using namespace testing;

  math::MatrixInfo minfo (true, false, 10, 10);

  oap::generic::MatricesContext context;

  const char* HOST = "HOST";

  class Allocator
  {
    public:
      virtual math::Matrix* newMatrix (const math::MatrixInfo& minfo)
      {
        return oap::host::NewHostMatrixFromMatrixInfo (minfo);
      }
  };

  class AllocatorMock : public Allocator
  {
    public:
      MOCK_METHOD1 (newMatrix, math::Matrix*(const math::MatrixInfo&));
  };
  
  
  AllocatorMock allocatorMock;
  EXPECT_CALL (allocatorMock, newMatrix(_)).WillOnce (Invoke (oap::host::NewHostMatrixFromMatrixInfo));

  context.registerMemType (HOST, std::bind (&AllocatorMock::newMatrix, &allocatorMock, std::placeholders::_1), oap::host::DeleteMatrix);

  math::Matrix* m1 = reinterpret_cast<math::Matrix*>(0x1);
  math::Matrix* m2 = reinterpret_cast<math::Matrix*>(0x2);
  {
    oap::generic::MatricesContext::Getter getter = context.getter();
    m1 = getter.useMatrix (minfo, HOST);
  }
  {
    oap::generic::MatricesContext::Getter getter = context.getter();
    m2 = getter.useMatrix (minfo, HOST);
  }

  EXPECT_EQ (m1, m2);
}

TEST_F(OapMatricesContextTests, GetterInMethod_ApiTest)
{
  using namespace testing;

  math::MatrixInfo minfo (true, false, 10, 10);

  oap::generic::MatricesContext context;

  const char* HOST = "HOST";

  class Allocator
  {
    public:
      virtual math::Matrix* newMatrix (const math::MatrixInfo& minfo)
      {
        return oap::host::NewHostMatrixFromMatrixInfo (minfo);
      }
  };

  class AllocatorMock : public Allocator
  {
    public:
      MOCK_METHOD1 (newMatrix, math::Matrix*(const math::MatrixInfo&));
  };
  
  
  AllocatorMock allocatorMock;
  EXPECT_CALL (allocatorMock, newMatrix (_)).Times (2).WillRepeatedly(Invoke (oap::host::NewHostMatrixFromMatrixInfo));

  context.registerMemType (HOST, std::bind (&AllocatorMock::newMatrix, &allocatorMock, std::placeholders::_1), oap::host::DeleteMatrix);

  math::Matrix* m1 = reinterpret_cast<math::Matrix*>(0x1);
  math::Matrix* m2 = reinterpret_cast<math::Matrix*>(0x2);
  {
    oap::generic::MatricesContext::Getter getter = context.getter();
    m1 = getter.useMatrix (minfo, HOST);
    {
      oap::generic::MatricesContext::Getter getter = context.getter();
      m2 = getter.useMatrix (minfo, HOST);
    }
    m2 = getter.useMatrix (minfo, HOST);
  }

  EXPECT_NE (m1, m2);
}

TEST_F(OapMatricesContextTests, GeneralTest_1)
{
  using namespace testing;

  math::MatrixInfo minfo (true, false, 10, 10);

  oap::generic::MatricesContext context;

  context.registerMemType ("CUDA", oap::host::NewHostMatrixFromMatrixInfo, oap::host::DeleteMatrix);

  auto func = [&context, &minfo](math::Matrix** m1)
  {
    auto getter = context.getter();
    math::Matrix* matrix = getter.useMatrix (minfo, "CUDA");
    *m1 = matrix;
  };

  math::Matrix* m01;
  math::Matrix* m11;

  func (&m01);
  func (&m11);
  
  EXPECT_EQ (m01, m11);
}

TEST_F(OapMatricesContextTests, GeneralTest_2)
{
  using namespace testing;

  math::MatrixInfo minfo (true, false, 10, 10);

  oap::generic::MatricesContext context;

  context.registerMemType ("CUDA", oap::host::NewHostMatrixFromMatrixInfo, oap::host::DeleteMatrix);

  auto func = [&context, &minfo](math::Matrix** m1, math::Matrix** m2)
  {
    auto getter = context.getter();
    math::Matrix* matrix = getter.useMatrix (minfo, "CUDA");
    math::Matrix* matrix1 = getter.useMatrix (minfo, "CUDA");
    *m1 = matrix;
    *m2 = matrix1;
  };

  math::Matrix* m1[2];
  math::Matrix* m2[2];

  func (&m1[0], &m2[0]);
  EXPECT_TRUE (context.isUsed (m1[0]) == oap::generic::MatricesContext::FALSE);
  EXPECT_TRUE (context.isUsed (m2[0]) == oap::generic::MatricesContext::FALSE);

  func (&m1[1], &m2[1]);
  EXPECT_TRUE (context.isUsed (m1[1]) == oap::generic::MatricesContext::FALSE);
  EXPECT_TRUE (context.isUsed (m2[1]) == oap::generic::MatricesContext::FALSE);
  
  EXPECT_TRUE ((m1[0] == m1[1] && m2[0] == m2[1]) || (m1[0] == m2[1] && m1[1] == m2[0])) << "m1[0] = " << m1[0] << " m1[1] = " << m1[1] << " m2[0] " << m2[0] << "m2[1] " << m2[1];
}
