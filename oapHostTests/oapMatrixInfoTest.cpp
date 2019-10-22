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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "MatchersUtils.h"
#include "MatrixInfo.h"


class OapMatrixInfoTests : public testing::Test {
  public:
    virtual void SetUp() {}

    virtual void TearDown() {}
};

TEST_F(OapMatrixInfoTests, SizeTests)
{
  {
    math::MatrixInfo minfo (false, true, 1, 1024);
    EXPECT_EQ(sizeof(floatt), minfo.getSize().first);
    EXPECT_EQ(math::MatrixInfo::KB, minfo.getSize().second);

    EXPECT_EQ(1, minfo.columns());
    EXPECT_EQ(1024, minfo.rows());
    EXPECT_EQ(minfo.columns(), minfo.m_matrixDim.columns);
    EXPECT_EQ(minfo.rows(), minfo.m_matrixDim.rows);
  }
  {
    math::MatrixInfo minfo (false, true, 1024, 1024);
    EXPECT_EQ(sizeof(floatt), minfo.getSize().first);
    EXPECT_EQ(math::MatrixInfo::MB, minfo.getSize().second);

    EXPECT_EQ(1024, minfo.columns());
    EXPECT_EQ(1024, minfo.rows());
    EXPECT_EQ(minfo.columns(), minfo.m_matrixDim.columns);
    EXPECT_EQ(minfo.rows(), minfo.m_matrixDim.rows);
  }
  {
    math::MatrixInfo minfo (true, false, 1024, 1024);
    EXPECT_EQ(sizeof(floatt), minfo.getSize().first);
    EXPECT_EQ(math::MatrixInfo::MB, minfo.getSize().second);

    EXPECT_EQ(1024, minfo.columns());
    EXPECT_EQ(1024, minfo.rows());
    EXPECT_EQ(minfo.columns(), minfo.m_matrixDim.columns);
    EXPECT_EQ(minfo.rows(), minfo.m_matrixDim.rows);
  }
  {
    math::MatrixInfo minfo (true, true, 1024, 1024);
    EXPECT_EQ(sizeof(floatt) * 2, minfo.getSize().first);
    EXPECT_EQ(math::MatrixInfo::MB, minfo.getSize().second);

    EXPECT_EQ(1024, minfo.columns());
    EXPECT_EQ(1024, minfo.rows());
    EXPECT_EQ(minfo.columns(), minfo.m_matrixDim.columns);
    EXPECT_EQ(minfo.rows(), minfo.m_matrixDim.rows);
  }
}

TEST_F(OapMatrixInfoTests, LargeSizeTests)
{
  {
    math::MatrixInfo minfo (true, false, 1024*10, 1024*1024);
    EXPECT_EQ(sizeof(floatt) * 10, minfo.getSize().first);
    EXPECT_EQ(math::MatrixInfo::GB, minfo.getSize().second);

    EXPECT_EQ(1024*10, minfo.columns());
    EXPECT_EQ(1024*1024, minfo.rows());
    EXPECT_EQ(minfo.columns(), minfo.m_matrixDim.columns);
    EXPECT_EQ(minfo.rows(), minfo.m_matrixDim.rows);
  }
  {
    math::MatrixInfo minfo (true, false, 1024, 1024*1024*2);
    EXPECT_EQ(sizeof(floatt) * 2, minfo.getSize().first);
    EXPECT_EQ(math::MatrixInfo::GB, minfo.getSize().second);

    EXPECT_EQ(1024, minfo.columns());
    EXPECT_EQ(1024*1024*2, minfo.rows());
    EXPECT_EQ(minfo.columns(), minfo.m_matrixDim.columns);
    EXPECT_EQ(minfo.rows(), minfo.m_matrixDim.rows);
  }

}

TEST_F(OapMatrixInfoTests, CompareTest_1)
{
  math::MatrixInfo minfo (true, true, 1, 8);
  math::MatrixInfo minfo1 (true, true, 1, 8);

  EXPECT_FALSE(minfo < minfo1);
  EXPECT_FALSE(minfo > minfo1);
}

TEST_F(OapMatrixInfoTests, CompareTest_2)
{
  math::MatrixInfo minfo (true, true, 1, 2);
  math::MatrixInfo minfo1 (true, true, 1, 8);

  EXPECT_TRUE (minfo < minfo1 || minfo1 < minfo);
}

TEST_F(OapMatrixInfoTests, CompareTest_3)
{
  math::MatrixInfo minfo (true, true, 2, 8);
  math::MatrixInfo minfo1 (true, true, 1, 8);

  EXPECT_TRUE (minfo < minfo1 || minfo1 < minfo);
}

TEST_F(OapMatrixInfoTests, CompareTest_4)
{
  math::MatrixInfo minfo (true, false, 1, 8);
  math::MatrixInfo minfo1 (true, true, 1, 8);

  EXPECT_TRUE (minfo < minfo1 || minfo1 < minfo);
}

TEST_F(OapMatrixInfoTests, CompareTest_5)
{
  math::MatrixInfo minfo (false, true, 1, 8);
  math::MatrixInfo minfo1 (true, true, 1, 8);

  EXPECT_TRUE (minfo < minfo1 || minfo1 < minfo);
}

