/*
 * Copyright 2016 - 2018 Marcin Matula
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
  }
  {
    math::MatrixInfo minfo (false, true, 1024, 1024);
    EXPECT_EQ(sizeof(floatt), minfo.getSize().first);
    EXPECT_EQ(math::MatrixInfo::MB, minfo.getSize().second);
  }
  {
    math::MatrixInfo minfo (true, false, 1024, 1024);
    EXPECT_EQ(sizeof(floatt), minfo.getSize().first);
    EXPECT_EQ(math::MatrixInfo::MB, minfo.getSize().second);
  }
  {
    math::MatrixInfo minfo (true, true, 1024, 1024);
    EXPECT_EQ(sizeof(floatt) * 2, minfo.getSize().first);
    EXPECT_EQ(math::MatrixInfo::MB, minfo.getSize().second);
  }
}

TEST_F(OapMatrixInfoTests, LargeSizeTests)
{
  {
    math::MatrixInfo minfo (true, false, 1024*10, 1024*1024);
    EXPECT_EQ(sizeof(floatt) * 10, minfo.getSize().first);
    EXPECT_EQ(math::MatrixInfo::GB, minfo.getSize().second);
  }
  {
    math::MatrixInfo minfo (true, false, 1024, 1024*1024*2);
    EXPECT_EQ(sizeof(floatt) * 2, minfo.getSize().first);
    EXPECT_EQ(math::MatrixInfo::GB, minfo.getSize().second);
  }

}

