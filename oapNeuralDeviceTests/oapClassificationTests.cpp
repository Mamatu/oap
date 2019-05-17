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
#include "gtest/gtest.h"
#include "CuProceduresApi.h"

#include "PatternsClassification.h"
#include "Controllers.h"

class OapClassificationTests : public testing::Test
{
 public:
  CUresult status;

  virtual void SetUp()
  {}

  virtual void TearDown()
  {}
};

TEST_F(OapClassificationTests, LetterClassification)
{
  oap::PatternsClassificationParser::Args args;
  oap::PatternsClassification pc;

  args.m_onOutput1 = [](const std::vector<floatt>& outputs)
  {
    EXPECT_EQ(1, outputs.size());
    EXPECT_LE(0.5, outputs[0]);
  };

  args.m_onOutput2 = [](const std::vector<floatt>& outputs)
  {
    EXPECT_EQ(1, outputs.size());
    EXPECT_GE(0.5, outputs[0]);
  };

  args.m_onOpenFile = [](const oap::OptSize& width, const oap::OptSize& height, bool isLoaded)
  {
    EXPECT_EQ(20, width.optSize);
    EXPECT_EQ(20, height.optSize);
    EXPECT_TRUE(isLoaded);
  };

  pc.run (args); 
}
