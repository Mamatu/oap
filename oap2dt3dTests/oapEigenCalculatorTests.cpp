/*
 * Copyright 2016 Marcin Matula
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
#include "PngFile.h"
//#include "EigenCalculator.h"
#include "Exceptions.h"

using namespace ::testing;

class OapEigenCalculatorTests : public testing::Test {
 public:
  virtual void SetUp() {}

  virtual void TearDown() {}

 public:
};
#if 0 // Temporary disable from compilation
TEST_F(OapEigenCalculatorTests, NotInitializedTest) {
  oap::EigenCalculator eigenCalc;
  EXPECT_THROW(eigenCalc.calculate(), oap::exceptions::NotInitialzed);
  EXPECT_THROW(eigenCalc.getEigenvalues(NULL), oap::exceptions::NotInitialzed);
  EXPECT_THROW(eigenCalc.getEigenvectors(NULL), oap::exceptions::NotInitialzed);
}
#endif
