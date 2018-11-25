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
#include <stdio.h>
#include <math.h>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
//#include "gmock/gmock_link_test.h"

#include "MatrixUtils.h"
#include "MatrixPrinter.h"

#include "oapHostMatrixPtr.h"

class OapMatrixPrinterTests : public testing::Test {
 public:
  OapMatrixPrinterTests() {}

  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(OapMatrixPrinterTests, TestRe1x1WithZero)
{
  oap::HostMatrixPtr matrix = oap::host::NewMatrix(true, false, 1, 1);
  std::string str;
  matrixUtils::PrintMatrix (str, matrix.get());
  EXPECT_THAT(str, ::testing::HasSubstr("[0]"));
  printf("%s\n", str.c_str());
}

TEST_F(OapMatrixPrinterTests, TestRe1x10WithZero)
{
  oap::HostMatrixPtr matrix = oap::host::NewMatrix(true, false, 1, 10);
  std::string str;
  matrixUtils::PrintMatrix (str, matrix.get());
  EXPECT_THAT(str, ::testing::HasSubstr("[0 <repeats 10 times>]"));
  printf("%s\n", str.c_str());
}

TEST_F(OapMatrixPrinterTests, TestRe10x1WithZero)
{
  oap::HostMatrixPtr matrix = oap::host::NewMatrix(true, false, 10, 1);
  std::string str;
  matrixUtils::PrintMatrix (str, matrix.get());
  EXPECT_THAT(str, ::testing::HasSubstr("[0 <repeats 10 times>]"));
  printf("%s\n", str.c_str());
}

TEST_F(OapMatrixPrinterTests, TestRe10x10WithZero)
{
  oap::HostMatrixPtr matrix = oap::host::NewMatrix(true, false, 10, 10);
  std::string str;
  matrixUtils::PrintMatrix (str, matrix.get());
  EXPECT_THAT(str, ::testing::HasSubstr("[0 <repeats 100 times>]"));
  printf("%s\n", str.c_str());
}

TEST_F(OapMatrixPrinterTests, TestIm1x1WithZero)
{
  oap::HostMatrixPtr matrix = oap::host::NewMatrix(false, true, 1, 1);
  std::string str;
  matrixUtils::PrintMatrix (str, matrix.get());
  EXPECT_THAT(str, ::testing::HasSubstr("[0]"));
  printf("%s\n", str.c_str());
}

TEST_F(OapMatrixPrinterTests, TestIm1x10WithZero)
{
  oap::HostMatrixPtr matrix = oap::host::NewMatrix(false, true, 1, 10);
  std::string str;
  matrixUtils::PrintMatrix (str, matrix.get());
  EXPECT_THAT(str, ::testing::HasSubstr("[0 <repeats 10 times>]"));
  printf("%s\n", str.c_str());
}

TEST_F(OapMatrixPrinterTests, TestIm1x5WithZeroNoRepeats)
{
  oap::HostMatrixPtr matrix = oap::host::NewMatrix(false, true, 1, 5);
  std::string str;
  matrixUtils::PrintMatrix (str, matrix.get(), matrixUtils::PrintArgs<floatt>(0, true));
  EXPECT_THAT(str, ::testing::HasSubstr("[0, 0, 0, 0, 0]"));
  printf("%s\n", str.c_str());
}

TEST_F(OapMatrixPrinterTests, TestIm10x1WithZero)
{
  oap::HostMatrixPtr matrix = oap::host::NewMatrix(false, true, 10, 1);
  std::string str;
  matrixUtils::PrintMatrix (str, matrix.get());
  EXPECT_THAT(str, ::testing::HasSubstr("[0 <repeats 10 times>]"));
  printf("%s\n", str.c_str());
}

TEST_F(OapMatrixPrinterTests, TestIm10x10WithZero)
{
  oap::HostMatrixPtr matrix = oap::host::NewMatrix(false, true, 10, 10);
  std::string str;
  matrixUtils::PrintMatrix (str, matrix.get());
  EXPECT_THAT(str, ::testing::HasSubstr("[0 <repeats 100 times>]"));
  printf("%s\n", str.c_str());
}

TEST_F(OapMatrixPrinterTests, Test5x1)
{
  oap::HostMatrixPtr matrix = oap::host::NewMatrix(true, true, 5, 1);
  
  for (size_t idx = 0; idx < 5; ++idx)
  {
    matrix->reValues[idx] = idx;
    matrix->imValues[idx] = idx;
  }

  std::string str;
  matrixUtils::PrintMatrix (str, matrix.get());
  EXPECT_THAT(str, ::testing::HasSubstr("[0, 1, 2, 3, 4]"));
  printf("%s\n", str.c_str());
}

TEST_F(OapMatrixPrinterTests, Tests5x2DifferentArgs)
{
  {
    oap::HostMatrixPtr matrix = oap::host::NewMatrix(true, true, 5, 2);

    for (size_t idx = 0; idx < 10; ++idx)
    {
      matrix->reValues[idx] = idx;
      matrix->imValues[idx] = idx;
    }

    std::string str;
    matrixUtils::PrintMatrix (str, matrix.get(), matrixUtils::PrintArgs<floatt>(0, false, "\n", matrix->columns));
    EXPECT_THAT(str, ::testing::HasSubstr("[0, 1, 2, 3, 4\n5, 6, 7, 8, 9]"));
    printf("%s\n", str.c_str());
  }
  {
    oap::HostMatrixPtr matrix = oap::host::NewMatrix(true, true, 5, 2);

    for (size_t idx = 0; idx < 10; ++idx)
    {
      matrix->reValues[idx] = idx;
      matrix->imValues[idx] = idx;
    }

    std::string str;
    matrixUtils::PrintMatrix (str, matrix.get(), matrixUtils::PrintArgs<floatt>(0, false, "|", matrix->columns));
    EXPECT_THAT(str, ::testing::HasSubstr("[0, 1, 2, 3, 4|5, 6, 7, 8, 9]"));
    printf("%s\n", str.c_str());
  }
}


