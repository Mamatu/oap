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

#include <string>
#include "gtest/gtest.h"
#include "MatchersUtils.hpp"
#include "oapEigen.hpp"
#include "oapHostComplexMatrixApi.hpp"

class oapGenericApiTests_HadamardProduct : public testing::Test {
 public:
   oap::HostProcedures* hostApi;

  virtual void SetUp() {
    hostApi = new oap::HostProcedures();
  }

  virtual void TearDown() {
    delete hostApi;
  }
};

TEST_F(oapGenericApiTests_HadamardProduct, Test1)
{
  math::ComplexMatrix* hostM1 = oap::chost::NewReMatrixWithValue (4, 4, 1);
  math::ComplexMatrix* hostM2 = oap::chost::NewReMatrixWithValue (4, 4, 1);

  math::ComplexMatrix* houtput = oap::chost::NewReMatrix(4, 4);

  auto outputs = std::vector<math::ComplexMatrix*>({houtput});
  hostApi->v2_hadamardProduct (outputs, std::vector<math::ComplexMatrix*>({hostM1}), std::vector<math::ComplexMatrix*>({hostM2}));

  EXPECT_THAT(houtput, MatrixHasValues(1));

  oap::chost::DeleteMatrix(houtput);
  oap::chost::DeleteMatrix(hostM1);
  oap::chost::DeleteMatrix(hostM2);
}

TEST_F(oapGenericApiTests_HadamardProduct, Test2)
{
  math::ComplexMatrix* hostM1 = oap::chost::NewReMatrixWithValue (3, 4, 1);
  math::ComplexMatrix* hostM2 = oap::chost::NewReMatrixWithValue (3, 4, 1);

  math::ComplexMatrix* houtput = oap::chost::NewReMatrix(3, 4);

  auto outputs = std::vector<math::ComplexMatrix*>({houtput});
  hostApi->v2_hadamardProduct (outputs, std::vector<math::ComplexMatrix*>({hostM1}), std::vector<math::ComplexMatrix*>({hostM2}));

  EXPECT_THAT(houtput, MatrixHasValues(1));

  oap::chost::DeleteMatrix(houtput);
  oap::chost::DeleteMatrix(hostM1);
  oap::chost::DeleteMatrix(hostM2);
}

TEST_F(oapGenericApiTests_HadamardProduct, Test3)
{
  math::ComplexMatrix* hostM1 = oap::chost::NewReMatrixWithValue (3, 4, 2);
  math::ComplexMatrix* hostM2 = oap::chost::NewReMatrixWithValue (3, 4, 1);

  math::ComplexMatrix* houtput = oap::chost::NewReMatrix(3, 4);

  auto outputs = std::vector<math::ComplexMatrix*>({houtput});
  hostApi->v2_hadamardProduct (outputs, std::vector<math::ComplexMatrix*>({hostM1}), std::vector<math::ComplexMatrix*>({hostM2}));

  EXPECT_THAT(houtput, MatrixHasValues(2));

  oap::chost::DeleteMatrix(houtput);
  oap::chost::DeleteMatrix(hostM1);
  oap::chost::DeleteMatrix(hostM2);
}

TEST_F(oapGenericApiTests_HadamardProduct, Test4)
{
  math::ComplexMatrix* hostM1 = oap::chost::NewReMatrixWithValue (3, 4, 2);
  math::ComplexMatrix* hostM2 = oap::chost::NewReMatrixWithValue (3, 4, 3);

  math::ComplexMatrix* houtput = oap::chost::NewReMatrix(3, 4);

  auto outputs = std::vector<math::ComplexMatrix*>({houtput});
  hostApi->v2_hadamardProduct (outputs, std::vector<math::ComplexMatrix*>({hostM1}), std::vector<math::ComplexMatrix*>({hostM2}));

  EXPECT_THAT(houtput, MatrixHasValues(6));

  oap::chost::DeleteMatrix(houtput);
  oap::chost::DeleteMatrix(hostM1);
  oap::chost::DeleteMatrix(hostM2);
}

TEST_F(oapGenericApiTests_HadamardProduct, Test5)
{
  hostApi->setMaxThreadsPerBlock (2);

  math::ComplexMatrix* hostM1 = oap::chost::NewReMatrixWithValue (3, 3, 2);
  math::ComplexMatrix* hostM2 = oap::chost::NewReMatrixWithValue (3, 3, 3);

  math::ComplexMatrix* houtput = oap::chost::NewReMatrix(3, 3);

  auto outputs = std::vector<math::ComplexMatrix*>({houtput});
  hostApi->v2_hadamardProduct (outputs, std::vector<math::ComplexMatrix*>({hostM1}), std::vector<math::ComplexMatrix*>({hostM2}));

  EXPECT_THAT(houtput, MatrixHasValues(6));

  oap::chost::DeleteMatrix(houtput);
  oap::chost::DeleteMatrix(hostM1);
  oap::chost::DeleteMatrix(hostM2);
}

TEST_F(oapGenericApiTests_HadamardProduct, Test6)
{
  hostApi->setMaxThreadsPerBlock (9);

  math::ComplexMatrix* hostM1 = oap::chost::NewReMatrixWithValue (4, 4, 2);
  math::ComplexMatrix* hostM2 = oap::chost::NewReMatrixWithValue (4, 4, 3);

  math::ComplexMatrix* houtput = oap::chost::NewReMatrix(4, 4);

  auto outputs = std::vector<math::ComplexMatrix*>({houtput});
  hostApi->v2_hadamardProduct (outputs, std::vector<math::ComplexMatrix*>({hostM1}), std::vector<math::ComplexMatrix*>({hostM2}));

  EXPECT_THAT(houtput, MatrixHasValues(6));

  oap::chost::DeleteMatrix(houtput);
  oap::chost::DeleteMatrix(hostM1);
  oap::chost::DeleteMatrix(hostM2);
}

TEST_F(oapGenericApiTests_HadamardProduct, Test7)
{
  math::ComplexMatrix* hostM1 = oap::chost::NewReMatrixWithValue (32, 32, 2);
  math::ComplexMatrix* hostM2 = oap::chost::NewReMatrixWithValue (32, 32, 3);

  math::ComplexMatrix* houtput = oap::chost::NewReMatrix(32, 32);

  auto outputs = std::vector<math::ComplexMatrix*>({houtput});
  hostApi->v2_hadamardProduct (outputs, std::vector<math::ComplexMatrix*>({hostM1}), std::vector<math::ComplexMatrix*>({hostM2}));

  EXPECT_THAT(houtput, MatrixHasValues(6));

  oap::chost::DeleteMatrix(houtput);
  oap::chost::DeleteMatrix(hostM1);
  oap::chost::DeleteMatrix(hostM2);
}

TEST_F(oapGenericApiTests_HadamardProduct, Test8)
{
  math::ComplexMatrix* hostM1 = oap::chost::NewReMatrixWithValue (33, 33, 2);
  math::ComplexMatrix* hostM2 = oap::chost::NewReMatrixWithValue (33, 33, 3);

  math::ComplexMatrix* houtput = oap::chost::NewReMatrix(33, 33);

  auto outputs = std::vector<math::ComplexMatrix*>({houtput});
  hostApi->v2_hadamardProduct (outputs, std::vector<math::ComplexMatrix*>({hostM1}), std::vector<math::ComplexMatrix*>({hostM2}));

  EXPECT_THAT(houtput, MatrixHasValues(6));

  oap::chost::DeleteMatrix(houtput);
  oap::chost::DeleteMatrix(hostM1);
  oap::chost::DeleteMatrix(hostM2);
}

TEST_F(oapGenericApiTests_HadamardProduct, Test9)
{
  math::ComplexMatrix* hostM1 = oap::chost::NewReMatrixWithValue (312, 456, 2);
  math::ComplexMatrix* hostM2 = oap::chost::NewReMatrixWithValue (312, 456, 3);

  math::ComplexMatrix* houtput = oap::chost::NewReMatrix(312, 456);

  auto outputs = std::vector<math::ComplexMatrix*>({houtput});
  hostApi->v2_hadamardProduct (outputs, std::vector<math::ComplexMatrix*>({hostM1}), std::vector<math::ComplexMatrix*>({hostM2}));

  EXPECT_THAT(houtput, MatrixHasValues(6));

  oap::chost::DeleteMatrix(houtput);
  oap::chost::DeleteMatrix(hostM1);
  oap::chost::DeleteMatrix(hostM2);
}
