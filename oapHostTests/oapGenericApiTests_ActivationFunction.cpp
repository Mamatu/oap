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

class oapGenericApiTests_ActivationFunction : public testing::Test {
 public:
   oap::HostProcedures* hostApi;

  virtual void SetUp() {
    hostApi = new oap::HostProcedures();
  }

  virtual void TearDown() {
    delete hostApi;
  }
};

floatt sigmoid (floatt x)
{
  return 1. / (1. + exp(-x));
}

TEST_F(oapGenericApiTests_ActivationFunction, SigmoidTest_1)
{
  using namespace std::placeholders;
  oap::HostProcedures hp;

  math::ComplexMatrix* output = oap::chost::NewReMatrixWithValue (1, 1, 0.);
  math::ComplexMatrix* param = oap::chost::NewReMatrixWithValue (1, 1, 2.);
  std::vector<math::ComplexMatrix*> outputs = {output};
  std::vector<math::ComplexMatrix*> params = {param};

  oap::HostComplexMatrixUPtr output1 = oap::chost::NewReMatrixWithValue (1, 1, 0.);
  oap::HostComplexMatrixUPtr param1 = oap::chost::NewReMatrixWithValue (1, 1, 2.);

  hp.v2_sigmoid (outputs, params);
  hp.sigmoid (output1, param1);
  EXPECT_DOUBLE_EQ (sigmoid(param->re.mem.ptr[0]), output->re.mem.ptr[0]);
  EXPECT_DOUBLE_EQ (sigmoid(param->re.mem.ptr[0]), output1->re.mem.ptr[0]);
  EXPECT_DOUBLE_EQ (output1->re.mem.ptr[0], output->re.mem.ptr[0]);

  oap::chost::deleteMatrices (outputs);
  oap::chost::deleteMatrices (params);
}

TEST_F(oapGenericApiTests_ActivationFunction, SigmoidTest_2)
{
  using namespace std::placeholders;
  oap::HostProcedures hp;

  math::ComplexMatrix* output = oap::chost::NewReMatrixWithValue (1, 1, 0.);
  math::ComplexMatrix* output1 = oap::chost::NewReMatrixWithValue (1, 1, 0.);
  math::ComplexMatrix* param = oap::chost::NewReMatrixWithValue (1, 1, 2.);
  math::ComplexMatrix* param1 = oap::chost::NewReMatrixWithValue (1, 1, 2.);
  std::vector<math::ComplexMatrix*> outputs = {output, output1};
  std::vector<math::ComplexMatrix*> params = {param, param1};

  oap::HostComplexMatrixUPtr output_ = oap::chost::NewReMatrixWithValue (1, 1, 0.);
  oap::HostComplexMatrixUPtr param_ = oap::chost::NewReMatrixWithValue (1, 1, 2.);

  hp.v2_sigmoid (outputs, params);
  hp.sigmoid (output_, param_);
  EXPECT_DOUBLE_EQ (sigmoid(param->re.mem.ptr[0]), output->re.mem.ptr[0]);
  EXPECT_DOUBLE_EQ (sigmoid(param1->re.mem.ptr[0]), output1->re.mem.ptr[0]);
  EXPECT_DOUBLE_EQ (sigmoid(param->re.mem.ptr[0]), output_->re.mem.ptr[0]);
  EXPECT_DOUBLE_EQ (output_->re.mem.ptr[0], output->re.mem.ptr[0]);

  oap::chost::deleteMatrices (outputs);
  oap::chost::deleteMatrices (params);
}

TEST_F(oapGenericApiTests_ActivationFunction, SigmoidTest_3)
{
  oap::HostProcedures hp;

  math::ComplexMatrix* output = oap::chost::NewReMatrixWithValue (4, 4, 0.);
  math::ComplexMatrix* output1 = oap::chost::NewReMatrixWithValue (4, 4, 0.);
  math::ComplexMatrix* param = oap::chost::NewReMatrixWithValue (4, 4, 2.);
  math::ComplexMatrix* param1 = oap::chost::NewReMatrixWithValue (4, 4, 2.);
  std::vector<math::ComplexMatrix*> outputs = {output, output1};
  std::vector<math::ComplexMatrix*> params = {param, param1};

  oap::HostComplexMatrixUPtr output_ = oap::chost::NewReMatrixWithValue (4, 4, 0.);
  oap::HostComplexMatrixUPtr param_ = oap::chost::NewReMatrixWithValue (4, 4, 2.);

  hp.v2_sigmoid (outputs, params);
  hp.sigmoid (output_, param_);
  EXPECT_THAT(output, MatrixHasValues (sigmoid(2)));
  EXPECT_THAT(output1, MatrixHasValues (sigmoid(2)));
  EXPECT_THAT(output_.get(), MatrixHasValues (sigmoid(2)));

  oap::chost::deleteMatrices (outputs);
  oap::chost::deleteMatrices (params);
}

TEST_F(oapGenericApiTests_ActivationFunction, SigmoidTest_4)
{
  oap::HostProcedures hp;

  oap::HostComplexMatrixUPtr output = oap::chost::NewReMatrixWithValue (1, 4, 0.);
  oap::HostComplexMatrixUPtr output1 = oap::chost::NewReMatrixWithValue (1, 4, 0.);

  math::ComplexMatrix* suboutput = oap::chost::NewSharedSubMatrix ({0, 0}, {1, 3}, output);
  math::ComplexMatrix* suboutput1 = oap::chost::NewSharedSubMatrix ({0, 0}, {1, 3}, output1);
  math::ComplexMatrix* param = oap::chost::NewReMatrixWithValue (1, 3, 2.);
  math::ComplexMatrix* param1 = oap::chost::NewReMatrixWithValue (1, 3, 2.);
  std::vector<math::ComplexMatrix*> outputs = {suboutput, suboutput1};
  std::vector<math::ComplexMatrix*> params = {param, param1};

  oap::HostComplexMatrixUPtr output_ = oap::chost::NewReMatrixWithValue (1, 4, 0.);
  oap::HostComplexMatrixUPtr param_ = oap::chost::NewReMatrixWithValue (1, 4, 2.);

  hp.v2_sigmoid (outputs, params);
  hp.sigmoid (output_, param_);
  EXPECT_THAT(suboutput, MatrixHasValues (sigmoid(2)));
  EXPECT_THAT(suboutput1, MatrixHasValues (sigmoid(2)));
  EXPECT_THAT(output_.get(), MatrixHasValues (sigmoid(2)));

  oap::chost::deleteMatrices (outputs);
  oap::chost::deleteMatrices (params);
}
