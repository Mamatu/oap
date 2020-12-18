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
#include "MatchersUtils.h"
#include "MathOperationsCpu.h"
#include "oapHostMatrixUtils.h"

class oapGenericApiTests_ActivationFunction : public testing::Test {
 public:
  HostProcedures* hostApi;

  virtual void SetUp() {
    hostApi = new HostProcedures();
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
  HostProcedures hp;

  math::Matrix* output = oap::host::NewReMatrixWithValue (1, 1, 0.);
  math::Matrix* param = oap::host::NewReMatrixWithValue (1, 1, 2.);
  std::vector<math::Matrix*> outputs = {output};
  std::vector<math::Matrix*> params = {param};

  oap::HostMatrixUPtr output1 = oap::host::NewReMatrixWithValue (1, 1, 0.);
  oap::HostMatrixUPtr param1 = oap::host::NewReMatrixWithValue (1, 1, 2.);

  hp.v2_sigmoid (outputs, params);
  hp.sigmoid (output1, param1);
  EXPECT_DOUBLE_EQ (sigmoid(param->re.ptr[0]), output->re.ptr[0]);
  EXPECT_DOUBLE_EQ (sigmoid(param->re.ptr[0]), output1->re.ptr[0]);
  EXPECT_DOUBLE_EQ (output1->re.ptr[0], output->re.ptr[0]);

  oap::host::deleteMatrices (outputs);
  oap::host::deleteMatrices (params);
}

TEST_F(oapGenericApiTests_ActivationFunction, SigmoidTest_2)
{
  using namespace std::placeholders;
  HostProcedures hp;

  math::Matrix* output = oap::host::NewReMatrixWithValue (1, 1, 0.);
  math::Matrix* output1 = oap::host::NewReMatrixWithValue (1, 1, 0.);
  math::Matrix* param = oap::host::NewReMatrixWithValue (1, 1, 2.);
  math::Matrix* param1 = oap::host::NewReMatrixWithValue (1, 1, 2.);
  std::vector<math::Matrix*> outputs = {output, output1};
  std::vector<math::Matrix*> params = {param, param1};

  oap::HostMatrixUPtr output_ = oap::host::NewReMatrixWithValue (1, 1, 0.);
  oap::HostMatrixUPtr param_ = oap::host::NewReMatrixWithValue (1, 1, 2.);

  hp.v2_sigmoid (outputs, params);
  hp.sigmoid (output_, param_);
  EXPECT_DOUBLE_EQ (sigmoid(param->re.ptr[0]), output->re.ptr[0]);
  EXPECT_DOUBLE_EQ (sigmoid(param1->re.ptr[0]), output1->re.ptr[0]);
  EXPECT_DOUBLE_EQ (sigmoid(param->re.ptr[0]), output_->re.ptr[0]);
  EXPECT_DOUBLE_EQ (output_->re.ptr[0], output->re.ptr[0]);

  oap::host::deleteMatrices (outputs);
  oap::host::deleteMatrices (params);
}

TEST_F(oapGenericApiTests_ActivationFunction, SigmoidTest_3)
{
  HostProcedures hp;

  math::Matrix* output = oap::host::NewReMatrixWithValue (4, 4, 0.);
  math::Matrix* output1 = oap::host::NewReMatrixWithValue (4, 4, 0.);
  math::Matrix* param = oap::host::NewReMatrixWithValue (4, 4, 2.);
  math::Matrix* param1 = oap::host::NewReMatrixWithValue (4, 4, 2.);
  std::vector<math::Matrix*> outputs = {output, output1};
  std::vector<math::Matrix*> params = {param, param1};

  oap::HostMatrixUPtr output_ = oap::host::NewReMatrixWithValue (4, 4, 0.);
  oap::HostMatrixUPtr param_ = oap::host::NewReMatrixWithValue (4, 4, 2.);

  hp.v2_sigmoid (outputs, params);
  hp.sigmoid (output_, param_);
  EXPECT_THAT(output, MatrixHasValues (sigmoid(2)));
  EXPECT_THAT(output1, MatrixHasValues (sigmoid(2)));
  EXPECT_THAT(output_.get(), MatrixHasValues (sigmoid(2)));

  oap::host::deleteMatrices (outputs);
  oap::host::deleteMatrices (params);
}

TEST_F(oapGenericApiTests_ActivationFunction, SigmoidTest_4)
{
  HostProcedures hp;

  oap::HostMatrixUPtr output = oap::host::NewReMatrixWithValue (1, 4, 0.);
  oap::HostMatrixUPtr output1 = oap::host::NewReMatrixWithValue (1, 4, 0.);

  math::Matrix* suboutput = oap::host::NewSharedSubMatrix ({0, 0}, {1, 3}, output);
  math::Matrix* suboutput1 = oap::host::NewSharedSubMatrix ({0, 0}, {1, 3}, output1);
  math::Matrix* param = oap::host::NewReMatrixWithValue (1, 3, 2.);
  math::Matrix* param1 = oap::host::NewReMatrixWithValue (1, 3, 2.);
  std::vector<math::Matrix*> outputs = {suboutput, suboutput1};
  std::vector<math::Matrix*> params = {param, param1};

  oap::HostMatrixUPtr output_ = oap::host::NewReMatrixWithValue (1, 4, 0.);
  oap::HostMatrixUPtr param_ = oap::host::NewReMatrixWithValue (1, 4, 2.);

  hp.v2_sigmoid (outputs, params);
  hp.sigmoid (output_, param_);
  EXPECT_THAT(suboutput, MatrixHasValues (sigmoid(2)));
  EXPECT_THAT(suboutput1, MatrixHasValues (sigmoid(2)));
  EXPECT_THAT(output_.get(), MatrixHasValues (sigmoid(2)));

  oap::host::deleteMatrices (outputs);
  oap::host::deleteMatrices (params);
}
