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

#include "gtest/gtest.h"
#include "CuProceduresApi.h"

#include "MatchersUtils.h"
#include "MathOperationsCpu.h"

#include "oapHostMatrixUtils.h"
#include "oapCudaMatrixUtils.h"
#include "oapHostMemoryApi.h"
#include "oapCudaMemoryApi.h"

#include "oapHostMatrixPtr.h"
#include "oapDeviceMatrixPtr.h"
#include "oapFuncTests.h"

using namespace ::testing;

class OapGenericApiTests_DotProduct : public testing::Test
{
  public:
    oap::CuProceduresApi* m_cuApi;
    CUresult status;

    virtual void SetUp() {
      status = CUDA_SUCCESS;
      oap::cuda::Context::Instance().create();
      m_cuApi = new oap::CuProceduresApi();
    }

    virtual void TearDown() {
      delete m_cuApi;
      oap::cuda::Context::Instance().destroy();
    }
};

TEST_F(OapGenericApiTests_DotProduct, Test_1)
{
  std::vector<std::vector<floatt>> params1_raw =
    {
      { 0.313512845,  0.078433419,  0.394670532, -0.078382355, -0.330297794, -0.190181526,  0.000000000,  0.000000000,  0.000000000},
      { 0.313512845,  0.078433419,  0.394670532, -0.078382355, -0.330297794, -0.190181526,  0.000000000,  0.000000000,  0.000000000},
      { 0.313512845,  0.078433419,  0.394670532, -0.078382355, -0.330297794, -0.190181526,  0.000000000,  0.000000000,  0.000000000},
      { 0.313512845,  0.078433419,  0.394670532, -0.078382355, -0.330297794, -0.190181526,  0.000000000,  0.000000000,  0.000000000},
      { 0.313512845,  0.078433419,  0.394670532, -0.078382355, -0.330297794, -0.190181526,  0.000000000,  0.000000000,  0.000000000},
      { 0.313512845,  0.078433419,  0.394670532, -0.078382355, -0.330297794, -0.190181526,  0.000000000,  0.000000000,  0.000000000},
      { 0.313512845,  0.078433419,  0.394670532, -0.078382355, -0.330297794, -0.190181526,  0.000000000,  0.000000000,  0.000000000}
    };

  std::vector<std::vector<floatt>> params2_raw =
    {
      {-0.403730130, -0.039522964,  1.000000000},
      {0.391752045, 0.531012116, 1.000000000},
      {0.102792422, 0.431479308, 1.000000000},
      {-0.195958194,  0.656772873,  1.000000000},
      { 0.170171300, -0.208179640,  1.000000000},
      {-0.054548469,  0.013982228,  1.000000000},
      {-0.116176394, -0.543101653,  1.000000000}
    };

  std::vector<math::Matrix*> outputs;
  std::vector<math::Matrix*> params1;
  std::vector<math::Matrix*> params2;
  for (auto& vec : params1_raw)
  {
    math::Matrix* output = oap::cuda::NewDeviceReMatrix (1, 3);
    outputs.push_back (output);

    math::Matrix* matrix = oap::cuda::NewDeviceReMatrixCopyOfArray (3, 3, vec.data());
    params1.push_back (matrix);
  }

  for (auto& vec : params2_raw)
  {
    math::Matrix* matrix = oap::cuda::NewDeviceReMatrixCopyOfArray (1, 3, vec.data());
    params2.push_back (matrix);
  }

  m_cuApi->v2_multiply (outputs, params1, params2);

  PRINT_CUMATRIX_CARRAY(outputs[0]);
  PRINT_CUMATRIX_CARRAY(outputs[1]);
  PRINT_CUMATRIX_CARRAY(outputs[2]);
  PRINT_CUMATRIX_CARRAY(outputs[3]);
  PRINT_CUMATRIX_CARRAY(outputs[4]);
  PRINT_CUMATRIX_CARRAY(outputs[5]);
  PRINT_CUMATRIX_CARRAY(outputs[6]);

  oap::cuda::deleteDeviceMatrices (outputs);
  oap::cuda::deleteDeviceMatrices (params1);
  oap::cuda::deleteDeviceMatrices (params2);
}
