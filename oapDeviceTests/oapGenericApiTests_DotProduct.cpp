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

#include "oapHostComplexMatrixPtr.h"
#include "oapDeviceComplexMatrixPtr.h"
#include "oapFuncTests.h"

#include "CudaMatchersUtils.h"

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

  math::MatrixInfo minfo(true, false, 1, 3);
  math::MatrixInfo minfo1(true, false, 3, 3);
  std::vector<math::ComplexMatrix*> outputs = oap::cuda::NewDeviceMatrices (minfo, params1_raw.size());
  std::vector<math::ComplexMatrix*> params1 = oap::cuda::NewDeviceMatricesCopyOfArray (minfo1, params1_raw);
  std::vector<math::ComplexMatrix*> params2 = oap::cuda::NewDeviceMatricesCopyOfArray (minfo, params2_raw);

  m_cuApi->v2_multiply (outputs, params1, params2);

  PRINT_CUMATRIX_CARRAY(outputs);
  /*PRINT_CUMATRIX(outputs[0]);
  PRINT_CUMATRIX(outputs[1]);
  PRINT_CUMATRIX(outputs[2]);
  PRINT_CUMATRIX(outputs[3]);
  PRINT_CUMATRIX(outputs[4]);
  PRINT_CUMATRIX(outputs[5]);
  PRINT_CUMATRIX(outputs[6]);*/

  /*std::vector<std::vector<floatt>> expected_raw =
    {
      { 0.264996029, -0.327199891,  0.000000000},
      { 0.559138926, -0.371919774,  0.000000000},
      { 0.460739674, -0.364118158,  0.000000000},
      { 0.384748063, -0.381777198,  0.000000000},
      { 0.431693179, -0.313980184,  0.000000000},
      { 0.378665560, -0.331393754,  0.000000000},
      { 0.315650421, -0.287728207,  0.000000000}
    };*/
  std::vector<std::vector<floatt>> expected_raw =
  {
    { 0.264996029, -0.145481860,  0.000000000},
    { 0.559138926, -0.396280104,  0.000000000},
    { 0.460739674, -0.340755302,  0.000000000},
    { 0.384748063, -0.391752492,  0.000000000},
    { 0.431693179, -0.134758677,  0.000000000},
    { 0.378665560, -0.190524188,  0.000000000},
    { 0.315650421, -0.001690069,  0.000000000}
  };

  for (uintt idx = 0; idx < outputs.size(); ++idx)
  {
    oap::HostMatrixPtr matrix = oap::host::NewReMatrixCopyOfArray (1, 3, expected_raw[idx].data());
    EXPECT_THAT (matrix.get(), oap::cuda::MatrixIsEqualHK (outputs[idx]));
  }

  oap::cuda::deleteDeviceMatrices (outputs);
  oap::cuda::deleteDeviceMatrices (params1);
  oap::cuda::deleteDeviceMatrices (params2);
}

TEST_F(OapGenericApiTests_DotProduct, Test_2)
{
  std::vector<std::vector<floatt>> params1_raw =
  {
    { 0.432950500, -0.228049107,  0.146509119,  0.264489281, -0.087344033, -0.351855513,  0.000000000,  0.000000000,  0.000000000},
    { 0.432950500, -0.228049107,  0.146509119,  0.264489281, -0.087344033, -0.351855513,  0.000000000,  0.000000000,  0.000000000},
    { 0.432950500, -0.228049107,  0.146509119,  0.264489281, -0.087344033, -0.351855513,  0.000000000,  0.000000000,  0.000000000},
    { 0.432950500, -0.228049107,  0.146509119,  0.264489281, -0.087344033, -0.351855513,  0.000000000,  0.000000000,  0.000000000},
    { 0.432950500, -0.228049107,  0.146509119,  0.264489281, -0.087344033, -0.351855513,  0.000000000,  0.000000000,  0.000000000},
    { 0.432950500, -0.228049107,  0.146509119,  0.264489281, -0.087344033, -0.351855513,  0.000000000,  0.000000000,  0.000000000},
    { 0.432950500, -0.228049107,  0.146509119,  0.264489281, -0.087344033, -0.351855513,  0.000000000,  0.000000000,  0.000000000}
  };

  std::vector<std::vector<floatt>> params2_raw =
  {
    {-0.235627664, -0.112502067,  1.000000000},
    { 0.138050341, -0.034401554,  1.000000000},
    {0.053861780, 0.270415441, 1.000000000},
    {0.226386317, 0.443905679, 1.000000000},
    {-0.540624305, -0.315438209,  1.000000000},
    {0.458402072, 0.293494103, 1.000000000},
    {-0.147255045, -0.207675798,  1.000000000}
  };

  math::MatrixInfo minfo(true, false, 1, 3);
  math::MatrixInfo minfo1(true, false, 3, 3);
  std::vector<math::ComplexMatrix*> outputs = oap::cuda::NewDeviceMatrices (minfo, params1_raw.size());
  std::vector<math::ComplexMatrix*> params1 = oap::cuda::NewDeviceMatricesCopyOfArray (minfo1, params1_raw);
  std::vector<math::ComplexMatrix*> params2 = oap::cuda::NewDeviceMatricesCopyOfArray (minfo, params2_raw);

  m_cuApi->v2_multiply (outputs, params1, params2);

  PRINT_CUMATRIX_CARRAY(outputs);
  /*PRINT_CUMATRIX(outputs[0]);
  PRINT_CUMATRIX(outputs[1]);
  PRINT_CUMATRIX(outputs[2]);
  PRINT_CUMATRIX(outputs[3]);
  PRINT_CUMATRIX(outputs[4]);
  PRINT_CUMATRIX(outputs[5]);
  PRINT_CUMATRIX(outputs[6]);*/

  /*std::vector<std::vector<floatt>> expected_raw =
    {
      { 0.264996029, -0.327199891,  0.000000000},
      { 0.559138926, -0.371919774,  0.000000000},
      { 0.460739674, -0.364118158,  0.000000000},
      { 0.384748063, -0.381777198,  0.000000000},
      { 0.431693179, -0.313980184,  0.000000000},
      { 0.378665560, -0.331393754,  0.000000000},
      { 0.315650421, -0.287728207,  0.000000000}
    };*/
  std::vector<std::vector<floatt>> expected_raw =
  {
    { 0.264996029, -0.145481860,  0.000000000},
    { 0.559138926, -0.396280104,  0.000000000},
    { 0.460739674, -0.340755302,  0.000000000},
    { 0.384748063, -0.391752492,  0.000000000},
    { 0.431693179, -0.134758677,  0.000000000},
    { 0.378665560, -0.190524188,  0.000000000},
    { 0.315650421, -0.001690069,  0.000000000}
  };

  for (uintt idx = 0; idx < outputs.size(); ++idx)
  {
    //oap::HostMatrixPtr matrix = oap::host::NewReMatrixCopyOfArray (1, 3, expected_raw[idx].data());
    //EXPECT_THAT (matrix.get(), oap::cuda::MatrixIsEqualHK (outputs[idx]));
  }

  oap::cuda::deleteDeviceMatrices (outputs);
  oap::cuda::deleteDeviceMatrices (params1);
  oap::cuda::deleteDeviceMatrices (params2);
}

TEST_F(OapGenericApiTests_DotProduct, Test_3)
{
  std::vector<std::vector<floatt>> params1_raw =
  {
    { 0.456317577, -0.390043207,  0.254673487,  0.061695443,  0.133965712, -0.498161629,  0.000000000,  0.000000000,  0.000000000},
    { 0.456317577, -0.390043207,  0.254673487,  0.061695443,  0.133965712, -0.498161629,  0.000000000,  0.000000000,  0.000000000},
    { 0.456317577, -0.390043207,  0.254673487,  0.061695443,  0.133965712, -0.498161629,  0.000000000,  0.000000000,  0.000000000},
    { 0.456317577, -0.390043207,  0.254673487,  0.061695443,  0.133965712, -0.498161629,  0.000000000,  0.000000000,  0.000000000},
    { 0.456317577, -0.390043207,  0.254673487,  0.061695443,  0.133965712, -0.498161629,  0.000000000,  0.000000000,  0.000000000},
    { 0.456317577, -0.390043207,  0.254673487,  0.061695443,  0.133965712, -0.498161629,  0.000000000,  0.000000000,  0.000000000},
    { 0.456317577, -0.390043207,  0.254673487,  0.061695443,  0.133965712, -0.498161629,  0.000000000,  0.000000000,  0.000000000}
  };

  std::vector<std::vector<floatt>> params2_raw =
  {
    {-0.664504746, -0.269933422,  1.000000000},
    {0.038228665, 0.031585489, 1.000000000},
    {-0.451562665,  0.148885711,  1.000000000},
    {0.565967395, 0.006534367, 1.000000000},
    {0.262102990, 0.022468290, 1.000000000},
    {0.629933906, 0.029740866, 1.000000000},
    {-0.407287159,  0.549972349,  1.000000000}
  };

  math::MatrixInfo minfo(true, false, 1, 3);
  math::MatrixInfo minfo1(true, false, 3, 3);
  std::vector<math::ComplexMatrix*> outputs = oap::cuda::NewDeviceMatrices (minfo, params1_raw.size());
  std::vector<math::ComplexMatrix*> params1 = oap::cuda::NewDeviceMatricesCopyOfArray (minfo1, params1_raw);
  std::vector<math::ComplexMatrix*> params2 = oap::cuda::NewDeviceMatricesCopyOfArray (minfo, params2_raw);

  m_cuApi->v2_multiply (outputs, params1, params2);

  PRINT_CUMATRIX_CARRAY(outputs);  

  /*std::vector<std::vector<floatt>> expected_raw =
  {
    { 0.264996029, -0.145481860,  0.000000000},
    { 0.559138926, -0.396280104,  0.000000000},
    { 0.460739674, -0.340755302,  0.000000000},
    { 0.384748063, -0.391752492,  0.000000000},
    { 0.431693179, -0.134758677,  0.000000000},
    { 0.378665560, -0.190524188,  0.000000000},
    { 0.315650421, -0.001690069,  0.000000000}
  };*/

  oap::cuda::deleteDeviceMatrices (outputs);
  oap::cuda::deleteDeviceMatrices (params1);
  oap::cuda::deleteDeviceMatrices (params2);
}
