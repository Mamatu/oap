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

#include "MatchersUtils.hpp"
#include "oapEigen.hpp"

#include "oapHostComplexMatrixApi.hpp"
#include "oapHostMemoryApi.hpp"

#include "oapHostComplexMatrixPtr.hpp"
#include "oapFuncTests.hpp"

#include <vector>
#include <iostream>

#include "HostProcedures.hpp"

using namespace ::testing;

class OapGenericApiTests_DotProduct : public testing::Test
{
 public:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

TEST_F(OapGenericApiTests_DotProduct, Test_1)
{
  oap::HostProcedures hp;
  oap::Memory memory = oap::host::NewMemoryWithValues ({1, 1}, 0.);

  oap::HostComplexMatrixUPtr output1 = oap::chost::NewReMatrixFromMemory (1, 1, memory, {0, 0});

  oap::HostComplexMatrixUPtr matrix1 = oap::chost::NewReMatrixWithValue (1, 1, 2.);

  oap::HostComplexMatrixUPtr matrix2 = oap::chost::NewReMatrixWithValue (1, 1, 1.);

  std::vector<math::ComplexMatrix*> outputs = {output1};
  hp.v2_multiply (outputs, std::vector<math::ComplexMatrix*>({matrix1}), std::vector<math::ComplexMatrix*>({matrix2}));

  std::vector<floatt> expected1 =
  {
    2,
  };

  std::vector<floatt> actual1 = {oap::common::GetValue (output1->re.mem, output1->re.reg, 0, 0)};

  std::cout << "Memory: " << std::endl << std::to_string (memory);

  EXPECT_EQ (expected1, actual1);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapGenericApiTests_DotProduct, Test_2)
{
  oap::HostProcedures hp;
  oap::Memory memory = oap::host::NewMemoryWithValues ({1, 2}, 0.);

  oap::HostComplexMatrixUPtr output11 = oap::chost::NewReMatrixFromMemory (1, 1, memory, {0, 0});
  oap::HostComplexMatrixUPtr output12 = oap::chost::NewReMatrixFromMemory (1, 1, memory, {0, 1});

  oap::HostComplexMatrixUPtr matrix11 = oap::chost::NewReMatrixWithValue (1, 1, 4.);
  oap::HostComplexMatrixUPtr matrix12 = oap::chost::NewReMatrixWithValue (1, 1, 3.);

  oap::HostComplexMatrixUPtr matrix21 = oap::chost::NewReMatrixWithValue (1, 1, 4.);
  oap::HostComplexMatrixUPtr matrix22 = oap::chost::NewReMatrixWithValue (1, 1, 3.);

  std::vector<math::ComplexMatrix*> outputs = {output11, output12};
  std::vector<math::ComplexMatrix*> matrixs1 = {matrix11, matrix12};
  std::vector<math::ComplexMatrix*> matrixs2 = {matrix21, matrix22};
  hp.v2_multiply (outputs, matrixs1, matrixs2);

  std::vector<floatt> expected1 =
  {
    16,
  };

  std::vector<floatt> expected2 =
  {
    9,
  };

  std::vector<floatt> actual1 = {oap::common::GetValue (output11->re.mem, output11->re.reg, 0, 0)};
  std::vector<floatt> actual2 = {oap::common::GetValue (output12->re.mem, output12->re.reg, 0, 0)};

  std::cout << "Memory: " << std::endl << std::to_string (memory);

  EXPECT_EQ (expected1, actual1);
  EXPECT_EQ (expected2, actual2);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapGenericApiTests_DotProduct, Test_3)
{
  oap::HostProcedures hp;
  oap::Memory memory = oap::host::NewMemoryWithValues ({1, 2}, 0.);

  oap::HostComplexMatrixUPtr output11 = oap::chost::NewReMatrixFromMemory (1, 1, memory, {0, 0});
  oap::HostComplexMatrixUPtr output12 = oap::chost::NewReMatrixFromMemory (1, 1, memory, {0, 1});

  oap::HostComplexMatrixUPtr matrix11 = oap::chost::NewReMatrixWithValue (1, 1, 4.);
  oap::HostComplexMatrixUPtr matrix12 = oap::chost::NewReMatrixWithValue (1, 1, 3.);

  oap::HostComplexMatrixUPtr matrix21 = oap::chost::NewReMatrixWithValue (1, 1, 4.);
  oap::HostComplexMatrixUPtr matrix22 = oap::chost::NewReMatrixWithValue (1, 1, 3.);

  std::vector<math::ComplexMatrix*> outputs = {output11, output12};
  std::vector<math::ComplexMatrix*> matrixs1 = {matrix11, matrix12};
  std::vector<math::ComplexMatrix*> matrixs2 = {matrix21, matrix22};
  hp.v2_multiply (outputs, matrixs1, matrixs2);

  std::vector<floatt> expected1 =
  {
    16,
  };

  std::vector<floatt> expected2 =
  {
    9,
  };

  std::vector<floatt> actual1 = {oap::common::GetValue (output11->re.mem, output11->re.reg, 0, 0)};
  std::vector<floatt> actual2 = {oap::common::GetValue (output12->re.mem, output12->re.reg, 0, 0)};

  std::cout << "Memory: " << std::endl << std::to_string (memory);

  EXPECT_EQ (expected1, actual1);
  EXPECT_EQ (expected2, actual2);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapGenericApiTests_DotProduct, Test_4)
{
  oap::HostProcedures hp;

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
  std::vector<math::ComplexMatrix*> outputs = oap::chost::NewMatrices (minfo, params1_raw.size());
  std::vector<math::ComplexMatrix*> params1 = oap::chost::NewMatricesCopyOfArray (minfo1, params1_raw);
  std::vector<math::ComplexMatrix*> params2 = oap::chost::NewMatricesCopyOfArray (minfo, params2_raw);

  hp.v2_multiply (outputs, params1, params2);

  //PRINT_MATRIX_CARRAY(outputs);

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

  for (uintt idx = 0; idx < outputs.size(); ++idx)
  {
    oap::HostComplexMatrixPtr matrix = oap::chost::NewReMatrixCopyOfArray (1, 3, expected_raw[idx].data());
    EXPECT_THAT (matrix.get(), MatrixIsEqual (outputs[idx]));
  }

  oap::chost::deleteMatrices (outputs);
  oap::chost::deleteMatrices (params1);
  oap::chost::deleteMatrices (params2);
}

TEST_F(OapGenericApiTests_DotProduct, Test_5)
{
  oap::HostProcedures hp;

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
  std::vector<math::ComplexMatrix*> outputs = oap::chost::NewMatrices (minfo, params1_raw.size());
  std::vector<math::ComplexMatrix*> params1 = oap::chost::NewMatricesCopyOfArray (minfo1, params1_raw);
  std::vector<math::ComplexMatrix*> params2 = oap::chost::NewMatricesCopyOfArray (minfo, params2_raw);

  hp.v2_multiply (outputs, params1, params2);

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

  oap::chost::deleteMatrices (outputs);
  oap::chost::deleteMatrices (params1);
  oap::chost::deleteMatrices (params2);
}

TEST_F(OapGenericApiTests_DotProduct, Test_6)
{
  oap::HostProcedures hp;

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
    {-0.664504746, -0.269933422,  1.000000000, 0},
    {0.038228665, 0.031585489, 1.000000000, 0},
    {-0.451562665,  0.148885711,  1.000000000, 0},
    {0.565967395, 0.006534367, 1.000000000, 0},
    {0.262102990, 0.022468290, 1.000000000, 0},
    {0.629933906, 0.029740866, 1.000000000, 0},
    {-0.407287159,  0.549972349,  1.000000000, 0}
  };

  math::MatrixInfo minfo(true, false, 1, 4);
  math::MatrixDim mdim = {1, 3};
  math::MatrixInfo minfo1(true, false, 3, 3);
  std::vector<math::ComplexMatrix*> outputs = oap::chost::NewMatrices (minfo, params1_raw.size());
  std::vector<math::ComplexMatrix*> outputs_wb;
  for (auto* matrix : outputs)
  {
    outputs_wb.push_back (oap::chost::NewSharedSubMatrix (mdim, matrix));
  }
  std::vector<math::ComplexMatrix*> params1 = oap::chost::NewMatricesCopyOfArray (minfo1, params1_raw);
  std::vector<math::ComplexMatrix*> params2 = oap::chost::NewMatricesCopyOfArray (minfo, params2_raw);
  std::vector<math::ComplexMatrix*> params2_wb;
  for (auto* matrix : params2)
  {
    params2_wb.push_back (oap::chost::NewSharedSubMatrix (mdim, matrix));
  }

  hp.v2_multiply (outputs_wb, params1, params2_wb);

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

  oap::chost::deleteMatrices (outputs);
  oap::chost::deleteMatrices (outputs_wb);
  oap::chost::deleteMatrices (params1);
  oap::chost::deleteMatrices (params2);
  oap::chost::deleteMatrices (params2_wb);
}
