/*
 * Copyright 2016 - 2019 Marcin Matula
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

#include "HostProcedures.h"
#include "MatchersUtils.h"
#include "oapHostMatrixPtr.h"

class OapQRHTTests : public testing::Test {
 public:
  OapQRHTTests() {}

  virtual ~OapQRHTTests() {}

  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(OapQRHTTests, Test_1)
{
  HostProcedures hp;

  floatt h_expected_init[] =
  {
    -4.4529e-01, -1.8641e+00, -2.8109e+00, 7.2941e+00,
    8.0124e+00, 6.2898e+00, 1.2058e+01, -1.6088e+01,
    0.0000e+00, 4.0087e-01, 1.1545e+00, -3.3722e-01,
    0.0000e+00, 0.0000e+00, -1.5744e-01, 3.0010e+00,
  };

  floatt unwanted = h_expected_init[15];

  floatt h_expected_init_m_unwanted[] =
  {
    -4.4529e-01 - unwanted, -1.8641e+00, -2.8109e+00, 7.2941e+00,
    8.0124e+00, 6.2898e+00 - unwanted, 1.2058e+01, -1.6088e+01,
    0.0000e+00, 4.0087e-01, 1.1545e+00 - unwanted, -3.3722e-01,
    0.0000e+00, 0.0000e+00, -1.5744e-01, 3.0010e+00 - unwanted,
  };

  floatt r_expected[] =
  {
    8.7221,    3.7577,   12.1875,  -17.6610,
    0.0000,    0.5755,   -2.8519,   -0.4816,
    0.0000,    0.0000,   -0.2507,    0.0019,
    0.0000,    0.0000,    0.0000,   -0.0015,
  };

  floatt q_expected[] =
  {
    -0.3951,   -0.6591,   -0.4979,    0.4019,
    0.9186,   -0.2835,   -0.2142,    0.1728,
    0.0000,    0.6965,   -0.5584,    0.4506,
    0.0000,    0.0000,    0.6280,    0.7782,
  };

  oap::HostMatrixPtr Q = oap::host::NewReMatrix (4, 4);
  oap::HostMatrixPtr R = oap::host::NewReMatrix (4, 4);

  oap::HostMatrixPtr expectedR = oap::host::NewReMatrixCopyOfArray (4, 4, r_expected);
  oap::HostMatrixPtr expectedQ = oap::host::NewReMatrixCopyOfArray (4, 4, q_expected);

  oap::HostMatrixPtr A1 = oap::host::NewReMatrix (4, 4);
  hp.dotProduct (A1, expectedQ, expectedR);

  oap::HostMatrixPtr A = oap::host::NewReMatrixCopyOfArray (4, 4, h_expected_init_m_unwanted);

  PRINT_MATRIX(A.get());

  EXPECT_THAT (A1.get(), MatrixIsEqual(A.get(), 0.001)); 

  oap::HostMatrixPtr V = oap::host::NewReMatrix (1, 4);

  oap::HostMatrixPtr VT = oap::host::NewReMatrix (4, 1);
  oap::HostMatrixPtr VVT = oap::host::NewReMatrix (4, 4);
  oap::HostMatrixPtr P = oap::host::NewReMatrix (4, 4);

  hp.QRHT (Q, R, A, V, VT, P, VVT);

  oap::HostMatrixPtr A2 = oap::host::NewReMatrix (4, 4);
  hp.dotProduct (A2, Q, R);
  EXPECT_THAT (A2.get(), MatrixIsEqual(A.get(), 0.001)); 

#if 0 
  EXPECT_THAT (expectedR.get(), MatrixIsEqual(R.get())); 
  EXPECT_THAT (expectedQ.get(), MatrixIsEqual(Q.get())); 
#endif

  PRINT_MATRIX(A.get());
  EXPECT_THAT (Q.get(), MatrixIsOrthogonal (hp)); 
  EXPECT_THAT (R.get(), MatrixIsUpperTriangular ()); 
}

TEST_F(OapQRHTTests, Test_2)
{
  HostProcedures hp;

  floatt h_init[] =
  {
    2, -2, 18,
    2, 1, 0,
    1, 2, 0
  };

  floatt q_expected[] =
  {
    -2./3., 2./3., -1./3.,
    -2./3., -1./3., 2./3.,
    -1./3., -2./3., -2./3.
  };

  floatt r_expected[] =
  {
    -3, 0, -12,
    0, -3, 12,
    0, 0, -6
  };

  oap::HostMatrixPtr Q = oap::host::NewReMatrix (3, 3);
  oap::HostMatrixPtr R = oap::host::NewReMatrix (3, 3);

  oap::HostMatrixPtr expectedQ = oap::host::NewReMatrixCopyOfArray (3, 3, q_expected);
  oap::HostMatrixPtr expectedR = oap::host::NewReMatrixCopyOfArray (3, 3, r_expected);

  oap::HostMatrixPtr A = oap::host::NewReMatrixCopyOfArray (3, 3, h_init);

  oap::HostMatrixPtr V = oap::host::NewReMatrix (1, 3);
  PRINT_MATRIX(V.get());

  oap::HostMatrixPtr VT = oap::host::NewReMatrix (3, 1);
  oap::HostMatrixPtr VVT = oap::host::NewReMatrix (3, 3);
  oap::HostMatrixPtr P = oap::host::NewReMatrix (3, 3);

  PRINT_MATRIX(A.get());
  hp.QRHT (Q, R, A, V, VT, P, VVT);

#if 0 
  EXPECT_THAT (expectedR.get(), MatrixIsEqual (R.get())); 
  EXPECT_THAT (expectedQ.get(), MatrixIsEqual (Q.get())); 
#endif

  EXPECT_THAT (Q.get(), MatrixIsOrthogonal (hp)); 
  EXPECT_THAT (R.get(), MatrixIsUpperTriangular ()); 
  PRINT_MATRIX(A.get());
}

