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
#include "gmock/gmock.h"

#include "HostProcedures.hpp"
#include "MatchersUtils.hpp"
#include "oapHostComplexMatrixPtr.hpp"

class OapQRHTTests : public testing::Test {
 public:
  OapQRHTTests() {}

  virtual ~OapQRHTTests() {}

  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(OapQRHTTests, Test_1)
{
  oap::HostProcedures hp;

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

  oap::HostComplexMatrixPtr Q = oap::chost::NewReMatrix (4, 4);
  oap::HostComplexMatrixPtr R = oap::chost::NewReMatrix (4, 4);

  oap::HostComplexMatrixPtr expectedR = oap::chost::NewReMatrixCopyOfArray (4, 4, r_expected);
  oap::HostComplexMatrixPtr expectedQ = oap::chost::NewReMatrixCopyOfArray (4, 4, q_expected);

  oap::HostComplexMatrixPtr A1 = oap::chost::NewReMatrix (4, 4);
  hp.dotProduct (A1, expectedQ, expectedR);

  oap::HostComplexMatrixPtr A = oap::chost::NewReMatrixCopyOfArray (4, 4, h_expected_init_m_unwanted);

  EXPECT_THAT (A1.get(), MatrixIsEqual(A.get(), 0.001)); 

  oap::HostComplexMatrixPtr V = oap::chost::NewReMatrix (1, 4);

  oap::HostComplexMatrixPtr VT = oap::chost::NewReMatrix (4, 1);
  oap::HostComplexMatrixPtr VVT = oap::chost::NewReMatrix (4, 4);
  oap::HostComplexMatrixPtr P = oap::chost::NewReMatrix (4, 4);

  hp.QRHT (Q, R, A, V, VT, P, VVT);

  oap::HostComplexMatrixPtr A2 = oap::chost::NewReMatrix (4, 4);
  hp.dotProduct (A2, Q, R);
  EXPECT_THAT (A2.get(), MatrixIsEqual(A.get(), 0.001)); 

#if 0 
  EXPECT_THAT (expectedR.get(), MatrixIsEqual(R.get())); 
  EXPECT_THAT (expectedQ.get(), MatrixIsEqual(Q.get())); 
#endif

  EXPECT_THAT (Q.get(), MatrixIsOrthogonal (hp)); 
  EXPECT_THAT (R.get(), MatrixIsUpperTriangular ()); 
}

TEST_F(OapQRHTTests, Test_2)
{
  oap::HostProcedures hp;

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

  oap::HostComplexMatrixPtr Q = oap::chost::NewReMatrix (3, 3);
  oap::HostComplexMatrixPtr R = oap::chost::NewReMatrix (3, 3);

  oap::HostComplexMatrixPtr expectedQ = oap::chost::NewReMatrixCopyOfArray (3, 3, q_expected);
  oap::HostComplexMatrixPtr expectedR = oap::chost::NewReMatrixCopyOfArray (3, 3, r_expected);

  oap::HostComplexMatrixPtr A = oap::chost::NewReMatrixCopyOfArray (3, 3, h_init);

  oap::HostComplexMatrixPtr V = oap::chost::NewReMatrix (1, 3);

  oap::HostComplexMatrixPtr VT = oap::chost::NewReMatrix (3, 1);
  oap::HostComplexMatrixPtr VVT = oap::chost::NewReMatrix (3, 3);
  oap::HostComplexMatrixPtr P = oap::chost::NewReMatrix (3, 3);

  hp.QRHT (Q, R, A, V, VT, P, VVT);

#if 0 
  EXPECT_THAT (expectedR.get(), MatrixIsEqual (R.get())); 
  EXPECT_THAT (expectedQ.get(), MatrixIsEqual (Q.get())); 
#endif

  EXPECT_THAT (Q.get(), MatrixIsOrthogonal (hp)); 
  EXPECT_THAT (R.get(), MatrixIsUpperTriangular ()); 
}

