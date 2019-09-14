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

#include <string>
#include "gtest/gtest.h"
#include "KernelExecutor.h"

#include "oapCuHArnoldiS.h"
#include "oapGenericArnoldiApi.h"
#include "oapCudaMatrixUtils.h"
#include "oapHostMatrixUtils.h"

#include "oapDeviceMatrixPtr.h"

#include "MatchersUtils.h"

#include "CuProceduresApi.h"

class OapGenericArnoldiApiTests : public testing::Test {
 public:
  CUresult status;

  virtual void SetUp()
  {
    oap::cuda::Context::Instance().create();
  }

  virtual void TearDown()
  {
    oap::cuda::Context::Instance().destroy();
  }
};

TEST_F(OapGenericArnoldiApiTests, QR_Test_1)
{ 
  floatt h_expected_init[] =
  {
    -4.4529e-01, -1.8641e+00, -2.8109e+00, 7.2941e+00,
    8.0124e+00, 6.2898e+00, 1.2058e+01, -1.6088e+01,
    0.0000e+00, 4.0087e-01, 1.1545e+00, -3.3722e-01,
    0.0000e+00, 0.0000e+00, -1.5744e-01, 3.0010e+00,
  };

  floatt h_expected_1[] =
  {
    3.0067e+00, 1.6742e+00, -2.3047e+01, -4.0863e+00,
    5.2870e-01, 8.5146e-01,  1.1660e+00, -1.5609e+00,
    0,          -1.7450e-01, 3.1421e+00, -1.1140e-01,
    0,          0,          -1.0210e-03, 2.9998e+00,
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

  floatt h_expected_1_m_unwanted[] =
  {
    0.005672933,   1.674653622, -23.044559922,  -4.096797564,
    0.528687072,  -2.149635203,   1.166660659,  -1.560477982,
    0.000000000,  -0.174611528,   0.141164119,  -0.111489468,
    0.000000000,   0.000000000,  -0.000961898,  -0.001191852,
  };

  floatt h_expected_1_m_unwanted_p_unwanted [16];

  floatt unwanted = h_expected_init[15];

  floatt h_expected_init_m_unwanted[] =
  {
    -4.4529e-01 - unwanted, -1.8641e+00, -2.8109e+00, 7.2941e+00,
    8.0124e+00, 6.2898e+00 - unwanted, 1.2058e+01, -1.6088e+01,
    0.0000e+00, 4.0087e-01, 1.1545e+00 - unwanted, -3.3722e-01,
    0.0000e+00, 0.0000e+00, -1.5744e-01, 3.0010e+00 - unwanted,
  };

  memcpy (h_expected_1_m_unwanted_p_unwanted, h_expected_1_m_unwanted, 16 * sizeof (floatt));
  h_expected_1_m_unwanted_p_unwanted[0] = h_expected_1_m_unwanted[0] + unwanted;
  h_expected_1_m_unwanted_p_unwanted[5] = h_expected_1_m_unwanted[5] + unwanted;
  h_expected_1_m_unwanted_p_unwanted[10] = h_expected_1_m_unwanted[10] + unwanted;
  h_expected_1_m_unwanted_p_unwanted[15] = h_expected_1_m_unwanted[15] + unwanted;

  uint length = sizeof(h_expected_init) / sizeof(h_expected_init[0]);

  oap::generic::CuHArnoldiS ca;

  math::MatrixInfo matrixInfo (true, false, 4, 4);

  oap::HostMatrixPtr hexpectedInit = oap::host::NewReMatrixCopyOfArray (4, 4, h_expected_init);
  oap::HostMatrixPtr hexpectedInitMUnwanted = oap::host::NewReMatrixCopyOfArray (4, 4, h_expected_init_m_unwanted);

  PRINT_MATRIX(hexpectedInitMUnwanted.get());

  oap::HostMatrixPtr hexpected1 = oap::host::NewReMatrixCopyOfArray (4, 4, h_expected_1);
  oap::HostMatrixPtr hexpected1MUnwanted = oap::host::NewReMatrixCopyOfArray (4, 4, h_expected_1_m_unwanted);
  oap::HostMatrixPtr hexpected1MUnwantedPUnwanted = oap::host::NewReMatrixCopyOfArray (4, 4, h_expected_1_m_unwanted_p_unwanted);

  oap::HostMatrixPtr qexpected = oap::host::NewReMatrixCopyOfArray (4, 4, q_expected);

  oap::HostMatrixPtr rexpected = oap::host::NewReMatrixCopyOfArray (4, 4, r_expected);

  oap::generic::allocStage1 (ca, matrixInfo, oap::cuda::NewKernelMatrix);
  oap::generic::allocStage2 (ca, matrixInfo, 4, oap::cuda::NewKernelMatrix, oap::host::NewHostMatrix);
  oap::generic::allocStage3 (ca, matrixInfo, 4, oap::cuda::NewKernelMatrix);

  oap::cuda::CopyHostArrayToDeviceReMatrix (ca.m_H, h_expected_init, length);
  ca.m_unwanted.push_back (unwanted);

  oap::CuProceduresApi cuApi;

  oap::generic::shiftedQRIteration (ca, cuApi, 0);

  {
    using namespace oap::generic::iram_shiftedQRIteration;
    EXPECT_THAT (qexpected.get(), oap::cuda::MatrixIsEqualHK (getQ(ca)));
    EXPECT_THAT (rexpected.get(), oap::cuda::MatrixIsEqualHK (getR(ca)));
  }

  {
    using namespace oap::generic::iram_shiftedQRIteration;
    cuApi.dotProduct (ca.m_H, getQ (ca), getR (ca));

    EXPECT_THAT (hexpectedInitMUnwanted.get(), oap::cuda::MatrixIsEqualHK (ca.m_H));

    cuApi.setDiagonal (ca.m_I, unwanted, 0);
    cuApi.add (ca.m_H, ca.m_H, ca.m_I);

    EXPECT_THAT (hexpectedInit.get(), oap::cuda::MatrixIsEqualHK (ca.m_H));
    PRINT_CUMATRIX(ca.m_H);
  }

  {
    using namespace oap::generic::iram_shiftedQRIteration;
    cuApi.dotProduct (ca.m_H, getR (ca), getQ (ca));
  }

  EXPECT_THAT (hexpected1MUnwanted.get(), oap::cuda::MatrixIsEqualHK (ca.m_H));

  cuApi.setDiagonal (ca.m_I, unwanted, 0);
  cuApi.add (ca.m_H, ca.m_H, ca.m_I);


  EXPECT_THAT (hexpected1.get(), oap::cuda::MatrixIsEqualHK (ca.m_H));
  EXPECT_THAT (hexpected1MUnwantedPUnwanted.get(), oap::cuda::MatrixIsEqualHK (ca.m_H));
  EXPECT_THAT (hexpected1MUnwantedPUnwanted.get(), MatrixIsEqual (hexpected1.get(), 0.01));

  oap::generic::deallocStage1 (ca, oap::cuda::DeleteDeviceMatrix);
  oap::generic::deallocStage2 (ca, oap::cuda::DeleteDeviceMatrix, oap::host::DeleteMatrix);
  oap::generic::deallocStage3 (ca, oap::cuda::DeleteDeviceMatrix);
}

TEST_F(OapGenericArnoldiApiTests, QR_Test_2)
{ 
  floatt values[] =
  {
    6, 5, 0,
    5, 1, 4,
    0, 4, 3
  };

  floatt q_expected[] =
  {
    0.7682, 0.3327, 0.547,
    0.6402, -0.3992, -0.6564,
    0, 0.8544, -0.5196
  };

  floatt r_expected[] =
  {
    7.8102, 4.4813, 2.5607,
    0,      4.6817, 0.9664,
    0,      0,      -4.1843
  };

  uint length = sizeof(values) / sizeof(values[0]);

  oap::DeviceMatrixPtr H = oap::cuda::NewDeviceReMatrix (3, 3);
  oap::DeviceMatrixPtr R = oap::cuda::NewDeviceReMatrix (3, 3);
  oap::DeviceMatrixPtr Q = oap::cuda::NewDeviceReMatrix (3, 3);
  oap::DeviceMatrixPtr aux1 = oap::cuda::NewDeviceReMatrix (1, 3);
  oap::DeviceMatrixPtr aux2 = oap::cuda::NewDeviceReMatrix (3, 1);
  oap::DeviceMatrixPtr aux3 = oap::cuda::NewDeviceReMatrix (3, 3);
  oap::DeviceMatrixPtr aux4 = oap::cuda::NewDeviceReMatrix (3, 3);
  oap::DeviceMatrixPtr aux5 = oap::cuda::NewDeviceReMatrix (3, 3);

  oap::HostMatrixPtr qexpected = oap::host::NewReMatrix (3, 3);
  oap::host::CopyArrayToReMatrix (qexpected, q_expected);

  oap::HostMatrixPtr rexpected = oap::host::NewReMatrix (3, 3);
  oap::host::CopyArrayToReMatrix (rexpected, r_expected);

  oap::cuda::CopyHostArrayToDeviceReMatrix (H, values, length);

  oap::CuProceduresApi cuApi;
  cuApi.QRHT (Q, R, H, aux1, aux2, aux3, aux4, aux5);

  EXPECT_THAT (qexpected.get(), oap::cuda::MatrixIsEqualHK (Q.get()));
  EXPECT_THAT (rexpected.get(), oap::cuda::MatrixIsEqualHK (R.get()));
}

TEST_F(OapGenericArnoldiApiTests, QR_Test_3)
{ 
  floatt values[] =
  {
    7, 2,
    2, 4,
  };

  floatt q_expected[] =
  {
    0.962, -.275,
    .275, .962
  };

  floatt r_expected[] =
  {
    7.28, 3.02,
    0,    3.3
  };

  floatt h_expected[] =
  {
    7.83, 0.906,
    0.906, 3.17
  };


  uint length = sizeof(values) / sizeof(values[0]);

  oap::DeviceMatrixPtr H = oap::cuda::NewDeviceReMatrix (2, 2);
  oap::DeviceMatrixPtr R = oap::cuda::NewDeviceReMatrix (2, 2);
  oap::DeviceMatrixPtr Q = oap::cuda::NewDeviceReMatrix (2, 2);
  oap::DeviceMatrixPtr aux1 = oap::cuda::NewDeviceReMatrix (1, 2);
  oap::DeviceMatrixPtr aux2 = oap::cuda::NewDeviceReMatrix (2, 1);
  oap::DeviceMatrixPtr aux3 = oap::cuda::NewDeviceReMatrix (2, 2);
  oap::DeviceMatrixPtr aux4 = oap::cuda::NewDeviceReMatrix (2, 2);
  oap::DeviceMatrixPtr aux5 = oap::cuda::NewDeviceReMatrix (2, 2);

  oap::HostMatrixPtr qexpected = oap::host::NewReMatrix (2, 2);
  oap::host::CopyArrayToReMatrix (qexpected, q_expected);

  oap::HostMatrixPtr rexpected = oap::host::NewReMatrix (2, 2);
  oap::host::CopyArrayToReMatrix (rexpected, r_expected);

  oap::HostMatrixPtr hexpected = oap::host::NewReMatrix (2, 2);
  oap::host::CopyArrayToReMatrix (hexpected, h_expected);

  oap::cuda::CopyHostArrayToDeviceReMatrix (H, values, length);

  oap::CuProceduresApi cuApi;
  cuApi.QRHT (Q, R, H, aux1, aux2, aux3, aux4, aux5);
  cuApi.dotProduct (H, R, Q);

  EXPECT_THAT (qexpected.get(), oap::cuda::MatrixIsEqualHK (Q.get(), 0.01));
  EXPECT_THAT (rexpected.get(), oap::cuda::MatrixIsEqualHK (R.get(), 0.01));
  EXPECT_THAT (hexpected.get(), oap::cuda::MatrixIsEqualHK (H.get(), 0.01));
}
