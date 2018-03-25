/*
 * Copyright 2016, 2017 Marcin Matula
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
#include "CuProceduresApi.h"
#include "MathOperationsCpu.h"
#include "oapHostMatrixUtils.h"
#include "oapCudaMatrixUtils.h"
#include "KernelExecutor.h"
#include "matrix6.h"

class OapMatrixCudaTests : public testing::Test {
 public:
  math::Matrix* output;
  math::Matrix* eq_output;
  oap::CuProceduresApi* cuMatrix;
  CUresult status;

  virtual void SetUp() {
    status = CUDA_SUCCESS;
    oap::cuda::Context::Instance().create();
    output = NULL;
    eq_output = NULL;
    cuMatrix = new oap::CuProceduresApi();
  }

  virtual void TearDown() {
    delete cuMatrix;
    if (output != NULL && eq_output != NULL) {
      EXPECT_THAT(output, MatrixIsEqual(eq_output));
    }
    EXPECT_EQ(status, CUDA_SUCCESS);
    oap::host::DeleteMatrix(output);
    oap::host::DeleteMatrix(eq_output);
    oap::cuda::Context::Instance().destroy();
  }
};

TEST_F(OapMatrixCudaTests, SetVectorTest) {
  floatt hArray[] = {
      5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  floatt hOutputArray[] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  floatt hVArray[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };

  output = oap::host::NewReMatrixCopy(10, 10, hArray);
  math::Matrix* V = oap::cuda::NewDeviceMatrix(output, 10, 10);
  math::Matrix* v = oap::cuda::NewDeviceMatrix(output, 1, 10);
  oap::cuda::CopyHostArraysToDeviceMatrix(v, hVArray, NULL);
  oap::cuda::CopyHostMatrixToDeviceMatrix(V, output);

  cuMatrix->setVector(V, 0, v, 10);
  oap::cuda::CopyDeviceMatrixToHostMatrix(output, V);

  oap::cuda::DeleteDeviceMatrix(V);
  oap::cuda::DeleteDeviceMatrix(v);

  eq_output = oap::host::NewReMatrixCopy(10, 10, hOutputArray);
}

TEST_F(OapMatrixCudaTests, SetVectorTest1) {
  floatt hArray[] = {
      5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  floatt hOutputArray[] = {
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  floatt hVArray[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };

  output = oap::host::NewReMatrixCopy(10, 10, hArray);
  math::Matrix* V = oap::cuda::NewDeviceMatrix(output, 10, 10);
  math::Matrix* v = oap::cuda::NewDeviceMatrix(output, 2, 10);
  oap::cuda::CopyHostArraysToDeviceMatrix(v, hVArray, NULL);
  oap::cuda::CopyHostMatrixToDeviceMatrix(V, output);

  cuMatrix->setVector(V, 0, v, 10);
  oap::cuda::CopyDeviceMatrixToHostMatrix(output, V);

  oap::cuda::DeleteDeviceMatrix(V);
  oap::cuda::DeleteDeviceMatrix(v);

  eq_output = oap::host::NewReMatrixCopy(10, 10, hOutputArray);
}

TEST_F(OapMatrixCudaTests, GetVectorTest) {
  floatt hArray[] = {
      5, 5, 5, 5, 5, 0, 0, 0, 0, 0,
  };

  floatt hVArray[] = {
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  floatt hOutputArray[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };

  output = oap::host::NewReMatrixCopy(1, 10, hArray);
  math::Matrix* V = oap::cuda::NewDeviceMatrix(output, 10, 10);
  math::Matrix* v = oap::cuda::NewDeviceMatrix(output, 1, 10);
  oap::cuda::CopyHostArraysToDeviceMatrix(v, hArray, NULL);
  oap::cuda::CopyHostArraysToDeviceMatrix(V, hVArray, NULL);

  cuMatrix->getVector(v, 10, V, 0);
  oap::cuda::CopyDeviceMatrixToHostMatrix(output, v);

  oap::cuda::DeleteDeviceMatrix(V);
  oap::cuda::DeleteDeviceMatrix(v);

  eq_output = oap::host::NewReMatrixCopy(1, 10, hOutputArray);
}

TEST_F(OapMatrixCudaTests, GetVectorTest1) {
  floatt hArray[] = {
      5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  floatt hVArray[] = {
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  floatt hOutputArray[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };

  output = oap::host::NewReMatrixCopy(2, 10, hArray);
  math::Matrix* V = oap::cuda::NewDeviceMatrix(output, 10, 10);
  math::Matrix* v = oap::cuda::NewDeviceMatrix(output, 2, 10);
  oap::cuda::CopyHostArraysToDeviceMatrix(v, hArray, NULL);
  oap::cuda::CopyHostArraysToDeviceMatrix(V, hVArray, NULL);

  cuMatrix->getVector(v, 10, V, 0);
  oap::cuda::CopyDeviceMatrixToHostMatrix(output, v);

  oap::cuda::DeleteDeviceMatrix(V);
  oap::cuda::DeleteDeviceMatrix(v);

  eq_output = oap::host::NewReMatrixCopy(2, 10, hOutputArray);
}

TEST_F(OapMatrixCudaTests, SetIdentityReMatrixTest) {
  floatt hArray[] = {
      5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  floatt hOutputArray[] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
  };

  output = oap::host::NewReMatrixCopy(10, 10, hArray);
  math::Matrix* matrix = oap::cuda::NewDeviceMatrix(output, 10, 10);
  oap::cuda::CopyHostArraysToDeviceMatrix(matrix, hArray, NULL);
  cuMatrix->setIdentity(matrix);
  oap::cuda::CopyDeviceMatrixToHostMatrix(output, matrix);

  oap::cuda::DeleteDeviceMatrix(matrix);

  eq_output = oap::host::NewReMatrixCopy(10, 10, hOutputArray);
}

TEST_F(OapMatrixCudaTests, SetDiagonalReMatrixTest) {
  floatt hArray[] = {
      5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  floatt hOutputArray[] = {
      2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
  };

  output = oap::host::NewReMatrixCopy(10, 10, hArray);
  math::Matrix* matrix = oap::cuda::NewDeviceMatrix(output, 10, 10);
  oap::cuda::CopyHostArraysToDeviceMatrix(matrix, hArray, NULL);
  cuMatrix->setDiagonal(matrix, 2, 0);
  oap::cuda::CopyDeviceMatrixToHostMatrix(output, matrix);

  oap::cuda::DeleteDeviceMatrix(matrix);

  eq_output = oap::host::NewReMatrixCopy(10, 10, hOutputArray);
}

TEST_F(OapMatrixCudaTests, MultiplyConstantReMatrixTest) {
  floatt hArray[] = {
      1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
  };

  floatt hOutputArray[] = {
      5, 0, 10, 0, 0,  0, 0,  0, 0,  0, 0, 5, 0, 10, 0, 0,  0, 0,  0, 0,
      0, 0, 5,  0, 10, 0, 0,  0, 0,  0, 0, 0, 0, 5,  0, 10, 0, 0,  0, 0,
      0, 0, 0,  0, 5,  0, 10, 0, 0,  0, 0, 0, 0, 0,  0, 5,  0, 10, 0, 0,
      0, 0, 0,  0, 0,  0, 5,  0, 10, 0, 0, 0, 0, 0,  0, 0,  0, 5,  0, 10,
      0, 0, 0,  0, 0,  0, 0,  0, 5,  0, 0, 0, 0, 0,  0, 0,  0, 0,  0, 5,
  };

  output = oap::host::NewReMatrixCopy(10, 10, hArray);
  math::Matrix* doutput = oap::cuda::NewDeviceMatrix(output, 10, 10);
  math::Matrix* dparam0 = oap::cuda::NewDeviceMatrix(output, 10, 10);
  oap::cuda::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);

  cuMatrix->multiplyReConstant(doutput, dparam0, 5);

  oap::cuda::CopyDeviceMatrixToHostMatrix(output, doutput);
  eq_output = oap::host::NewReMatrixCopy(10, 10, hOutputArray);
}

TEST_F(OapMatrixCudaTests, TransponseReMatrixExTest1) {
  floatt hArray[] = {
      1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
  };

  floatt hOutputArray[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  output = oap::host::NewReMatrixCopy(10, 4, hArray);
  math::Matrix* doutput = oap::cuda::NewDeviceMatrix(output, 10, 4);
  math::Matrix* dparam0 = oap::cuda::NewDeviceMatrix(output, 4, 10);
  oap::cuda::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);

  MatrixEx* matrixEx = oap::cuda::NewDeviceMatrixEx();
  MatrixEx hMatrixEx = {0, 10, 0, 2, 0, 0};
  oap::cuda::SetMatrixEx(matrixEx, &hMatrixEx);

  cuMatrix->transposeEx(doutput, dparam0, matrixEx);

  oap::cuda::DeleteDeviceMatrixEx(matrixEx);
  oap::cuda::CopyDeviceMatrixToHostMatrix(output, doutput);
  eq_output = oap::host::NewReMatrixCopy(10, 4, hOutputArray);
}

TEST_F(OapMatrixCudaTests, TransponseReMatrixExTest2) {
  floatt hArray[] = {
      1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  floatt hOutputArray[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  output = oap::host::NewReMatrixCopy(10, 10, hArray);
  math::Matrix* doutput = oap::cuda::NewDeviceMatrix(output, 10, 10);
  math::Matrix* dparam0 = oap::cuda::NewDeviceMatrix(output, 10, 10);
  oap::cuda::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);

  MatrixEx* matrixEx = oap::cuda::NewDeviceMatrixEx();
  MatrixEx hMatrixEx = {0, 10, 0, 2, 0, 0};
  oap::cuda::SetMatrixEx(matrixEx, &hMatrixEx);

  cuMatrix->transposeEx(doutput, dparam0, matrixEx);

  oap::cuda::DeleteDeviceMatrixEx(matrixEx);
  oap::cuda::CopyDeviceMatrixToHostMatrix(output, doutput);
  eq_output = oap::host::NewReMatrixCopy(10, 10, hOutputArray);
}

TEST_F(OapMatrixCudaTests, TransponseReMatrixExTest3) {
  floatt hArray[] = {
      1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  floatt hOutputArray[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  output = oap::host::NewReMatrixCopy(10, 4, hArray);
  math::Matrix* doutput = oap::cuda::NewDeviceMatrix(output, 10, 4);
  math::Matrix* dparam0 = oap::cuda::NewDeviceMatrix(output, 10, 10);
  oap::cuda::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);

  MatrixEx* matrixEx = oap::cuda::NewDeviceMatrixEx();
  MatrixEx hMatrixEx = {0, 10, 0, 2, 0, 0};
  oap::cuda::SetMatrixEx(matrixEx, &hMatrixEx);

  cuMatrix->transposeEx(doutput, dparam0, matrixEx);

  oap::cuda::DeleteDeviceMatrixEx(matrixEx);
  oap::cuda::CopyDeviceMatrixToHostMatrix(output, doutput);
  eq_output = oap::host::NewReMatrixCopy(10, 4, hOutputArray);
}

TEST_F(OapMatrixCudaTests, MatrixExTest) {
  MatrixEx** dMatrixExs = oap::cuda::NewDeviceMatrixEx(5);
  uintt buffer[] = {0, 10, 0, 1, 0,  0,  0,  1, 0,  15, 0, 20, 0, 0, 0,
                    0, 0,  0, 0, 25, 30, 35, 0, 40, 0,  1, 0,  2, 3, 5};
  oap::cuda::SetMatrixEx(dMatrixExs, buffer, 5);
  MatrixEx matrixEx;

  CudaUtils::CopyDeviceToHost(&matrixEx, dMatrixExs[0], sizeof(MatrixEx));
  EXPECT_THAT(matrixEx, MatrixExIsEqual(buffer));

  CudaUtils::CopyDeviceToHost(&matrixEx, dMatrixExs[1], sizeof(MatrixEx));
  EXPECT_THAT(matrixEx, MatrixExIsEqual(&buffer[6]));

  CudaUtils::CopyDeviceToHost(&matrixEx, dMatrixExs[2], sizeof(MatrixEx));
  EXPECT_THAT(matrixEx, MatrixExIsEqual(&buffer[12]));

  CudaUtils::CopyDeviceToHost(&matrixEx, dMatrixExs[3], sizeof(MatrixEx));
  EXPECT_THAT(matrixEx, MatrixExIsEqual(&buffer[18]));

  CudaUtils::CopyDeviceToHost(&matrixEx, dMatrixExs[4], sizeof(MatrixEx));
  EXPECT_THAT(matrixEx, MatrixExIsEqual(&buffer[24]));
}

TEST_F(OapMatrixCudaTests, MagnitudeReMatrixTest) {
  floatt hArray[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };

  math::MathOperationsCpu mocpu;

  math::Matrix* matrix = oap::host::NewMatrixCopy<floatt>(1, 10, hArray, NULL);
  math::Matrix* dparam0 = oap::cuda::NewDeviceMatrix(true, false, 1, 10);
  oap::cuda::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);
  floatt output;
  floatt doutput;
  floatt doutput1;
  floatt doutput2;
  cuMatrix->magnitude(doutput, dparam0);
  cuMatrix->magnitudeOpt(doutput1, dparam0);
  cuMatrix->magnitudeOptVer2(doutput2, dparam0);

  mocpu.magnitude(&output, matrix);

  oap::host::DeleteMatrix(matrix);
  oap::cuda::DeleteDeviceMatrix(dparam0);
  EXPECT_DOUBLE_EQ(doutput, output);
  EXPECT_DOUBLE_EQ(doutput1, output);
  EXPECT_DOUBLE_EQ(doutput2, output);
}

TEST_F(OapMatrixCudaTests, MagnitudeReMatrixTest1) {
  floatt hArray[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  };

  math::MathOperationsCpu mocpu;

  math::Matrix* matrix = oap::host::NewMatrixCopy<floatt>(1, 10, hArray, NULL);
  math::Matrix* dparam0 = oap::cuda::NewDeviceMatrix(true, false, 1, 10);
  oap::cuda::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);
  floatt doutput;
  floatt doutput1;
  floatt doutput2;
  cuMatrix->magnitude(doutput, dparam0);
  cuMatrix->magnitudeOpt(doutput1, dparam0);
  cuMatrix->magnitudeOptVer2(doutput2, dparam0);

  floatt output;
  mocpu.magnitude(&output, matrix);

  oap::host::DeleteMatrix(matrix);
  oap::cuda::DeleteDeviceMatrix(dparam0);
  EXPECT_DOUBLE_EQ(output, doutput);
  EXPECT_DOUBLE_EQ(output, doutput1);
  EXPECT_DOUBLE_EQ(output, doutput2);
}

TEST_F(OapMatrixCudaTests, MagnitudeReMatrixTest2) {
  floatt hArray[] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  math::MathOperationsCpu mocpu;

  math::Matrix* matrix = oap::host::NewMatrixCopy<floatt>(1, 10, hArray, NULL);
  math::Matrix* dparam0 = oap::cuda::NewDeviceMatrix(true, false, 1, 10);
  oap::cuda::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);
  floatt doutput;
  floatt doutput1;
  floatt doutput2;
  cuMatrix->magnitude(doutput, dparam0);
  cuMatrix->magnitudeOpt(doutput1, dparam0);
  cuMatrix->magnitudeOptVer2(doutput2, dparam0);

  floatt output;
  mocpu.magnitude(&output, matrix);

  oap::host::DeleteMatrix(matrix);
  oap::cuda::DeleteDeviceMatrix(dparam0);
  EXPECT_DOUBLE_EQ(0, doutput);
  EXPECT_DOUBLE_EQ(output, doutput);
  EXPECT_DOUBLE_EQ(output, doutput1);
  EXPECT_DOUBLE_EQ(output, doutput2);
}

TEST_F(OapMatrixCudaTests, MagnitudeReMatrixBigDataTest) {
  size_t length = 16001;

  floatt* hArray = new floatt[length];
  memset(hArray, 0, sizeof(floatt) * length);
  hArray[4] = 3;

  math::MathOperationsCpu mocpu;

  math::Matrix* matrix = oap::host::NewMatrixCopy<floatt>(1, length, hArray, NULL);
  math::Matrix* dparam0 = oap::cuda::NewDeviceMatrix(true, false, 1, length);
  oap::cuda::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);
  floatt doutput;
  floatt doutput1;
  floatt doutput2;
  cuMatrix->magnitude(doutput, dparam0);
  cuMatrix->magnitudeOpt(doutput1, dparam0);
  cuMatrix->magnitudeOptVer2(doutput2, dparam0);

  floatt output;
  mocpu.magnitude(&output, matrix);

  oap::host::DeleteMatrix(matrix);
  oap::cuda::DeleteDeviceMatrix(dparam0);
  EXPECT_DOUBLE_EQ(3, doutput);
  EXPECT_DOUBLE_EQ(output, doutput);
  EXPECT_DOUBLE_EQ(output, doutput1);
  EXPECT_DOUBLE_EQ(output, doutput2);
  delete[] hArray;
}

TEST_F(OapMatrixCudaTests, MagnitudeRealMatrixBigDataTest) {
  size_t length = 16001;

  floatt* hArray = new floatt[length];
  floatt* hArray1 = new floatt[length];
  memset(hArray, 0, sizeof(floatt) * length);
  memset(hArray1, 0, sizeof(floatt) * length);
  hArray[4] = 3;

  math::MathOperationsCpu mocpu;

  math::Matrix* matrix = oap::host::NewMatrixCopy<floatt>(1, length, hArray, hArray1);
  math::Matrix* dparam0 = oap::cuda::NewDeviceMatrix(true, true, 1, length);
  oap::cuda::CopyHostArraysToDeviceMatrix(dparam0, hArray, hArray1);
  floatt doutput = 100;
  floatt doutput1 = 100;
  floatt doutput2 = 100;
  cuMatrix->magnitude(doutput, dparam0);
  cuMatrix->magnitudeOpt(doutput1, dparam0);
  cuMatrix->magnitudeOptVer2(doutput2, dparam0);

  floatt output;
  mocpu.magnitude(&output, matrix);

  oap::host::DeleteMatrix(matrix);
  oap::cuda::DeleteDeviceMatrix(dparam0);
  EXPECT_DOUBLE_EQ(3, doutput);
  EXPECT_DOUBLE_EQ(output, doutput);
  EXPECT_DOUBLE_EQ(output, doutput1);
  EXPECT_DOUBLE_EQ(output, doutput2);
  delete[] hArray;
  delete[] hArray1;
}

TEST_F(OapMatrixCudaTests, MagnitudeRealMatrixBigDataTest1) {
  size_t length = 16384;

  floatt* hArray = new floatt[length];
  floatt* hArray1 = new floatt[length];
  memset(hArray, 0, sizeof(floatt) * length);
  memset(hArray1, 0, sizeof(floatt) * length);

  math::MathOperationsCpu mocpu;

  math::Matrix* matrix = oap::host::NewMatrixCopy<floatt>(1, length, hArray, hArray1);
  math::Matrix* dparam0 = oap::cuda::NewDeviceMatrix(true, true, 1, length);
  oap::cuda::CopyHostArraysToDeviceMatrix(dparam0, hArray, hArray1);
  floatt doutput = 1;
  floatt doutput1 = 1;
  floatt doutput2 = 1;
  cuMatrix->magnitude(doutput, dparam0);
  cuMatrix->magnitudeOpt(doutput1, dparam0);
  cuMatrix->magnitudeOptVer2(doutput2, dparam0);

  floatt output;
  mocpu.magnitude(&output, matrix);

  oap::host::DeleteMatrix(matrix);
  oap::cuda::DeleteDeviceMatrix(dparam0);
  EXPECT_DOUBLE_EQ(0, doutput);
  EXPECT_DOUBLE_EQ(output, doutput);
  EXPECT_DOUBLE_EQ(output, doutput1);
  EXPECT_DOUBLE_EQ(output, doutput2);
  delete[] hArray;
  delete[] hArray1;
}

TEST_F(OapMatrixCudaTests, MagnitudeRealMatrixBigDataTest2) {
  std::string text =
      "(columns=1, rows=16384) [0, -0.25 <repeats 2 times>, 0, -0.25, 0 "
      "<repeats 3 times>, -0.25, 0 <repeats 7 times>, -0.25, 0 <repeats 15 "
      "times>, -0.25, 0 <repeats 95 times>, -0.25, 0 <repeats 127 times>, "
      "-0.25, 0 <repeats 255 times>, -0.25, 0 <repeats 511 times>, -0.25, 0 "
      "<repeats 1023 times>, -0.25, 0 <repeats 2047 times>, -0.25, 0 <repeats "
      "4095 times>, -0.25, 0 <repeats 8191 times>] (length=16384) [0 <repeats "
      "16384 times>] (length=16384)";

  math::Matrix* matrix = oap::host::NewMatrix(text);
  math::Matrix* dmatrix = oap::cuda::NewDeviceMatrixHostRef(matrix);
  oap::cuda::CopyHostMatrixToDeviceMatrix(dmatrix, matrix);

  floatt doutput = 10;
  floatt doutput1 = 10;
  floatt doutput2 = 10;

  cuMatrix->magnitude(doutput, dmatrix);
  cuMatrix->magnitudeOpt(doutput1, dmatrix);
  cuMatrix->magnitudeOptVer2(doutput2, dmatrix);

  EXPECT_DOUBLE_EQ(0.9013878188659973, doutput);
  EXPECT_DOUBLE_EQ(0.9013878188659973, doutput1);
  EXPECT_DOUBLE_EQ(0.9013878188659973, doutput2);

  oap::cuda::DeleteDeviceMatrix(dmatrix);
  oap::host::DeleteMatrix(matrix);
}

TEST_F(OapMatrixCudaTests, DotProductBigDataTest) {
  math::Matrix* Q = oap::host::NewMatrix(Qstr);
  math::Matrix* QJ = oap::host::NewMatrix(QJstr);

  math::Matrix* dQJ = oap::cuda::NewDeviceMatrixHostRef(QJ);
  math::Matrix* dQ = oap::cuda::NewDeviceMatrixHostRef(Q);
  math::Matrix* doutput = oap::cuda::NewDeviceMatrixHostRef(Q);

  cuMatrix->dotProduct(doutput, dQ, dQJ);
  cuMatrix->dotProduct(doutput, dQJ, dQ);

  oap::cuda::DeleteDeviceMatrix(dQJ);
  oap::cuda::DeleteDeviceMatrix(dQ);
  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::host::DeleteMatrix(Q);
  oap::host::DeleteMatrix(QJ);
}

TEST_F(OapMatrixCudaTests, SetVectorAndCopyTest) {
  size_t columns = 2000;
  math::Matrix* host = oap::host::NewReMatrix(columns, 1);
  for (size_t fa = 0; fa < columns; ++fa) {
    host->reValues[fa] = fa;
  }

  math::Matrix* device = oap::cuda::NewDeviceMatrixCopy(host);
  oap::host::DeleteMatrix(host);
  host = NULL;

  math::Matrix* host1 = oap::host::NewReMatrix(columns, 1);

  for (size_t fa = 0; fa < columns; ++fa) {
    host1->reValues[fa] = columns + 1;
    EXPECT_EQ(columns + 1, host1->reValues[fa]);
  }

  oap::cuda::CopyDeviceMatrixToHostMatrix(host1, device);

  for (size_t fa = 0; fa < columns; ++fa) {
    EXPECT_EQ(fa, host1->reValues[fa]);
  }

  oap::host::DeleteMatrix(host1);
  oap::cuda::DeleteDeviceMatrix(device);
}
