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
#include "MatrixProcedures.h"
#include "MathOperationsCpu.h"
#include "HostMatrixUtils.h"
#include "DeviceMatrixModules.h"
#include "KernelExecutor.h"
#include "matrix6.h"

class OapMatrixCudaTests : public testing::Test {
 public:
  math::Matrix* output;
  math::Matrix* eq_output;
  CuMatrix* cuMatrix;
  CUresult status;

  virtual void SetUp() {
    status = CUDA_SUCCESS;
    device::Context::Instance().create();
    output = NULL;
    eq_output = NULL;
    cuMatrix = new CuMatrix();
  }

  virtual void TearDown() {
    delete cuMatrix;
    if (output != NULL && eq_output != NULL) {
      EXPECT_THAT(output, MatrixIsEqual(eq_output));
    }
    EXPECT_EQ(status, CUDA_SUCCESS);
    host::DeleteMatrix(output);
    host::DeleteMatrix(eq_output);
    device::Context::Instance().destroy();
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

  output = host::NewReMatrixCopy(10, 10, hArray);
  math::Matrix* V = device::NewDeviceMatrix(output, 10, 10);
  math::Matrix* v = device::NewDeviceMatrix(output, 1, 10);
  device::CopyHostArraysToDeviceMatrix(v, hVArray, NULL);
  device::CopyHostMatrixToDeviceMatrix(V, output);

  cuMatrix->setVector(V, 0, v, 10);
  device::CopyDeviceMatrixToHostMatrix(output, V);

  device::DeleteDeviceMatrix(V);
  device::DeleteDeviceMatrix(v);

  eq_output = host::NewReMatrixCopy(10, 10, hOutputArray);
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

  output = host::NewReMatrixCopy(10, 10, hArray);
  math::Matrix* V = device::NewDeviceMatrix(output, 10, 10);
  math::Matrix* v = device::NewDeviceMatrix(output, 2, 10);
  device::CopyHostArraysToDeviceMatrix(v, hVArray, NULL);
  device::CopyHostMatrixToDeviceMatrix(V, output);

  cuMatrix->setVector(V, 0, v, 10);
  device::CopyDeviceMatrixToHostMatrix(output, V);

  device::DeleteDeviceMatrix(V);
  device::DeleteDeviceMatrix(v);

  eq_output = host::NewReMatrixCopy(10, 10, hOutputArray);
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

  output = host::NewReMatrixCopy(1, 10, hArray);
  math::Matrix* V = device::NewDeviceMatrix(output, 10, 10);
  math::Matrix* v = device::NewDeviceMatrix(output, 1, 10);
  device::CopyHostArraysToDeviceMatrix(v, hArray, NULL);
  device::CopyHostArraysToDeviceMatrix(V, hVArray, NULL);

  cuMatrix->getVector(v, 10, V, 0);
  device::CopyDeviceMatrixToHostMatrix(output, v);

  device::DeleteDeviceMatrix(V);
  device::DeleteDeviceMatrix(v);

  eq_output = host::NewReMatrixCopy(1, 10, hOutputArray);
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

  output = host::NewReMatrixCopy(2, 10, hArray);
  math::Matrix* V = device::NewDeviceMatrix(output, 10, 10);
  math::Matrix* v = device::NewDeviceMatrix(output, 2, 10);
  device::CopyHostArraysToDeviceMatrix(v, hArray, NULL);
  device::CopyHostArraysToDeviceMatrix(V, hVArray, NULL);

  cuMatrix->getVector(v, 10, V, 0);
  device::CopyDeviceMatrixToHostMatrix(output, v);

  device::DeleteDeviceMatrix(V);
  device::DeleteDeviceMatrix(v);

  eq_output = host::NewReMatrixCopy(2, 10, hOutputArray);
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

  output = host::NewReMatrixCopy(10, 10, hArray);
  math::Matrix* matrix = device::NewDeviceMatrix(output, 10, 10);
  device::CopyHostArraysToDeviceMatrix(matrix, hArray, NULL);
  cuMatrix->setIdentity(matrix);
  device::CopyDeviceMatrixToHostMatrix(output, matrix);

  device::DeleteDeviceMatrix(matrix);

  eq_output = host::NewReMatrixCopy(10, 10, hOutputArray);
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

  output = host::NewReMatrixCopy(10, 10, hArray);
  math::Matrix* matrix = device::NewDeviceMatrix(output, 10, 10);
  device::CopyHostArraysToDeviceMatrix(matrix, hArray, NULL);
  cuMatrix->setDiagonal(matrix, 2, 0);
  device::CopyDeviceMatrixToHostMatrix(output, matrix);

  device::DeleteDeviceMatrix(matrix);

  eq_output = host::NewReMatrixCopy(10, 10, hOutputArray);
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

  output = host::NewReMatrixCopy(10, 10, hArray);
  math::Matrix* doutput = device::NewDeviceMatrix(output, 10, 10);
  math::Matrix* dparam0 = device::NewDeviceMatrix(output, 10, 10);
  device::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);

  cuMatrix->multiplyConstantMatrix(doutput, dparam0, 5);

  device::CopyDeviceMatrixToHostMatrix(output, doutput);
  eq_output = host::NewReMatrixCopy(10, 10, hOutputArray);
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

  output = host::NewReMatrixCopy(10, 4, hArray);
  math::Matrix* doutput = device::NewDeviceMatrix(output, 10, 4);
  math::Matrix* dparam0 = device::NewDeviceMatrix(output, 4, 10);
  device::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);

  MatrixEx* matrixEx = device::NewDeviceMatrixEx();
  MatrixEx hMatrixEx = {0, 10, 0, 2, 0, 0};
  device::SetMatrixEx(matrixEx, &hMatrixEx);

  cuMatrix->transposeMatrixEx(doutput, dparam0, matrixEx);

  device::DeleteDeviceMatrixEx(matrixEx);
  device::CopyDeviceMatrixToHostMatrix(output, doutput);
  eq_output = host::NewReMatrixCopy(10, 4, hOutputArray);
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

  output = host::NewReMatrixCopy(10, 10, hArray);
  math::Matrix* doutput = device::NewDeviceMatrix(output, 10, 10);
  math::Matrix* dparam0 = device::NewDeviceMatrix(output, 10, 10);
  device::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);

  MatrixEx* matrixEx = device::NewDeviceMatrixEx();
  MatrixEx hMatrixEx = {0, 10, 0, 2, 0, 0};
  device::SetMatrixEx(matrixEx, &hMatrixEx);

  cuMatrix->transposeMatrixEx(doutput, dparam0, matrixEx);

  device::DeleteDeviceMatrixEx(matrixEx);
  device::CopyDeviceMatrixToHostMatrix(output, doutput);
  eq_output = host::NewReMatrixCopy(10, 10, hOutputArray);
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

  output = host::NewReMatrixCopy(10, 4, hArray);
  math::Matrix* doutput = device::NewDeviceMatrix(output, 10, 4);
  math::Matrix* dparam0 = device::NewDeviceMatrix(output, 10, 10);
  device::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);

  MatrixEx* matrixEx = device::NewDeviceMatrixEx();
  MatrixEx hMatrixEx = {0, 10, 0, 2, 0, 0};
  device::SetMatrixEx(matrixEx, &hMatrixEx);

  cuMatrix->transposeMatrixEx(doutput, dparam0, matrixEx);

  device::DeleteDeviceMatrixEx(matrixEx);
  device::CopyDeviceMatrixToHostMatrix(output, doutput);
  eq_output = host::NewReMatrixCopy(10, 4, hOutputArray);
}

TEST_F(OapMatrixCudaTests, MatrixExTest) {
  MatrixEx** dMatrixExs = device::NewDeviceMatrixEx(5);
  uintt buffer[] = {0, 10, 0, 1, 0,  0,  0,  1, 0,  15, 0, 20, 0, 0, 0,
                    0, 0,  0, 0, 25, 30, 35, 0, 40, 0,  1, 0,  2, 3, 5};
  device::SetMatrixEx(dMatrixExs, buffer, 5);
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

  math::Matrix* matrix = host::NewMatrixCopy<floatt>(1, 10, hArray, NULL);
  math::Matrix* dparam0 = device::NewDeviceMatrix(true, false, 1, 10);
  device::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);
  floatt output;
  floatt doutput;
  floatt doutput1;
  floatt doutput2;
  cuMatrix->magnitude(doutput, dparam0);
  cuMatrix->magnitudeOpt(doutput1, dparam0);
  cuMatrix->magnitudeOptVer2(doutput2, dparam0);

  mocpu.magnitude(&output, matrix);

  host::DeleteMatrix(matrix);
  device::DeleteDeviceMatrix(dparam0);
  EXPECT_DOUBLE_EQ(doutput, output);
  EXPECT_DOUBLE_EQ(doutput1, output);
  EXPECT_DOUBLE_EQ(doutput2, output);
}

TEST_F(OapMatrixCudaTests, MagnitudeReMatrixTest1) {
  floatt hArray[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  };

  math::MathOperationsCpu mocpu;

  math::Matrix* matrix = host::NewMatrixCopy<floatt>(1, 10, hArray, NULL);
  math::Matrix* dparam0 = device::NewDeviceMatrix(true, false, 1, 10);
  device::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);
  floatt doutput;
  floatt doutput1;
  floatt doutput2;
  cuMatrix->magnitude(doutput, dparam0);
  cuMatrix->magnitudeOpt(doutput1, dparam0);
  cuMatrix->magnitudeOptVer2(doutput2, dparam0);

  floatt output;
  mocpu.magnitude(&output, matrix);

  host::DeleteMatrix(matrix);
  device::DeleteDeviceMatrix(dparam0);
  EXPECT_DOUBLE_EQ(output, doutput);
  EXPECT_DOUBLE_EQ(output, doutput1);
  EXPECT_DOUBLE_EQ(output, doutput2);
}

TEST_F(OapMatrixCudaTests, MagnitudeReMatrixTest2) {
  floatt hArray[] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  math::MathOperationsCpu mocpu;

  math::Matrix* matrix = host::NewMatrixCopy<floatt>(1, 10, hArray, NULL);
  math::Matrix* dparam0 = device::NewDeviceMatrix(true, false, 1, 10);
  device::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);
  floatt doutput;
  floatt doutput1;
  floatt doutput2;
  cuMatrix->magnitude(doutput, dparam0);
  cuMatrix->magnitudeOpt(doutput1, dparam0);
  cuMatrix->magnitudeOptVer2(doutput2, dparam0);

  floatt output;
  mocpu.magnitude(&output, matrix);

  host::DeleteMatrix(matrix);
  device::DeleteDeviceMatrix(dparam0);
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

  math::Matrix* matrix = host::NewMatrixCopy<floatt>(1, length, hArray, NULL);
  math::Matrix* dparam0 = device::NewDeviceMatrix(true, false, 1, length);
  device::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);
  floatt doutput;
  floatt doutput1;
  floatt doutput2;
  cuMatrix->magnitude(doutput, dparam0);
  cuMatrix->magnitudeOpt(doutput1, dparam0);
  cuMatrix->magnitudeOptVer2(doutput2, dparam0);

  floatt output;
  mocpu.magnitude(&output, matrix);

  host::DeleteMatrix(matrix);
  device::DeleteDeviceMatrix(dparam0);
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

  math::Matrix* matrix = host::NewMatrixCopy<floatt>(1, length, hArray, hArray1);
  math::Matrix* dparam0 = device::NewDeviceMatrix(true, true, 1, length);
  device::CopyHostArraysToDeviceMatrix(dparam0, hArray, hArray1);
  floatt doutput = 100;
  floatt doutput1 = 100;
  floatt doutput2 = 100;
  cuMatrix->magnitude(doutput, dparam0);
  cuMatrix->magnitudeOpt(doutput1, dparam0);
  cuMatrix->magnitudeOptVer2(doutput2, dparam0);

  floatt output;
  mocpu.magnitude(&output, matrix);

  host::DeleteMatrix(matrix);
  device::DeleteDeviceMatrix(dparam0);
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

  math::Matrix* matrix = host::NewMatrixCopy<floatt>(1, length, hArray, hArray1);
  math::Matrix* dparam0 = device::NewDeviceMatrix(true, true, 1, length);
  device::CopyHostArraysToDeviceMatrix(dparam0, hArray, hArray1);
  floatt doutput = 1;
  floatt doutput1 = 1;
  floatt doutput2 = 1;
  cuMatrix->magnitude(doutput, dparam0);
  cuMatrix->magnitudeOpt(doutput1, dparam0);
  cuMatrix->magnitudeOptVer2(doutput2, dparam0);

  floatt output;
  mocpu.magnitude(&output, matrix);

  host::DeleteMatrix(matrix);
  device::DeleteDeviceMatrix(dparam0);
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

  math::Matrix* matrix = host::NewMatrix(text);
  math::Matrix* dmatrix = device::NewDeviceMatrixHostRef(matrix);
  device::CopyHostMatrixToDeviceMatrix(dmatrix, matrix);

  floatt doutput = 10;
  floatt doutput1 = 10;
  floatt doutput2 = 10;

  cuMatrix->magnitude(doutput, dmatrix);
  cuMatrix->magnitudeOpt(doutput1, dmatrix);
  cuMatrix->magnitudeOptVer2(doutput2, dmatrix);

  EXPECT_DOUBLE_EQ(0.9013878188659973, doutput);
  EXPECT_DOUBLE_EQ(0.9013878188659973, doutput1);
  EXPECT_DOUBLE_EQ(0.9013878188659973, doutput2);

  device::DeleteDeviceMatrix(dmatrix);
  host::DeleteMatrix(matrix);
}

TEST_F(OapMatrixCudaTests, DotProductBigDataTest) {
  math::Matrix* Q = host::NewMatrix(Qstr);
  math::Matrix* QJ = host::NewMatrix(QJstr);

  math::Matrix* dQJ = device::NewDeviceMatrixHostRef(QJ);
  math::Matrix* dQ = device::NewDeviceMatrixHostRef(Q);
  math::Matrix* doutput = device::NewDeviceMatrixHostRef(Q);

  cuMatrix->dotProduct(doutput, dQ, dQJ);
  cuMatrix->dotProduct(doutput, dQJ, dQ);

  device::DeleteDeviceMatrix(dQJ);
  device::DeleteDeviceMatrix(dQ);
  device::DeleteDeviceMatrix(doutput);
  host::DeleteMatrix(Q);
  host::DeleteMatrix(QJ);
}

TEST_F(OapMatrixCudaTests, SetVectorAndCopyTest) {
  size_t columns = 2000;
  math::Matrix* host = host::NewReMatrix(columns, 1);
  for (size_t fa = 0; fa < columns; ++fa) {
    host->reValues[fa] = fa;
  }

  math::Matrix* device = device::NewDeviceMatrixCopy(host);
  host::DeleteMatrix(host);
  host = NULL;

  math::Matrix* host1 = host::NewReMatrix(columns, 1);

  for (size_t fa = 0; fa < columns; ++fa) {
    host1->reValues[fa] = columns + 1;
    EXPECT_EQ(columns + 1, host1->reValues[fa]);
  }

  device::CopyDeviceMatrixToHostMatrix(host1, device);

  for (size_t fa = 0; fa < columns; ++fa) {
    EXPECT_EQ(fa, host1->reValues[fa]);
  }

  host::DeleteMatrix(host1);
  device::DeleteDeviceMatrix(device);
}
