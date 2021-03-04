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
#include "CuProceduresApi.h"
#include "MathOperationsCpu.h"
#include "oapHostMatrixUtils.h"
#include "oapCudaMatrixUtils.h"
#include "KernelExecutor.h"

#include "oapHostMatrixPtr.h"
#include "oapHostMatrixUPtr.h"
#include "oapDeviceMatrixPtr.h"
#include "oapDeviceMatrixUPtr.h"
#include "oapCudaMemoryApi.h"
#include "oapHostMemoryApi.h"

class OapMatrixCudaTests : public testing::Test {
 public:
  math::ComplexMatrix* output;
  math::ComplexMatrix* eq_output;
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

TEST_F(OapMatrixCudaTests, CreationAndCopyTests) {
  oap::DeviceMatrixUPtr dmatrix = oap::cuda::NewDeviceReMatrixWithValue (1, 1, 12.);
  oap::HostMatrixUPtr hmatrix = oap::host::NewReMatrixWithValue (1, 1, 0);
  oap::cuda::CopyDeviceMatrixToHostMatrix (hmatrix, dmatrix);
  EXPECT_EQ (12., hmatrix->re.mem.ptr[0]);
  EXPECT_EQ (1, hmatrix->re.mem.dims.width);
  EXPECT_EQ (1, hmatrix->re.mem.dims.height);
}

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
  math::ComplexMatrix* V = oap::cuda::NewDeviceMatrix (output, 10, 10);
  math::ComplexMatrix* v = oap::cuda::NewDeviceMatrix (output, 1, 10);
  oap::cuda::CopyHostArrayToDeviceMatrix (v, hVArray, NULL, sizeof(hVArray) / sizeof(floatt));
  oap::cuda::CopyHostMatrixToDeviceMatrix (V, output);

  cuMatrix->setVector (V, 0, v, 10);
  oap::cuda::CopyDeviceMatrixToHostMatrix(output, V);

  oap::cuda::DeleteDeviceMatrix(V);
  oap::cuda::DeleteDeviceMatrix(v);

  eq_output = oap::host::NewReMatrixCopy (10, 10, hOutputArray);
}

TEST_F(OapMatrixCudaTests, SetVectorTest1) {
  floatt hArray[] = {
      5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  floatt hOutputArray[] = {
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  floatt hVArray[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };

  output = oap::host::NewReMatrixCopy(10, 10, hArray);
  math::ComplexMatrix* V = oap::cuda::NewDeviceMatrix(output, 10, 10);
  math::ComplexMatrix* v = oap::cuda::NewDeviceMatrix(output, 2, 10);
  oap::cuda::CopyHostArrayToDeviceMatrix(v, hVArray, NULL, sizeof(hVArray) / sizeof(floatt));
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
  math::ComplexMatrix* V = oap::cuda::NewDeviceMatrix(output, 10, 10);
  math::ComplexMatrix* v = oap::cuda::NewDeviceMatrix(output, 1, 10);
  oap::cuda::CopyHostArrayToDeviceMatrix(v, hArray, NULL, sizeof(hArray) / sizeof(floatt));
  oap::cuda::CopyHostArrayToDeviceMatrix(V, hVArray, NULL, sizeof(hVArray) / sizeof(floatt));

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
  math::ComplexMatrix* V = oap::cuda::NewDeviceMatrix(output, 10, 10);
  math::ComplexMatrix* v = oap::cuda::NewDeviceMatrix(output, 2, 10);
  oap::cuda::CopyHostArrayToDeviceMatrix(v, hArray, NULL, sizeof(hArray) / sizeof(floatt));
  oap::cuda::CopyHostArrayToDeviceMatrix(V, hVArray, NULL, sizeof(hVArray) / sizeof(floatt));

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
  math::ComplexMatrix* matrix = oap::cuda::NewDeviceMatrix(output, 10, 10);
  oap::cuda::CopyHostArrayToDeviceMatrix(matrix, hArray, NULL, sizeof (hArray) / sizeof (floatt));
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
  math::ComplexMatrix* matrix = oap::cuda::NewDeviceMatrix(output, 10, 10);
  oap::cuda::CopyHostArrayToDeviceMatrix(matrix, hArray, NULL, sizeof (hArray) / sizeof (floatt));
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
  oap::DeviceMatrixPtr doutput = oap::cuda::NewDeviceMatrix(output, 10, 10);
  oap::DeviceMatrixPtr dparam0 = oap::cuda::NewDeviceMatrix(output, 10, 10);
  oap::cuda::CopyHostArrayToDeviceMatrix(dparam0, hArray, NULL, sizeof (hArray) / sizeof (floatt));

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
  oap::DeviceMatrixPtr doutput = oap::cuda::NewDeviceMatrix(output, 10, 4);
  oap::DeviceMatrixPtr dparam0 = oap::cuda::NewDeviceMatrix(output, 4, 10);
  oap::cuda::CopyHostArrayToDeviceMatrix(dparam0, hArray, NULL, sizeof (hArray) / sizeof (floatt));

  MatrixEx* matrixEx = oap::cuda::NewDeviceMatrixEx();
  MatrixEx hMatrixEx = {0, 10, 0, 2};
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
  oap::DeviceMatrixPtr doutput = oap::cuda::NewDeviceMatrix(output, 10, 10);
  oap::DeviceMatrixPtr dparam0 = oap::cuda::NewDeviceMatrix(output, 10, 10);
  oap::cuda::CopyHostArrayToDeviceMatrix(dparam0, hArray, NULL, sizeof (hArray) / sizeof (floatt));

  MatrixEx* matrixEx = oap::cuda::NewDeviceMatrixEx();
  MatrixEx hMatrixEx = {0, 10, 0, 2};
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
  oap::DeviceMatrixPtr doutput = oap::cuda::NewDeviceMatrix(output, 10, 4);
  oap::DeviceMatrixPtr dparam0 = oap::cuda::NewDeviceMatrix(output, 10, 10);
  oap::cuda::CopyHostArrayToDeviceMatrix(dparam0, hArray, NULL, sizeof(hArray) / sizeof (floatt));

  MatrixEx* matrixEx = oap::cuda::NewDeviceMatrixEx();
  MatrixEx hMatrixEx = {0, 10, 0, 2};
  oap::cuda::SetMatrixEx(matrixEx, &hMatrixEx);

  cuMatrix->transposeEx(doutput, dparam0, matrixEx);

  oap::cuda::DeleteDeviceMatrixEx(matrixEx);
  oap::cuda::CopyDeviceMatrixToHostMatrix(output, doutput);
  eq_output = oap::host::NewReMatrixCopy(10, 4, hOutputArray);
}

TEST_F(OapMatrixCudaTests, MatrixExTest) {
  MatrixEx** dMatrixExs = oap::cuda::NewDeviceMatrixEx(5);
  uintt buffer[] = {0, 10, 0, 1,  0,  1, 0,  15, 0, 0, 0,
                    0, 0, 25, 30, 35, 0,  1, 0,  2};
  oap::cuda::SetMatrixEx(dMatrixExs, buffer, 5);
  MatrixEx matrixEx;

  CudaUtils::CopyDeviceToHost(&matrixEx, dMatrixExs[0], sizeof(MatrixEx));
  EXPECT_THAT(matrixEx, MatrixExIsEqual(buffer));

  CudaUtils::CopyDeviceToHost(&matrixEx, dMatrixExs[1], sizeof(MatrixEx));
  EXPECT_THAT(matrixEx, MatrixExIsEqual(&buffer[4]));

  CudaUtils::CopyDeviceToHost(&matrixEx, dMatrixExs[2], sizeof(MatrixEx));
  EXPECT_THAT(matrixEx, MatrixExIsEqual(&buffer[8]));

  CudaUtils::CopyDeviceToHost(&matrixEx, dMatrixExs[3], sizeof(MatrixEx));
  EXPECT_THAT(matrixEx, MatrixExIsEqual(&buffer[12]));

  CudaUtils::CopyDeviceToHost(&matrixEx, dMatrixExs[4], sizeof(MatrixEx));
  EXPECT_THAT(matrixEx, MatrixExIsEqual(&buffer[16]));
}

TEST_F(OapMatrixCudaTests, MagnitudeReMatrixTest) {
  floatt hArray[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };

  math::MathOperationsCpu mocpu;

  math::ComplexMatrix* matrix = oap::host::NewMatrixCopy<floatt>(1, 10, hArray, NULL);
  math::ComplexMatrix* dparam0 = oap::cuda::NewDeviceMatrix(true, false, 1, 10);
  oap::cuda::CopyHostArrayToDeviceMatrix(dparam0, hArray, NULL, sizeof (hArray) / sizeof (floatt));
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

  math::ComplexMatrix* matrix = oap::host::NewMatrixCopy<floatt>(1, 10, hArray, NULL);
  math::ComplexMatrix* dparam0 = oap::cuda::NewDeviceMatrix(true, false, 1, 10);
  oap::cuda::CopyHostArrayToDeviceMatrix(dparam0, hArray, NULL, sizeof (hArray) / sizeof (floatt));
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

  math::ComplexMatrix* matrix = oap::host::NewMatrixCopy<floatt>(1, 10, hArray, NULL);
  math::ComplexMatrix* dparam0 = oap::cuda::NewDeviceMatrix(true, false, 1, 10);
  oap::cuda::CopyHostArrayToDeviceMatrix(dparam0, hArray, NULL, sizeof (hArray) / sizeof (floatt));
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

  math::ComplexMatrix* matrix = oap::host::NewMatrixCopy<floatt>(1, length, hArray, NULL);
  math::ComplexMatrix* dparam0 = oap::cuda::NewDeviceMatrix(true, false, 1, length);
  oap::cuda::CopyHostArrayToDeviceMatrix(dparam0, hArray, NULL, length);
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

  math::ComplexMatrix* matrix = oap::host::NewMatrixCopy<floatt>(1, length, hArray, hArray1);
  math::ComplexMatrix* dparam0 = oap::cuda::NewDeviceMatrix(true, true, 1, length);
  oap::cuda::CopyHostArrayToDeviceMatrix(dparam0, hArray, hArray1, length);
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

  math::ComplexMatrix* matrix = oap::host::NewMatrixCopy<floatt>(1, length, hArray, hArray1);
  math::ComplexMatrix* dparam0 = oap::cuda::NewDeviceMatrix(true, true, 1, length);
  oap::cuda::CopyHostArrayToDeviceMatrix(dparam0, hArray, hArray1, length);
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

  math::ComplexMatrix* matrix = oap::host::NewMatrix(text);
  math::ComplexMatrix* dmatrix = oap::cuda::NewDeviceMatrixHostRef(matrix);
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

TEST_F(OapMatrixCudaTests, SetVectorAndCopyTest) {
  size_t columns = 2000;
  math::ComplexMatrix* host = oap::host::NewReMatrix(columns, 1);
  for (size_t fa = 0; fa < columns; ++fa) {
    *GetRePtrIndex (host, fa) = fa;
  }

  math::ComplexMatrix* device = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(host);
  oap::host::DeleteMatrix(host);
  host = NULL;

  math::ComplexMatrix* host1 = oap::host::NewReMatrix(columns, 1);

  for (size_t fa = 0; fa < columns; ++fa) {
    *GetRePtrIndex (host1, fa) = columns + 1;
    EXPECT_EQ(columns + 1, GetReIndex (host1, fa));
  }

  oap::cuda::CopyDeviceMatrixToHostMatrix(host1, device);

  for (size_t fa = 0; fa < columns; ++fa) {
    EXPECT_EQ(fa, GetReIndex (host1, fa));
  }

  oap::host::DeleteMatrix(host1);
  oap::cuda::DeleteDeviceMatrix(device);
}

TEST_F(OapMatrixCudaTests, CopyTest_1)
{
  {
    oap::Memory memory1 = oap::cuda::NewMemoryWithValues ({1, 8}, 0);
    oap::Memory memory2 = oap::cuda::NewMemoryWithValues ({1, 1}, 2.f);

    oap::cuda::CopyDeviceToDevice (memory1, {0, 0}, memory2, {{0, 0}, {1, 1}});

    oap::Memory hmemory = oap::host::NewMemoryWithValues ({1, 8}, 10);
    oap::cuda::CopyDeviceToHost (hmemory, memory1);

    EXPECT_EQ (2.f, oap::common::GetValue (hmemory, oap::common::OAP_NONE_REGION(), 0, 0));
    EXPECT_EQ (0.f, oap::common::GetValue (hmemory, oap::common::OAP_NONE_REGION(), 0, 1));
    EXPECT_EQ (0.f, oap::common::GetValue (hmemory, oap::common::OAP_NONE_REGION(), 0, 3));
    EXPECT_EQ (0.f, oap::common::GetValue (hmemory, oap::common::OAP_NONE_REGION(), 0, 4));
    EXPECT_EQ (0.f, oap::common::GetValue (hmemory, oap::common::OAP_NONE_REGION(), 0, 5));
    EXPECT_EQ (0.f, oap::common::GetValue (hmemory, oap::common::OAP_NONE_REGION(), 0, 6));
    EXPECT_EQ (0.f, oap::common::GetValue (hmemory, oap::common::OAP_NONE_REGION(), 0, 7));

    oap::host::DeleteMemory (hmemory);
    oap::cuda::DeleteMemory (memory1);
    oap::cuda::DeleteMemory (memory2);
  }
  {
    math::ComplexMatrix* matrix1 = oap::cuda::NewDeviceReMatrixWithValue (1, 8, 0);
    math::ComplexMatrix* matrix2 = oap::cuda::NewDeviceReMatrixWithValue (1, 1, 2.f);

    oap::cuda::SetReMatrix (matrix1, matrix2, 0, 0);

    oap::HostMatrixUPtr hmatrix = oap::host::NewReMatrixWithValue (1, 8, 10);
    oap::cuda::CopyDeviceMatrixToHostMatrix (hmatrix, matrix1);

    EXPECT_EQ (2.f, oap::common::GetValue (hmatrix->re.mem, oap::common::OAP_NONE_REGION(), 0, 0));
    EXPECT_EQ (0.f, oap::common::GetValue (hmatrix->re.mem, oap::common::OAP_NONE_REGION(), 0, 1));
    EXPECT_EQ (0.f, oap::common::GetValue (hmatrix->re.mem, oap::common::OAP_NONE_REGION(), 0, 2));
    EXPECT_EQ (0.f, oap::common::GetValue (hmatrix->re.mem, oap::common::OAP_NONE_REGION(), 0, 3));
    EXPECT_EQ (0.f, oap::common::GetValue (hmatrix->re.mem, oap::common::OAP_NONE_REGION(), 0, 4));
    EXPECT_EQ (0.f, oap::common::GetValue (hmatrix->re.mem, oap::common::OAP_NONE_REGION(), 0, 5));
    EXPECT_EQ (0.f, oap::common::GetValue (hmatrix->re.mem, oap::common::OAP_NONE_REGION(), 0, 6));
    EXPECT_EQ (0.f, oap::common::GetValue (hmatrix->re.mem, oap::common::OAP_NONE_REGION(), 0, 7));

    oap::cuda::DeleteDeviceMatrix (matrix1);
    oap::cuda::DeleteDeviceMatrix (matrix2);
  }
}
