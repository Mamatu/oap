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
#include <string>
#include "Matrix.h"
#include "HostMatrixKernels.h"
#include "DeviceMatrixKernels.h"
#include "oapCudaMatrixUtils.h"
#include "CuProceduresApi.h"
#include "MatchersUtils.h"
#include "schur1.h"
#include "schur2.h"
#include "schur3.h"

class OapSchurDecomposition : public testing::Test {
 public:
  OapSchurDecomposition() {}

  virtual ~OapSchurDecomposition() {}

  virtual void SetUp() { oap::cuda::Context::Instance().create(); }

  virtual void TearDown() { oap::cuda::Context::Instance().destroy(); }

  enum KernelType { DEVICE, HOST };

  void executeTestQUQT(math::Matrix* H, math::Matrix* Q, math::Matrix* QT,
                       math::Matrix* output1, math::Matrix* output2,
                       math::Matrix* eq_initMatrix, math::Matrix* hostMatrix,
                       oap::CuProceduresApi& cuMatrix) {
    //cuMatrix.transposeMatrix(QT, Q);
    //cuMatrix.dotProduct(output2, H, QT);
    //cuMatrix.dotProduct(output1, Q, output2);

    cuMatrix.calculateQTHQ(output1, H, Q, QT);

    oap::cuda::CopyDeviceMatrixToHostMatrix(hostMatrix, output1);
    EXPECT_THAT(eq_initMatrix, MatrixIsEqual(hostMatrix));
  }

  void executeTest(const std::string& matrixStr,
                   const std::string& eq_matrixStr,
                   OapSchurDecomposition::KernelType kernelType) {
    oap::CuProceduresApi cuMatrix;
    math::Matrix* matrix = oap::cuda::NewDeviceMatrix(matrixStr);
    math::Matrix* matrix1 = oap::cuda::NewDeviceMatrixDeviceRef(matrix);
    math::Matrix* matrix2 = oap::cuda::NewDeviceMatrixDeviceRef(matrix);
    math::Matrix* matrix3 = oap::cuda::NewDeviceMatrixDeviceRef(matrix);
    math::Matrix* matrix4 = oap::cuda::NewDeviceMatrixDeviceRef(matrix);
    math::Matrix* matrix5 = oap::cuda::NewDeviceMatrixDeviceRef(matrix);
    math::Matrix* matrix6 = oap::cuda::NewDeviceMatrixDeviceRef(matrix);
    math::Matrix* matrix7 = oap::cuda::NewDeviceMatrixDeviceRef(matrix);
    math::Matrix* matrix8 = oap::cuda::NewDeviceMatrixDeviceRef(matrix);

    math::Matrix* eq_hostMatrix = oap::host::NewMatrix(eq_matrixStr);
    math::Matrix* eq_initMatrix = oap::host::NewMatrix(matrixStr);
    math::Matrix* hostMatrix = oap::host::NewMatrixRef(eq_hostMatrix);

    math::Matrix* H = matrix;
    math::Matrix* Q = matrix1;
    math::Matrix* QT = matrix8;
    math::Matrix* output1 = matrix7;
    math::Matrix* output2 = matrix6;

    if (kernelType == OapSchurDecomposition::DEVICE) {
      oap::cuda::Kernel kernel;
      kernel.load("liboapMatrixCuda.cubin");
      uintt columns = gColumns (eq_hostMatrix);
      uintt rows = gRows (eq_hostMatrix);
      DEVICEKernel_CalcTriangularH(matrix, matrix1, matrix2, matrix3, matrix4,
                                   matrix5, matrix6, matrix7, matrix8, columns,
                                   rows, kernel);
      oap::cuda::CopyDeviceMatrixToHostMatrix(hostMatrix, matrix);
      //EXPECT_THAT(eq_hostMatrix, MatrixIsEqual(hostMatrix));
      EXPECT_THAT(eq_hostMatrix, MatrixContainsDiagonalValues(hostMatrix));
    } else if (kernelType == OapSchurDecomposition::HOST) {
    }

    //executeTestQUQT(H, Q, QT, output1, output2, eq_initMatrix, hostMatrix,
    //                cuMatrix);

    oap::host::DeleteMatrix(eq_hostMatrix);
    oap::host::DeleteMatrix(eq_initMatrix);
    oap::host::DeleteMatrix(hostMatrix);
    oap::cuda::DeleteDeviceMatrix(matrix);
    oap::cuda::DeleteDeviceMatrix(matrix1);
    oap::cuda::DeleteDeviceMatrix(matrix2);
    oap::cuda::DeleteDeviceMatrix(matrix3);
    oap::cuda::DeleteDeviceMatrix(matrix4);
    oap::cuda::DeleteDeviceMatrix(matrix5);
    oap::cuda::DeleteDeviceMatrix(matrix6);
    oap::cuda::DeleteDeviceMatrix(matrix7);
    oap::cuda::DeleteDeviceMatrix(matrix8);
  }
};

TEST_F(OapSchurDecomposition, Test1) {
  executeTest(test::schur1::matrix, test::schur1::eq_matrix,
              OapSchurDecomposition::DEVICE);
}

TEST_F(OapSchurDecomposition, Test2) {
  executeTest(test::schur2::matrix, test::schur2::eq_matrix,
              OapSchurDecomposition::DEVICE);
}

TEST_F(OapSchurDecomposition, DISABLED_Test3) {
  executeTest(test::schur3::matrix, test::schur3::eq_matrix,
              OapSchurDecomposition::DEVICE);
}
