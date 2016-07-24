
#include "gtest/gtest.h"
#include <string>
#include "Matrix.h"
#include "HostMatrixKernels.h"
#include "DeviceMatrixKernels.h"
#include "DeviceMatrixModules.h"
#include "MatrixProcedures.h"
#include "MockUtils.h"
#include "schur1.h"
#include "schur2.h"
#include "schur3.h"

class OglaSchurDecomposition : public testing::Test {
 public:
  OglaSchurDecomposition() {}

  virtual ~OglaSchurDecomposition() {}

  virtual void SetUp() { device::Context::Instance().create(); }

  virtual void TearDown() { device::Context::Instance().destroy(); }

  enum KernelType { DEVICE, HOST };

  void executeTestQUQT(math::Matrix* H, math::Matrix* Q, math::Matrix* QT,
                       math::Matrix* output1, math::Matrix* output2,
                       math::Matrix* eq_initMatrix, math::Matrix* hostMatrix,
                       CuMatrix& cuMatrix) {
    //cuMatrix.transposeMatrix(QT, Q);
    //cuMatrix.dotProduct(output2, H, QT);
    //cuMatrix.dotProduct(output1, Q, output2);

    cuMatrix.calculateQTHQ(output1, H, Q, QT);

    device::CopyDeviceMatrixToHostMatrix(hostMatrix, output1);
    EXPECT_THAT(eq_initMatrix, MatrixIsEqual(hostMatrix));
  }

  void executeTest(const std::string& matrixStr,
                   const std::string& eq_matrixStr,
                   OglaSchurDecomposition::KernelType kernelType) {
    CuMatrix cuMatrix;
    math::Matrix* matrix = device::NewDeviceMatrix(matrixStr);
    math::Matrix* matrix1 = device::NewDeviceMatrix(matrix);
    math::Matrix* matrix2 = device::NewDeviceMatrix(matrix);
    math::Matrix* matrix3 = device::NewDeviceMatrix(matrix);
    math::Matrix* matrix4 = device::NewDeviceMatrix(matrix);
    math::Matrix* matrix5 = device::NewDeviceMatrix(matrix);
    math::Matrix* matrix6 = device::NewDeviceMatrix(matrix);
    math::Matrix* matrix7 = device::NewDeviceMatrix(matrix);
    math::Matrix* matrix8 = device::NewDeviceMatrix(matrix);

    math::Matrix* eq_hostMatrix = host::NewMatrix(eq_matrixStr);
    math::Matrix* eq_initMatrix = host::NewMatrix(matrixStr);
    math::Matrix* hostMatrix = host::NewMatrix(eq_hostMatrix);

    math::Matrix* H = matrix;
    math::Matrix* Q = matrix1;
    math::Matrix* QT = matrix8;
    math::Matrix* output1 = matrix7;
    math::Matrix* output2 = matrix6;

    if (kernelType == OglaSchurDecomposition::DEVICE) {
      device::Kernel kernel;
      kernel.load("liboglaMatrixCuda.cubin");
      uintt columns = eq_hostMatrix->columns;
      uintt rows = eq_hostMatrix->rows;
      DEVICEKernel_CalcTriangularH(matrix, matrix1, matrix2, matrix3, matrix4,
                                   matrix5, matrix6, matrix7, matrix8, columns,
                                   rows, kernel);
      device::CopyDeviceMatrixToHostMatrix(hostMatrix, matrix);
      //EXPECT_THAT(eq_hostMatrix, MatrixIsEqual(hostMatrix));
      EXPECT_THAT(eq_hostMatrix, MatrixContainsDiagonalValues(hostMatrix));
    } else if (kernelType == OglaSchurDecomposition::HOST) {
    }

    //executeTestQUQT(H, Q, QT, output1, output2, eq_initMatrix, hostMatrix,
    //                cuMatrix);

    host::DeleteMatrix(eq_hostMatrix);
    host::DeleteMatrix(eq_initMatrix);
    host::DeleteMatrix(hostMatrix);
    device::DeleteDeviceMatrix(matrix);
    device::DeleteDeviceMatrix(matrix1);
    device::DeleteDeviceMatrix(matrix2);
    device::DeleteDeviceMatrix(matrix3);
    device::DeleteDeviceMatrix(matrix4);
    device::DeleteDeviceMatrix(matrix5);
    device::DeleteDeviceMatrix(matrix6);
    device::DeleteDeviceMatrix(matrix7);
    device::DeleteDeviceMatrix(matrix8);
  }
};

TEST_F(OglaSchurDecomposition, Test1) {
  executeTest(test::schur1::matrix, test::schur1::eq_matrix,
              OglaSchurDecomposition::DEVICE);
}

TEST_F(OglaSchurDecomposition, Test2) {
  executeTest(test::schur2::matrix, test::schur2::eq_matrix,
              OglaSchurDecomposition::DEVICE);
}

TEST_F(OglaSchurDecomposition, Test3) {
  executeTest(test::schur3::matrix, test::schur3::eq_matrix,
              OglaSchurDecomposition::DEVICE);
}