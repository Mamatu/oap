/*
 * Copyright 2016 Marcin Matula
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
#include "PngFile.h"
#include "EigenCalculator.h"
#include "DeviceDataLoader.h"
#include "Exceptions.h"

#include "ArnoldiProceduresImpl.h"
#include "DeviceMatrixModules.h"
#include "MatrixProcedures.h"

#include <memory>

using namespace ::testing;

class ArnoldiOperations {
 public:
  oap::DeviceDataLoader* m_dataLoader;
  CuMatrix m_cuMatrix;
  math::Matrix* value;

  ArnoldiOperations(oap::DeviceDataLoader* dataLoader)
      : m_dataLoader(dataLoader) {
    value = device::NewDeviceReMatrix(1, 1);
  }

  ~ArnoldiOperations() { device::DeleteDeviceMatrix(value); }

  static void multiplyFunc(math::Matrix* m_w, math::Matrix* m_v, void* userData,
                           CuHArnoldi::MultiplicationType mt) {
    if (mt == CuHArnoldi::TYPE_WV) {
      ArnoldiOperations* ao = static_cast<ArnoldiOperations*>(userData);
      oap::DeviceDataLoader* dataLoader = ao->m_dataLoader;

      math::MatrixInfo matrixInfo = dataLoader->getMatrixInfo();

      for (uintt index = 0; index < matrixInfo.m_matrixDim.columns; ++index) {
        math::Matrix* vec = dataLoader->createDeviceRowVector(index);

        //device::PrintMatrix("vec =", vec);

        ao->m_cuMatrix.dotProduct(ao->value, vec, m_v);
        device::SetMatrix(m_w, ao->value, 0, index);

        device::DeleteDeviceMatrix(vec);
      }
    }
  }

  bool verifyOutput(math::Matrix* vector, floatt value) {
    math::Matrix* matrix = m_dataLoader->createMatrix();

    const uintt partSize = matrix->columns;

    math::Matrix* refMatrix =
        host::NewMatrix(matrix, matrix->columns, partSize);

    host::CopyMatrix(refMatrix, matrix);

    math::Matrix* drefMatrix = device::NewDeviceMatrixCopy(refMatrix);

    math::MatrixInfo info = host::GetMatrixInfo(refMatrix);

    const uintt matrixcolumns = info.m_matrixDim.columns;
    uintt matrixrows = info.m_matrixDim.rows;

    matrixrows = partSize;

    uintt vectorrows = device::GetRows(vector);

    vectorrows = matrix->columns;

    math::Matrix* matrix1 =
        device::NewDeviceReMatrix(matrixrows, matrixcolumns);
    math::Matrix* leftMatrix =
        device::NewDeviceReMatrix(matrixrows, matrixrows);

    math::Matrix* rightMatrix =
        device::NewDeviceReMatrix(matrixrows, matrixrows);

    math::Matrix* vectorT = device::NewDeviceReMatrix(vectorrows, 1);

    m_cuMatrix.transposeMatrix(matrix1, drefMatrix);
    m_cuMatrix.transposeMatrix(vectorT, vector);
    m_cuMatrix.dotProduct(leftMatrix, drefMatrix, matrix1);

    host::PrintMatrix("matrix =", matrix);
    host::PrintMatrix("refMatrix =", refMatrix);
    device::PrintMatrix("matrix1 =", matrix1);
    device::PrintMatrix("drefMatrix =", drefMatrix);
    device::PrintMatrix("vectorT =", vectorT);
    device::PrintMatrix("vector =", vector);
    floatt value2 = value * value;
    m_cuMatrix.multiplyConstantMatrix(vectorT, vectorT, value2);
    m_cuMatrix.dotProduct(rightMatrix, vector, vectorT);
    bool compareResult = m_cuMatrix.compare(leftMatrix, rightMatrix);

    device::PrintMatrix("leftMatrix =", leftMatrix);
    device::PrintMatrix("rightMatrix =", rightMatrix);

    host::DeleteMatrix(refMatrix);
    host::DeleteMatrix(matrix);

    device::DeleteDeviceMatrix(drefMatrix);
    device::DeleteDeviceMatrix(matrix1);
    device::DeleteDeviceMatrix(leftMatrix);
    device::DeleteDeviceMatrix(rightMatrix);
    device::DeleteDeviceMatrix(vectorT);

    return compareResult;
  }

  math::Matrix* createDeviceMatrix() const {
    return m_dataLoader->createDeviceMatrix();
  }

  CuMatrix* operator->() { return &m_cuMatrix; }
};

class OapEigenCalculatorTests : public testing::Test {
 public:
  virtual void SetUp() { device::Context::Instance().create(); }

  virtual void TearDown() { device::Context::Instance().destroy(); }
};

class TestCuHArnoldiCallback : public CuHArnoldiCallback {
 public:
  TestCuHArnoldiCallback(ArnoldiOperations* ao) : m_ao(ao) {}

  bool checkEigenspair(floatt value, math::Matrix* vector, uint index) {
    return true;
    // m_ao->verifyOutput();
  }

 private:
  ArnoldiOperations* m_ao;
};

TEST_F(OapEigenCalculatorTests, NotInitializedTest) {
  TestCuHArnoldiCallback cuharnoldi(nullptr);
  oap::EigenCalculator eigenCalc(&cuharnoldi);
  EXPECT_THROW(eigenCalc.calculate(), oap::exceptions::NotInitialzed);
}

TEST_F(OapEigenCalculatorTests, Calculate) {
  math::Matrix* matrix = NULL;
  debugLongTest();

  try {
    std::unique_ptr<oap::DeviceDataLoader> dataLoader(
        oap::DeviceDataLoader::createDataLoader<oap::PngFile,
                                                oap::DeviceDataLoader>(
            "oap2dt3d/data/images_monkey", "image", 1000, true));

    ArnoldiOperations ao(dataLoader.get());
    TestCuHArnoldiCallback cuharnoldi(&ao);
    cuharnoldi.setCallback(ArnoldiOperations::multiplyFunc, &ao);

    const int ecount = 3;

    floatt reoevalues[ecount];

    oap::EigenCalculator eigenCalculator(&cuharnoldi);

    eigenCalculator.setDataLoader(dataLoader.get());

    eigenCalculator.setEigensCount(ecount);

    math::MatrixInfo matrixInfo = eigenCalculator.getMatrixInfo();

    auto matricesDeleter = [ecount](math::Matrix** evectors) {
      for (int fa = 0; fa < ecount; ++fa) {
        debug("Deleted matrix %p", evectors[fa]);
        device::DeleteDeviceMatrix(evectors[fa]);
      }
      delete[] evectors;
    };

    std::unique_ptr<math::Matrix*, decltype(matricesDeleter)> evectorsUPtr(
        new math::Matrix* [ecount], matricesDeleter);

    [&evectorsUPtr, ecount, &matrixInfo]() {
      math::Matrix** evectors = evectorsUPtr.get();
      const uintt rows = matrixInfo.m_matrixDim.rows;
      for (int fa = 0; fa < ecount; ++fa) {
        evectors[fa] = device::NewDeviceReMatrix(1, rows);
        debug("Created matrix %p", evectors[fa]);
      }
    }();

    math::Matrix** evectors = evectorsUPtr.get();

    eigenCalculator.setEigenvaluesOutput(reoevalues);

    eigenCalculator.setEigenvectorsOutput(evectors);

    eigenCalculator.setEigenvectorsType(ArnUtils::DEVICE);

    eigenCalculator.calculate();

    for (int fa = 0; fa < ecount; ++fa) {
      EXPECT_TRUE(ao.verifyOutput(evectors[fa], reoevalues[fa]));
      debug("reoevalues[%d] = %f", fa, reoevalues[fa]);
    }
  } catch (const std::exception& ex) {
    debugException(ex);
    throw;
  }
}
