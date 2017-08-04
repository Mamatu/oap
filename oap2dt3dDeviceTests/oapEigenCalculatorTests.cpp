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

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "PngFile.h"
#include "EigenCalculator.h"
#include "DeviceDataLoader.h"
#include "Exceptions.h"
#include "MatchersUtils.h"

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

  bool verifyOutput(math::Matrix* vector, floatt value, oap::EigenCalculator* eigenCalc) {
    math::Matrix* matrix = m_dataLoader->createMatrix();

    const uintt partSize = matrix->columns;

    math::Matrix* dvector = NULL;
    uintt vectorrows = 0;

    bool dvectorIsCopy = false;

    vectorrows = matrix->columns;

    if (eigenCalc->getEigenvectorsType() == ArnUtils::HOST) {
      dvector = device::NewDeviceMatrixCopy(vector);
      dvectorIsCopy = true;
    } else if (eigenCalc->getEigenvectorsType() == ArnUtils::DEVICE) {
      dvector = vector;
      dvectorIsCopy = false;
    }

    if (dvector == NULL)  {
      debugAssert("Invalid eigenvectors type.");
    }

    math::Matrix* refMatrix =
        host::NewMatrix(matrix, matrix->columns, partSize);

    host::CopyMatrix(refMatrix, matrix);

    math::Matrix* drefMatrix = device::NewDeviceMatrixCopy(refMatrix);

    math::MatrixInfo info = host::GetMatrixInfo(refMatrix);

    const uintt matrixcolumns = info.m_matrixDim.columns;
    uintt matrixrows = info.m_matrixDim.rows;

    matrixrows = partSize;


    math::Matrix* matrix1 =
        device::NewDeviceReMatrix(matrixrows, matrixcolumns);
    math::Matrix* leftMatrix =
        device::NewDeviceReMatrix(matrixrows, matrixrows);

    math::Matrix* rightMatrix =
        device::NewDeviceReMatrix(matrixrows, matrixrows);

    math::Matrix* vectorT = device::NewDeviceReMatrix(vectorrows, 1);

    m_cuMatrix.transposeMatrix(matrix1, drefMatrix);
    m_cuMatrix.transposeMatrix(vectorT, dvector);
    m_cuMatrix.dotProduct(leftMatrix, drefMatrix, matrix1);

    host::PrintMatrix("matrix =", matrix);
    host::PrintMatrix("refMatrix =", refMatrix);
    device::PrintMatrix("matrix1 =", matrix1);
    device::PrintMatrix("drefMatrix =", drefMatrix);
    device::PrintMatrix("vectorT =", vectorT);
    device::PrintMatrix("vector =", vector);
    floatt value2 = value * value;
    m_cuMatrix.multiplyConstantMatrix(vectorT, vectorT, value2);
    m_cuMatrix.dotProduct(rightMatrix, dvector, vectorT);
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

    if (dvectorIsCopy) {
      device::DeleteDeviceMatrix(dvector);
    }

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


class MatricesDeleter {
  int m_eigensCount;
  ArnUtils::Type m_type;
  public:
    MatricesDeleter(int eigensCount, ArnUtils::Type type) : 
      m_eigensCount(eigensCount), m_type(type) {}

    MatricesDeleter& operator() (math::Matrix** evectors) {
      for (int fa = 0; fa < m_eigensCount; ++fa) {
        debug("Deleted matrix %p", evectors[fa]);
        if (m_type == ArnUtils::HOST) {
          host::DeleteMatrix(evectors[fa]);
        } else if (m_type == ArnUtils::DEVICE) {
          device::DeleteDeviceMatrix(evectors[fa]);
        }
      }
      delete[] evectors;
      return *this;
    }

};

using MatricesUPtr = std::unique_ptr<math::Matrix*, MatricesDeleter>;

class TestCuHArnoldiCallback : public CuHArnoldiCallback {
 public:
  TestCuHArnoldiCallback(ArnoldiOperations* ao) : m_ao(ao) {}

  bool checkEigenspair(floatt value, math::Matrix* vector, uint index) {
    static int counter = 0;
    ++counter;
    debug("counter = %d", counter);
    return counter < 4;
  }

  static MatricesUPtr launchTest(ArnUtils::Type eigensType, int ecount) {
    std::unique_ptr<oap::DeviceDataLoader> dataLoader(
        oap::DeviceDataLoader::createDataLoader<oap::PngFile,
                                                oap::DeviceDataLoader>(
            "oap2dt3d/data/images_monkey", "image", 1000, true));

    ArnoldiOperations ao(dataLoader.get());
    TestCuHArnoldiCallback cuharnoldi(&ao);
    cuharnoldi.setCallback(ArnoldiOperations::multiplyFunc, &ao);

    floatt reoevalues[ecount];

    oap::EigenCalculator eigenCalculator(&cuharnoldi);

    eigenCalculator.setDataLoader(dataLoader.get());

    eigenCalculator.setEigensCount(ecount);

    math::MatrixInfo matrixInfo = eigenCalculator.getMatrixInfo();

    MatricesDeleter matricesDeleter(ecount, eigensType);

    MatricesUPtr evectorsUPtr(new math::Matrix* [ecount], matricesDeleter);

    auto matricesInitializer = [&evectorsUPtr, ecount, &matrixInfo, eigensType]() {
      math::Matrix** evectors = evectorsUPtr.get();
      const uintt rows = matrixInfo.m_matrixDim.rows;
      for (int fa = 0; fa < ecount; ++fa) {
        if (eigensType == ArnUtils::HOST) {
          evectors[fa] = host::NewReMatrix(1, rows);
        } else if (eigensType == ArnUtils::DEVICE) {
          evectors[fa] = device::NewDeviceReMatrix(1, rows);
        }
        debug("Created matrix %p", evectors[fa]);
      }
    };

    matricesInitializer();

    math::Matrix** evectors = evectorsUPtr.get();

    eigenCalculator.setEigenvaluesOutput(reoevalues);

    eigenCalculator.setEigenvectorsOutput(evectors, eigensType);

    eigenCalculator.calculate();

    for (int fa = 0; fa < ecount; ++fa) {
      EXPECT_TRUE(ao.verifyOutput(evectors[fa], reoevalues[fa], &eigenCalculator));
      debug("reoevalues[%d] = %f", fa, reoevalues[fa]);
    }
    return std::move(evectorsUPtr);
  }

 private:
  ArnoldiOperations* m_ao;
};

TEST_F(OapEigenCalculatorTests, NotInitializedTest) {
  TestCuHArnoldiCallback cuharnoldi(nullptr);
  oap::EigenCalculator eigenCalc(&cuharnoldi);
  EXPECT_THROW(eigenCalc.calculate(), oap::exceptions::NotInitialzed);
}

TEST_F(OapEigenCalculatorTests, CalculateDeviceOutput) {
  debugLongTest();

  try {
    int ecount = 6;  
    MatricesUPtr deviceEVectors = TestCuHArnoldiCallback::launchTest(ArnUtils::DEVICE, ecount);
    MatricesUPtr hostEVectors = TestCuHArnoldiCallback::launchTest(ArnUtils::HOST, ecount);
    math::Matrix** deviceMatrices = deviceEVectors.get();
    math::Matrix** hostMatrices = hostEVectors.get();
    math::Matrix* hostMatrix = host::NewMatrix(hostEVectors.get()[0]);
    for (int fa = 0; fa < ecount; ++fa) {
      device::CopyDeviceMatrixToHostMatrix(hostMatrix, deviceMatrices[fa]);
      EXPECT_THAT(hostMatrices[fa], MatrixIsEqual(hostMatrix));
    }
    host::DeleteMatrix(hostMatrix);
  } catch (const std::exception& ex) {
    debugException(ex);
    throw;
  }
}
