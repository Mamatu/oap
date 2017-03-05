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
  size_t m_counter;
  oap::DeviceDataLoader* m_dataLoader;
  CuMatrix m_cuMatrix;

  ArnoldiOperations(oap::DeviceDataLoader* dataLoader)
      : m_counter(0), m_dataLoader(dataLoader) {}

  static void multiplyFunc(math::Matrix* m_w, math::Matrix* m_v, void* userData,
                           CuHArnoldi::MultiplicationType mt) {
    if (mt == CuHArnoldi::TYPE_WV) {
      ArnoldiOperations* ao = static_cast<ArnoldiOperations*>(userData);
      oap::DeviceDataLoader* dataLoader = ao->m_dataLoader;
      math::Matrix* vec = dataLoader->createDeviceVector(ao->m_counter);
      ++(ao->m_counter);

      ao->m_cuMatrix.dotProduct(m_w, vec, m_v);

      device::DeleteDeviceMatrix(vec);
    }
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

  bool verifyOutput(math::Matrix* vector, floatt value, ArnoldiOperations& ao) {
    math::Matrix* matrix = ao.createDeviceMatrix();
    math::Matrix* matrix1 =
        device::NewDeviceReMatrix(matrix->rows, matrix->columns);
    math::Matrix* leftMatrix =
        device::NewDeviceReMatrix(matrix->rows, matrix->rows);

    math::Matrix* rightMatrix =
        device::NewDeviceReMatrix(matrix->rows, matrix->rows);

    math::Matrix* vector1 = device::NewDeviceReMatrix(vector->rows, 1);

    ao->transposeMatrix(matrix1, matrix);
    ao->transposeMatrix(vector1, vector);
    ao->dotProduct(leftMatrix, matrix, matrix1);

    floatt value2 = value * value;
    ao->multiplyConstantMatrix(vector1, vector1, value2);
    ao->dotProduct(rightMatrix, vector, vector1);
    bool output = ao->compare(leftMatrix, rightMatrix);

    device::DeleteDeviceMatrix(matrix1);
    device::DeleteDeviceMatrix(leftMatrix);
    device::DeleteDeviceMatrix(rightMatrix);

    device::DeleteDeviceMatrix(vector1);
    return output;
  }
};

class TestCuHArnoldiCallback : public CuHArnoldiCallback {
 public:
  bool checkEigenvector(math::Matrix* vector, uint index) { return true; }
};

TEST_F(OapEigenCalculatorTests, NotInitializedTest) {
  TestCuHArnoldiCallback cuharnoldi;
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

    TestCuHArnoldiCallback cuharnoldi;
    ArnoldiOperations data(dataLoader.get());
    cuharnoldi.setCallback(ArnoldiOperations::multiplyFunc, &data);

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

    debug("reoevalues[0] = %f", reoevalues[0]);
    debug("reoevalues[1] = %f", reoevalues[1]);
    debug("reoevalues[2] = %f", reoevalues[2]);

  } catch (const oap::exceptions::Exception& ex) {
    debugException(ex);
    throw;
  }
}
