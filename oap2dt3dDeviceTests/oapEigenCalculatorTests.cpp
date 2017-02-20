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

using namespace ::testing;

class OapEigenCalculatorTests : public testing::Test {
 public:
  virtual void SetUp() {
    device::Context::Instance().create();
    m_counter = 0;
    m_dataLoader = NULL;
    m_cuMatrix = new CuMatrix();
  }

  virtual void TearDown() {
    delete m_dataLoader;
    delete m_cuMatrix;
    device::Context::Instance().destroy();
  }

  size_t m_counter;
  oap::DeviceDataLoader* m_dataLoader;
  CuMatrix* m_cuMatrix;

  static void multiplyFunc(math::Matrix* m_w, math::Matrix* m_v, void* userData,
                           CuHArnoldi::MultiplicationType mt) {
    if (mt == CuHArnoldi::TYPE_WV) {
      OapEigenCalculatorTests* test =
          static_cast<OapEigenCalculatorTests*>(userData);
      oap::DeviceDataLoader* dataLoader = test->m_dataLoader;
      math::Matrix* vec = dataLoader->createDeviceVector(test->m_counter);
      ++(test->m_counter);

      test->m_cuMatrix->dotProduct(m_w, vec, m_v);

      device::DeleteDeviceMatrix(vec);
    }
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
  EXPECT_THROW(eigenCalc.getEigenvalues(NULL), oap::exceptions::NotInitialzed);
  EXPECT_THROW(eigenCalc.getEigenvectors(NULL), oap::exceptions::NotInitialzed);
}

TEST_F(OapEigenCalculatorTests, Calculate) {
  math::Matrix* matrix = NULL;
  debugLongTest();

  try {
    m_dataLoader =
        oap::DeviceDataLoader::createDataLoader<oap::PngFile,
                                                oap::DeviceDataLoader>(
            "oap2dt3d/data/images_monkey", "image", 1000, true);

    TestCuHArnoldiCallback cuharnoldi;
    cuharnoldi.setCallback(OapEigenCalculatorTests::multiplyFunc, this);

    floatt reoevalues[4];

    oap::EigenCalculator eigenCalculator(&cuharnoldi);

    eigenCalculator.setDataLoader(m_dataLoader);

    eigenCalculator.calculate();

    eigenCalculator.getEigenvalues(reoevalues);

    fprintf(stderr, "reoevalues[0] = %f \n", reoevalues[0]);
    fprintf(stderr, "reoevalues[1] = %f \n", reoevalues[1]);
    fprintf(stderr, "reoevalues[2] = %f \n", reoevalues[2]);
    fprintf(stderr, "reoevalues[3] = %f \n", reoevalues[3]);

  } catch (const oap::exceptions::Exception& ex) {
    delete m_dataLoader;
    m_dataLoader = NULL;

    delete m_cuMatrix;
    m_cuMatrix = NULL;

    debugException(ex);
    throw;
  }
}
