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
#include "PngDataLoader.h"
#include "EigenCalculator.h"
#include "HostMatrixModules.h"
#include "MatrixAPI.h"
#include "PngFile.h"

#include "MatchersUtils.h"

#include "Config.h"

using namespace ::testing;

class OapEigenCalculatorTests : public testing::Test {
 public:
  OapEigenCalculatorTests() {
    m_data_path = utils::Config::getPathInOap("oap2dt3dFuncTests/data/");
    m_images_path = m_data_path + "images/";
  }

  virtual void SetUp() {}

  virtual void TearDown() {}

  std::string m_data_path;
  std::string m_images_path;

  std::string getImagePath(const std::string& filename) {
    return m_images_path + filename;
  }

  math::Matrix* createMatrix(const std::string& imageName, size_t count) {
    size_t pngDataLoaderCount = count;
    std::vector<oap::PngDataLoader*> pdlsVec;
    oap::EigenCalculator eigenCalc;

    debugLongTest();

    for (size_t fa = 0; fa < pngDataLoaderCount; ++fa) {
      oap::PngDataLoader* pngDataLoader =
          new oap::PngDataLoader(getImagePath(imageName));
      pdlsVec.push_back(pngDataLoader);
    }

    for (size_t fa = 0; fa < pngDataLoaderCount; ++fa) {
      eigenCalc.addPngDataLoader(pdlsVec[fa]);
    }

    ArnUtils::MatrixInfo matrixInfo = eigenCalc.createMatrixInfo();
    math::Matrix* matrix = eigenCalc.createMatrix();

    EXPECT_EQ(pngDataLoaderCount, matrixInfo.m_matrixDim.columns);
    EXPECT_EQ(pdlsVec[0]->getLength(), matrixInfo.m_matrixDim.rows);

    for (size_t fa = 0; fa < pngDataLoaderCount; ++fa) {
      delete pdlsVec[fa];
    }

    return matrix;
  }
};

TEST_F(OapEigenCalculatorTests, CreateMatrixFromGreenScreen) {
  math::Matrix* matrix = createMatrix("green.png", 900);

  floatt expected = oap::IPngFile::convertRgbToFloatt(0, 255, 0);

  EXPECT_THAT(matrix, MatrixValuesAreEqual(expected));

  host::DeleteMatrix(matrix);
}

TEST_F(OapEigenCalculatorTests, CreateMatrixFromMonkeyScreen) {
  math::Matrix* matrix = createMatrix("monkey.png", 700);
  host::DeleteMatrix(matrix);
}
