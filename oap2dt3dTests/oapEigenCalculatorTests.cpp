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
#include "Matrix.h"
#include "MatrixAPI.h"
#include "DeviceMatrixModules.h"
#include "Exceptions.h"

using namespace ::testing;

class OapEigenCalculatorTests : public testing::Test {
 public:
  virtual void SetUp() {}

  virtual void TearDown() {}

 public:
  class PngFileMock : public oap::PngFile {
   public:
    PngFileMock(oap::pixel_t* matrix, size_t vectorsCount, size_t width,
                size_t height)
        : m_counter(0),
          m_vectorsCount(vectorsCount),
          m_vectorLength(width * height),
          m_width(width),
          m_height(height) {
      setPixelsMatrix(matrix);
    }

    virtual ~PngFileMock() {}

    MOCK_METHOD1(openInternal, bool(const char*));

    MOCK_METHOD3(read, bool(void*, size_t, size_t));

    MOCK_CONST_METHOD0(isPngInternal, bool());

    MOCK_METHOD0(loadBitmap, void());

    MOCK_METHOD0(freeBitmap, void());

    MOCK_METHOD0(close, void());

    size_t getWidth() const { return m_width; }

    size_t getHeight() const { return m_height; }

    void getPixelsVector(oap::pixel_t* vector) const {
      ASSERT_NE(m_counter, m_vectorsCount);

      memcpy(vector, &m_matrix[m_counter * m_vectorLength],
             sizeof(oap::pixel_t) * m_vectorLength);

      ++m_counter;
    }

    void setPixelsMatrix(oap::pixel_t* matrix) { m_matrix = matrix; }

   private:
    oap::pixel_t* m_matrix;
    mutable size_t m_counter;
    size_t m_vectorsCount;
    size_t m_vectorLength;
    size_t m_width;
    size_t m_height;
  };
};

TEST_F(OapEigenCalculatorTests, NotInitializedTest) {
  oap::EigenCalculator eigenCalc;
  EXPECT_THROW(eigenCalc.createMatrix(), oap::exceptions::NotInitialzed);
  EXPECT_THROW(eigenCalc.createMatrixInfo(), oap::exceptions::NotInitialzed);
  EXPECT_THROW(eigenCalc.calculate(), oap::exceptions::NotInitialzed);
  EXPECT_THROW(eigenCalc.getEigenvalue(0), oap::exceptions::NotInitialzed);
  EXPECT_THROW(eigenCalc.getEigenvector(0), oap::exceptions::NotInitialzed);
}

TEST_F(OapEigenCalculatorTests, Matrix4x4FromImage) {
  const oap::pixel_t pv = oap::Image::getPixelMax();
  oap::pixel_t pixels[16] = {pv, pv, pv, pv, pv, pv, pv, pv,
                             pv, pv, pv, pv, pv, pv, pv, pv};

  PngFileMock pngFileMock(pixels, 4, 2, 2);

  oap::DataLoader pdl1(&pngFileMock);
  oap::DataLoader pdl2(&pngFileMock);
  oap::DataLoader pdl3(&pngFileMock);
  oap::DataLoader pdl4(&pngFileMock);

  oap::EigenCalculator eigenCalc;

  eigenCalc.addPngDataLoader(&pdl1);
  eigenCalc.addPngDataLoader(&pdl2);
  eigenCalc.addPngDataLoader(&pdl3);
  eigenCalc.addPngDataLoader(&pdl4);

  math::Matrix* matrix = eigenCalc.createMatrix();

  EXPECT_EQ(4, matrix->columns);
  EXPECT_EQ(4, matrix->rows);

  for (size_t fa = 0; fa < matrix->columns; ++fa) {
    for (size_t fb = 0; fb < matrix->rows; ++fb) {
      const floatt value = GetRe(matrix, fa, fb);
      EXPECT_EQ(1, value);
    }
  }
  host::DeleteMatrix(matrix);
}
