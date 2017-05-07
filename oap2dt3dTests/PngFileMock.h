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

#ifndef PNGFILEMOCK_H
#define PNGFILEMOCK_H

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "PngFile.h"

using namespace ::testing;

class PngFileMock : public oap::PngFile {
 public:
  PngFileMock(const oap::pixel_t* matrix, size_t vectorsCount, size_t width,
              size_t height)
      : oap::PngFile(""),
        m_counter(0),
        m_vectorsCount(vectorsCount),
        m_vectorLength(width * height),
        m_width(width),
        m_height(height),
        m_optWidth(width),
        m_optHeight(height) {
    setPixelsMatrix(matrix);
    EXPECT_CALL(*this, openProtected(_)).WillRepeatedly(Return(true));
    EXPECT_CALL(*this, isCorrectFormat()).WillRepeatedly(Return(true));
  }

  PngFileMock(size_t width, size_t height)
      : oap::PngFile(""),
        m_counter(0),
        m_vectorsCount(0),
        m_vectorLength(width * height),
        m_width(width),
        m_height(height),
        m_optWidth(width),
        m_optHeight(height) {
    EXPECT_CALL(*this, openProtected(_)).WillRepeatedly(Return(true));
    EXPECT_CALL(*this, isCorrectFormat()).WillRepeatedly(Return(true));
  }

  virtual ~PngFileMock() {}

  MOCK_METHOD1(openProtected, bool(const std::string&));

  MOCK_METHOD3(read, bool(void*, size_t, size_t));

  MOCK_CONST_METHOD0(isCorrectFormat, bool());

  MOCK_METHOD0(loadBitmapProtected, void());

  MOCK_METHOD0(freeBitmapProtected, void());

  MOCK_CONST_METHOD0(getSufix, std::string());

  MOCK_METHOD0(closeProtected, void());

  virtual oap::OptSize getWidth() const override { return m_width; }

  virtual oap::OptSize getHeight() const override { return m_height; }

  virtual void forceOutputWidth(const oap::OptSize& optSize) override {
    m_optWidth = optSize;
  }

  virtual void forceOutputHeight(const oap::OptSize& optSize) override {
    m_optHeight = optSize;
  }

  virtual oap::OptSize getOutputWidth() const override { return m_optWidth; }

  virtual oap::OptSize getOutputHeight() const override { return m_optHeight; }

  virtual void getPixelsVectorProtected(oap::pixel_t* vector) const override {
    ASSERT_NE(m_counter, m_vectorsCount);

    size_t length = m_optWidth.optSize * m_optHeight.optSize;

    memcpy(vector, &m_matrix[m_counter * length],
           sizeof(oap::pixel_t) * length);

    ++m_counter;
  }

  void setPixelsMatrix(const oap::pixel_t* matrix) { m_matrix = matrix; }

 private:
  const oap::pixel_t* m_matrix;
  mutable size_t m_counter;
  size_t m_vectorsCount;
  size_t m_vectorLength;
  oap::OptSize m_width;
  oap::OptSize m_height;
  oap::OptSize m_optWidth;
  oap::OptSize m_optHeight;
};

using NicePngFileMock = NiceMock<PngFileMock>;

#endif  // PNGFILEMOCK
