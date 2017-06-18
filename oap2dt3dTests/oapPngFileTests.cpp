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
#include "Exceptions.h"

using namespace ::testing;

class OapPngFileTests : public testing::Test {
 public:
  virtual void SetUp() {}

  virtual void TearDown() {}

 public:
  class PngFileMock : public oap::PngFile {
   public:
    PngFileMock(const std::string& path) : oap::PngFile(path) {}

    virtual ~PngFileMock() {}

    MOCK_METHOD1(openProtected, bool(const std::string&));

    MOCK_METHOD3(read, bool(void*, size_t, size_t));

    MOCK_CONST_METHOD0(isCorrectFormat, bool());

    MOCK_METHOD0(loadBitmap, void());

    MOCK_METHOD0(freeBitmap, void());

    MOCK_METHOD0(closeProtected, void());

    MOCK_CONST_METHOD0(getWidth, oap::OptSize());

    MOCK_CONST_METHOD0(getHeight, oap::OptSize());

    MOCK_CONST_METHOD1(getPixelsVector, void(oap::pixel_t*));

    MOCK_CONST_METHOD0(getSufix, std::string());
  };
};

TEST_F(OapPngFileTests, LoadPixelOutOfRange) {
  NiceMock<PngFileMock> pngFileMock("test_path/test_file");

  ON_CALL(pngFileMock, getWidth()).WillByDefault(Return(1024));

  ON_CALL(pngFileMock, getHeight()).WillByDefault(Return(760));

  EXPECT_THROW(pngFileMock.getPixel(2000, 2000), oap::exceptions::OutOfRange);
}

TEST_F(OapPngFileTests, ConvertRGBTest) {
  oap::pixel_t pixel = oap::PngFile::convertRgbToPixel(100, 100, 100);
  EXPECT_EQ(pixel, 6579300);
}

TEST_F(OapPngFileTests, ConvertRGBTest1) {
  oap::pixel_t pixel = oap::PngFile::convertRgbToPixel(25, 89, 42);
  EXPECT_EQ(pixel, 1661226);
}

TEST_F(OapPngFileTests, ConvertRGBTest2) {
  oap::pixel_t pixel1 = oap::PngFile::convertRgbToPixel(25, 89, 42);
  oap::pixel_t pixel2 = oap::PngFile::convertRgbToPixel(100, 100, 100);
  floatt color1 = oap::PngFile::convertRgbToFloatt(25, 89, 42);
  floatt color2 = oap::PngFile::convertRgbToFloatt(100, 100, 100);
  EXPECT_NE(color1, color2);
  EXPECT_NE(pixel1, pixel2);
  EXPECT_THAT(static_cast<floatt>(pixel1) / static_cast<floatt>(pixel2),
              DoubleEq(color1 / color2));
}

TEST_F(OapPngFileTests, FileNametest) {
  oap::PngFile pngFile("a/b/c/d/e");
  std::string fileName = pngFile.getFileName();
  std::string filePath = pngFile.getFilePath();
  EXPECT_EQ("e", fileName);
  EXPECT_EQ("a/b/c/d", filePath);
}
