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
#include "PngFile.h"

#include "Config.h"

using namespace ::testing;

class OapPngLoaderTests : public testing::Test {
 public:
  OapPngLoaderTests() {
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
};

TEST_F(OapPngLoaderTests, LoadGreenScreen) {
  oap::PngFile pngFile;

  // fprintf(stderr, #OAP_PATH);
  EXPECT_NO_THROW({
    oap::PngDataLoader pngDataLoader(&pngFile, getImagePath("green.png"));
    oap::Pixel pixel = pngDataLoader.getPixel(0, 0);
    EXPECT_EQ(pixel.r, 0);
    EXPECT_EQ(pixel.g, 255);
    EXPECT_EQ(pixel.b, 0);
  });
}

TEST_F(OapPngLoaderTests, LoadRedScreen) {
  oap::PngFile pngFile;

  EXPECT_NO_THROW({
    oap::PngDataLoader pngDataLoader(&pngFile, getImagePath("red.png"));
    oap::Pixel pixel = pngDataLoader.getPixel(0, 0);
    EXPECT_EQ(pixel.r, 255);
    EXPECT_EQ(pixel.g, 0);
    EXPECT_EQ(pixel.b, 0);
  });
}

TEST_F(OapPngLoaderTests, LoadBlueScreen) {
  oap::PngFile pngFile;

  EXPECT_NO_THROW({
    oap::PngDataLoader pngDataLoader(&pngFile, getImagePath("blue.png"));
    oap::Pixel pixel = pngDataLoader.getPixel(0, 0);
    EXPECT_EQ(pixel.r, 0);
    EXPECT_EQ(pixel.g, 0);
    EXPECT_EQ(pixel.b, 255);
  });
}
