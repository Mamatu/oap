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

  void executeColorTest(const std::string& file, oap::pixel_t expected) {
    oap::PngFile pngFile;
    EXPECT_NO_THROW({
      oap::PngDataLoader pngDataLoader(&pngFile, getImagePath(file));
      const size_t width = pngDataLoader.getWidth();
      const size_t height = pngDataLoader.getHeight();
      oap::pixel_t* pixels = new oap::pixel_t[width * height];
      pngDataLoader.getPixelsVector(pixels);
      for (size_t fa = 0; fa < width; ++fa) {
        for (size_t fb = 0; fb < height; ++fb) {
          oap::pixel_t pixel = pngDataLoader.getPixel(fa, fb);
          EXPECT_EQ(pixel, expected);
          const oap::pixel_t pixel1 = pixels[fa * height + fb];
          EXPECT_EQ(pixel1, expected);
        }
      }
      delete[] pixels;
    });
  }
};

TEST_F(OapPngLoaderTests, LoadGreenScreen) {
  executeColorTest("green.png", 65280);
}

TEST_F(OapPngLoaderTests, LoadRedScreen) {
  executeColorTest("red.png", 16711680);
}

TEST_F(OapPngLoaderTests, LoadBlueScreen) { executeColorTest("blue.png", 255); }
