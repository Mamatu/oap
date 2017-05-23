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
#include "DataLoader.h"
#include "HostMatrixUtils.h"
#include "PngFile.h"

#include "MatchersUtils.h"

#include "Config.h"

using namespace ::testing;

class OapDataLoaderTests : public testing::Test {
 public:
  OapDataLoaderTests() {
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
    oap::PngFile pngFile(getImagePath(file));
    EXPECT_NO_THROW({
      pngFile.open();
      pngFile.loadBitmap();
      pngFile.close();
      const size_t width = pngFile.getWidth().optSize;
      const size_t height = pngFile.getHeight().optSize;
      oap::pixel_t* pixels = new oap::pixel_t[width * height];
      pngFile.getPixelsVector(pixels);
      for (size_t fa = 0; fa < width; ++fa) {
        for (size_t fb = 0; fb < height; ++fb) {
          oap::pixel_t pixel = pngFile.getPixel(fa, fb);
          EXPECT_EQ(pixel, expected);
          const oap::pixel_t pixel1 = pixels[fa * height + fb];
          EXPECT_EQ(pixel1, expected);
        }
      }
      pngFile.freeBitmap();
      delete[] pixels;
    });
  }

  math::Matrix* createMatrix(const std::string& imageName, size_t count,
                             bool frugalMode = true) {
    oap::Images images;

    debugLongTest();

    for (size_t fa = 0; fa < count; ++fa) {
      oap::PngFile* file = new oap::PngFile(getImagePath(imageName));
      images.push_back(file);
    }

    oap::DataLoader dataLoader(images, true, frugalMode);

    math::MatrixInfo matrixInfo = dataLoader.getMatrixInfo();
    math::Matrix* matrix = dataLoader.createMatrix();

    EXPECT_EQ(count, matrixInfo.m_matrixDim.columns);
    EXPECT_EQ(images[0]->getLength(), matrixInfo.m_matrixDim.rows);

    return matrix;
  }
};

TEST_F(OapDataLoaderTests, LoadGreenScreen) {
  executeColorTest("green.png", 65280);
}

TEST_F(OapDataLoaderTests, LoadRedScreen) {
  executeColorTest("red.png", 16711680);
}

TEST_F(OapDataLoaderTests, LoadBlueScreen) {
  executeColorTest("blue.png", 255);
}

TEST_F(OapDataLoaderTests, CreateMatrixFromGreenScreenNoFrugalMode) {
  math::Matrix* matrix = createMatrix("green.png", 900, false);

  floatt expected = oap::Image::convertRgbToFloatt(0, 255, 0);

  EXPECT_THAT(matrix, MatrixHasValues(expected));

  host::DeleteMatrix(matrix);
}

TEST_F(OapDataLoaderTests, CreateMatrixFromGreenScreenFrugalMode) {
  math::Matrix* matrix = createMatrix("green.png", 900, true);

  floatt expected = oap::Image::convertRgbToFloatt(0, 255, 0);

  EXPECT_THAT(matrix, MatrixHasValues(expected));

  host::DeleteMatrix(matrix);
}

TEST_F(OapDataLoaderTests, CreateMatrixFromMonkeyScreen) {
  math::Matrix* matrix = createMatrix("monkey.png", 1000);
  host::DeleteMatrix(matrix);
}

TEST_F(OapDataLoaderTests, LoadMonkeyImagesAndCreateMatrix) {
  oap::DataLoader* dataloader = NULL;
  math::Matrix* matrix = NULL;
  debugLongTest();

  EXPECT_NO_THROW(try {
    dataloader = oap::DataLoader::createDataLoader<oap::PngFile>(
        "oap2dt3d/data/images_monkey", "image", 1000, true);
    matrix = dataloader->createMatrix();
  } catch (const oap::exceptions::Exception& ex) {
    delete dataloader;
    debugException(ex);
    throw;
  });

  host::DeleteMatrix(matrix);

  delete dataloader;
}

TEST_F(OapDataLoaderTests, LoadBlueRecTest) {
  oap::DataLoader* dataloader = NULL;
  EXPECT_NO_THROW(try {
    dataloader = oap::DataLoader::createDataLoader<oap::PngFile>(
        "oap2dt3dFuncTests/data/images/bluerecs", "bluerec", 3);
    delete dataloader;
  } catch (const oap::exceptions::Exception& ex) {
    delete dataloader;
    debugException(ex);
    throw;
  });
}

TEST_F(OapDataLoaderTests, LoadMonkeyImagesCreateMatrix) {
  oap::DataLoader* dataloader = NULL;
  math::Matrix* matrix = NULL;
  debugLongTest();

  EXPECT_NO_THROW(try {
    dataloader = oap::DataLoader::createDataLoader<oap::PngFile>(
        "oap2dt3d/data/images_monkey", "image", 1000, true);
    matrix = dataloader->createMatrix();
  } catch (const oap::exceptions::Exception& ex) {
    delete dataloader;
    debugException(ex);
    throw;
  });

  host::DeleteMatrix(matrix);

  delete dataloader;
}

TEST_F(OapDataLoaderTests, LoadMonkeyImagesCreateRowVectors) {
  oap::DataLoader* dataloader = NULL;
  debugLongTest();

  int count = 100;

  math::Matrix* columnVecs[count];
  math::Matrix* rowVecs[count];

  EXPECT_NO_THROW(try {
    dataloader = oap::DataLoader::createDataLoader<oap::PngFile>(
        "oap2dt3d/data/images_monkey", "image", 1000, true);
    math::MatrixInfo matrixInfo = dataloader->getMatrixInfo();

    uintt columnIndexes[count];
    uintt rowIndexes[count];

    for (int fa = 0; fa < count; ++fa) {
      columnIndexes[fa] = (matrixInfo.m_matrixDim.columns * fa) / count;
      rowIndexes[fa] = (matrixInfo.m_matrixDim.rows * fa) / count;

      columnVecs[fa] = dataloader->createColumnVector(columnIndexes[fa]);
      rowVecs[fa] = dataloader->createRowVector(rowIndexes[fa]);
    }

    for (int fa = 0; fa < count - 1; ++fa) {
      EXPECT_THAT(columnVecs[fa], Not(MatrixIsEqual(columnVecs[fa + 1])))
          << "Actual: Columns vectors are equal: " << columnIndexes[fa] << ", "
          << columnIndexes[fa + 1];
    }

    for (int fa = 0; fa < count; ++fa) {
      EXPECT_EQ(GetRe(columnVecs[fa], 0, rowIndexes[fa]),
                GetRe(rowVecs[fa], columnIndexes[fa], 0));
    }
  } catch (const oap::exceptions::Exception& ex) {
    delete dataloader;
    debugException(ex);
    throw;
  });

  for (int fa = 0; fa < count; ++fa) {
    host::DeleteMatrix(columnVecs[fa]);
    host::DeleteMatrix(rowVecs[fa]);
  }

  delete dataloader;
}

TEST_F(OapDataLoaderTests, DataLoaderSaveTruncatedImagesTest) {
  math::Matrix* matrix = NULL;
  debugLongTest();

  std::string dir = "/tmp/tests_data";

  debug("Images will be saved in %s", dir.c_str());

  class DataLoaderImpl : public oap::DataLoader {
   public:
    DataLoaderImpl(const oap::Images& images, bool dealocateImages = false,
                   bool frugalMode = true)
        : oap::DataLoader(images, dealocateImages, frugalMode) {}

    size_t getImagesCount() const { return oap::DataLoader::getImagesCount(); }

    oap::Image* getImage(size_t index) const {
      return oap::DataLoader::getImage(index);
    }
  };

  DataLoaderImpl* dataloader = NULL;

  try {
    dataloader =
        oap::DataLoader::createDataLoader<oap::PngFile, DataLoaderImpl>(
            "oap2dt3d/data/images_monkey", "image", 1000, true);

    for (size_t fa = 0; fa < dataloader->getImagesCount(); ++fa) {
      oap::Image* image = dataloader->getImage(fa);
      oap::PngFile* pngFile = dynamic_cast<oap::PngFile*>(image);
      EXPECT_TRUE(pngFile->save("truncated_", dir))
          << "File can not be saved in " << dir
          << " . If this directory doesn't exist please create it.";
    }

  } catch (const oap::exceptions::Exception& ex) {
    delete dataloader;
    debugException(ex);
    throw;
  }

  delete dataloader;
}
