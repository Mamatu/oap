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
#include "DataLoader.h"
#include "DeviceMatrixModules.h"
#include "Image.h"
#include "Matrix.h"
#include "MatrixAPI.h"
#include "PngFile.h"

using namespace ::testing;

class OapDataLoaderTests : public testing::Test {
 public:
  virtual void SetUp() {}

  virtual void TearDown() {}

 public:
  class ImageMock : public oap::Image {
   public:
    ImageMock(const std::string& path) : oap::Image(path) {}

    virtual ~ImageMock() {}

    MOCK_METHOD1(openInternal, bool(const std::string& path));

    MOCK_METHOD3(read, bool(void*, size_t, size_t));

    MOCK_CONST_METHOD0(isCorrectFormat, bool());

    MOCK_METHOD0(loadBitmap, void());

    MOCK_METHOD0(freeBitmap, void());

    MOCK_METHOD0(close, void());

    MOCK_CONST_METHOD0(getWidth, size_t());

    MOCK_CONST_METHOD0(getHeight, size_t());

    MOCK_CONST_METHOD0(getSufix, std::string());

    MOCK_CONST_METHOD2(getPixelInternal,
                       oap::pixel_t(unsigned int, unsigned int));

    MOCK_CONST_METHOD1(getPixelsVector, void(oap::pixel_t*));
  };

  class PngFileMock : public oap::PngFile {
   public:
    PngFileMock(oap::pixel_t* matrix, size_t vectorsCount, size_t width,
                size_t height)
        : oap::PngFile(""),
          m_counter(0),
          m_vectorsCount(vectorsCount),
          m_vectorLength(width * height),
          m_width(width),
          m_height(height) {
      setPixelsMatrix(matrix);
      EXPECT_CALL(*this, openInternal(_)).WillRepeatedly(Return(true));
      EXPECT_CALL(*this, isCorrectFormat()).WillRepeatedly(Return(true));
      EXPECT_CALL(*this, loadBitmap());
      EXPECT_CALL(*this, freeBitmap());
      EXPECT_CALL(*this, close());
    }

    virtual ~PngFileMock() {}

    MOCK_METHOD1(openInternal, bool(const std::string&));

    MOCK_METHOD3(read, bool(void*, size_t, size_t));

    MOCK_CONST_METHOD0(isCorrectFormat, bool());

    MOCK_METHOD0(loadBitmap, void());

    MOCK_METHOD0(freeBitmap, void());

    MOCK_CONST_METHOD0(getSufix, std::string());

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

  class DataLoaderProxy : public oap::DataLoader {
   public:
    static oap::Images m_emptyImages;
    DataLoaderProxy() : DataLoader(m_emptyImages) {}

    static std::string constructAbsPath(const std::string& basePath) {
      return oap::DataLoader::constructAbsPath(basePath);
    }

    static std::string constructImagePath(const std::string& absPath,
                                          const std::string& nameBase,
                                          size_t index, size_t count) {
      return oap::DataLoader::constructImagePath(absPath, nameBase, index,
                                                 count);
    }
  };
};

oap::Images OapDataLoaderTests::DataLoaderProxy::m_emptyImages;

TEST_F(OapDataLoaderTests, LoadFail) {
  ImageMock imageMock("");

  EXPECT_CALL(imageMock, openInternal(_)).Times(1).WillOnce(Return(false));

  oap::Images images;
  images.push_back(&imageMock);

  EXPECT_THROW(oap::DataLoader pngDataLoader(images),
               oap::exceptions::FileNotExist);
}

TEST_F(OapDataLoaderTests, VerificationFail) {
  ImageMock imageMock("");

  EXPECT_CALL(imageMock, openInternal(_)).Times(1).WillOnce(Return(true));

  EXPECT_CALL(imageMock, isCorrectFormat()).Times(1).WillOnce(Return(false));

  oap::Images images;
  images.push_back(&imageMock);

  EXPECT_THROW(oap::DataLoader dataLoader(images),
               oap::exceptions::NotCorrectFormat);
}

TEST_F(OapDataLoaderTests, Load) {
  ImageMock imageMock("");

  EXPECT_CALL(imageMock, openInternal(_)).Times(1).WillOnce(Return(true));

  EXPECT_CALL(imageMock, isCorrectFormat()).Times(1).WillOnce(Return(true));

  EXPECT_CALL(imageMock, loadBitmap()).Times(1);

  EXPECT_CALL(imageMock, freeBitmap()).Times(1);

  ON_CALL(imageMock, getWidth()).WillByDefault(Return(1024));

  ON_CALL(imageMock, getHeight()).WillByDefault(Return(760));

  EXPECT_CALL(imageMock, close()).Times(1);

  oap::Images images;
  images.push_back(&imageMock);

  EXPECT_NO_THROW(oap::DataLoader dataLoader(images));
}

TEST_F(OapDataLoaderTests, Matrix4x4FromImage) {
  const oap::pixel_t pv = oap::Image::getPixelMax();
  oap::pixel_t pixels[16] = {pv, pv, pv, pv, pv, pv, pv, pv,
                             pv, pv, pv, pv, pv, pv, pv, pv};

  oap::Images images;
  images.push_back(new PngFileMock(pixels, 4, 2, 2));
  images.push_back(new PngFileMock(pixels, 4, 2, 2));
  images.push_back(new PngFileMock(pixels, 4, 2, 2));
  images.push_back(new PngFileMock(pixels, 4, 2, 2));

  oap::DataLoader dataLoader(images, true);

  math::Matrix* matrix = dataLoader.createMatrix();

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

TEST_F(OapDataLoaderTests, ContructAbsPathTest) {
  DataLoaderProxy dataLoaderProxy;
}

TEST_F(OapDataLoaderTests, ContructImagePathTest) {
  std::string path =
      DataLoaderProxy::constructImagePath("abs/", "image", 0, 1000);

  std::string path1 =
      DataLoaderProxy::constructImagePath("abs/", "image", 0, 100);

  std::string path2 =
      DataLoaderProxy::constructImagePath("abs/", "image", 0, 10);

  std::string path3 =
      DataLoaderProxy::constructImagePath("abs/", "image", 0, 1);

  std::string path4 =
      DataLoaderProxy::constructImagePath("abs/", "image", 1, 10000);

  EXPECT_EQ("abs/image001", path);
  EXPECT_EQ("abs/image01", path1);
  EXPECT_EQ("abs/image1", path2);
  EXPECT_EQ("abs/image1", path3);
  EXPECT_EQ("abs/image0002", path4);
}
