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
#include "Image.h"
#include "Matrix.h"
#include "HostMatrixModules.h"
#include "MatrixAPI.h"
#include "PngFile.h"

using namespace ::testing;

class OapDataLoaderTests : public testing::Test {
 public:
  virtual void SetUp() {}

  virtual void TearDown() {}

 public:
  class ImageMock : public oap::Image {
   public:  // ImageMock type
    ImageMock(const std::string& path) : oap::Image(path) {}

    virtual ~ImageMock() {}

    MOCK_METHOD1(openInternal, bool(const std::string& path));

    MOCK_METHOD3(read, bool(void*, size_t, size_t));

    MOCK_CONST_METHOD0(isCorrectFormat, bool());

    MOCK_METHOD0(loadBitmap, void());

    MOCK_METHOD0(freeBitmap, void());

    MOCK_METHOD0(close, void());

    MOCK_CONST_METHOD0(getWidth, oap::OptSize());

    MOCK_CONST_METHOD0(getHeight, oap::OptSize());

    MOCK_METHOD1(setOptWidth, void(const oap::OptSize& optWidth));

    MOCK_METHOD1(setOptHeight, void(const oap::OptSize& optHeight));

    MOCK_CONST_METHOD0(getSufix, std::string());

    MOCK_CONST_METHOD2(getPixelInternal,
                       oap::pixel_t(unsigned int, unsigned int));

    MOCK_CONST_METHOD1(getPixelsVector, void(oap::pixel_t*));
  };

 public:  // PngFileMock type
  class PngFileMock : public oap::PngFile {
   public:
    PngFileMock(const oap::pixel_t* matrix, size_t vectorsCount, size_t width,
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
    }

    PngFileMock(size_t width, size_t height)
        : oap::PngFile(""),
          m_counter(0),
          m_vectorsCount(0),
          m_vectorLength(width * height),
          m_width(width),
          m_height(height) {
      EXPECT_CALL(*this, openInternal(_)).WillRepeatedly(Return(true));
      EXPECT_CALL(*this, isCorrectFormat()).WillRepeatedly(Return(true));
    }

    virtual ~PngFileMock() {}

    MOCK_METHOD1(openInternal, bool(const std::string&));

    MOCK_METHOD3(read, bool(void*, size_t, size_t));

    MOCK_CONST_METHOD0(isCorrectFormat, bool());

    MOCK_METHOD0(loadBitmap, void());

    MOCK_METHOD0(freeBitmap, void());

    MOCK_CONST_METHOD0(getSufix, std::string());

    MOCK_METHOD0(close, void());

    virtual oap::OptSize getWidth() const { return m_width; }

    virtual oap::OptSize getHeight() const { return m_height; }

    virtual void setOptWidth(const oap::OptSize& optWidth) {
      m_width = optWidth;
    }

    virtual void setOptHeight(const oap::OptSize& optHeight) {
      m_height = optHeight;
    }

    void getPixelsVector(oap::pixel_t* vector) const {
      ASSERT_NE(m_counter, m_vectorsCount);

      memcpy(vector, &m_matrix[m_counter * m_vectorLength],
             sizeof(oap::pixel_t) * m_vectorLength);

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
  };

 public:  // DataLoaderProxy type
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
  NiceMock<ImageMock> imageMock("");

  EXPECT_CALL(imageMock, openInternal(_)).Times(1).WillOnce(Return(false));

  oap::Images images;
  images.push_back(&imageMock);

  EXPECT_THROW(oap::DataLoader pngDataLoader(images),
               oap::exceptions::FileNotExist);
}

TEST_F(OapDataLoaderTests, FailVerification) {
  NiceMock<ImageMock> imageMock("");

  EXPECT_CALL(imageMock, openInternal(_)).Times(1).WillOnce(Return(true));

  EXPECT_CALL(imageMock, isCorrectFormat()).Times(1).WillOnce(Return(false));

  oap::Images images;
  images.push_back(&imageMock);

  EXPECT_THROW(oap::DataLoader dataLoader(images),
               oap::exceptions::NotCorrectFormat);
}

TEST_F(OapDataLoaderTests, Load) {
  NiceMock<ImageMock> imageMock("");

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
  try {
    oap::Images images;
    images.push_back(new NiceMock<PngFileMock>(pixels, 4, 2, 2));
    images.push_back(new NiceMock<PngFileMock>(pixels, 4, 2, 2));
    images.push_back(new NiceMock<PngFileMock>(pixels, 4, 2, 2));
    images.push_back(new NiceMock<PngFileMock>(pixels, 4, 2, 2));

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
  } catch (const oap::exceptions::Exception& ex) {
    debugException(ex);
  }
}

TEST_F(OapDataLoaderTests, ContructAbsPathTest) {
  DataLoaderProxy dataLoaderProxy;
}

TEST_F(OapDataLoaderTests, ContructImagePathTest) {
  EXPECT_EQ("abs/image000",
            DataLoaderProxy::constructImagePath("abs/", "image", 0, 1000));
  EXPECT_EQ("abs/image00",
            DataLoaderProxy::constructImagePath("abs/", "image", 0, 100));
  EXPECT_EQ("abs/image0",
            DataLoaderProxy::constructImagePath("abs/", "image", 0, 10));
  EXPECT_EQ("abs/image0",
            DataLoaderProxy::constructImagePath("abs/", "image", 0, 1));
  EXPECT_EQ("abs/image001",
            DataLoaderProxy::constructImagePath("abs/", "image", 1, 1000));
  EXPECT_EQ("abs/image01",
            DataLoaderProxy::constructImagePath("abs/", "image", 1, 100));
  EXPECT_EQ("abs/image1",
            DataLoaderProxy::constructImagePath("abs/", "image", 1, 10));
  EXPECT_EQ("abs/image1",
            DataLoaderProxy::constructImagePath("abs/", "image", 1, 1));
  EXPECT_EQ("abs/image0002",
            DataLoaderProxy::constructImagePath("abs/", "image", 2, 10000));
}

class HaveSizesMatcher : public MatcherInterface<const oap::Images&> {
 public:
  HaveSizesMatcher(const oap::OptSize& optWidth, const oap::OptSize& optHeight)
      : m_optWidth(optWidth), m_optHeight(optHeight) {}

  virtual bool MatchAndExplain(const oap::Images& images,
                               MatchResultListener* listener) const {
    bool hasWidth = true;
    bool hasHeight = true;

    *listener << "{";

    for (oap::Images::const_iterator it = images.begin(); it != images.end();
         ++it) {
      if ((*it)->getWidth().optSize != m_optWidth.optSize) {
        hasWidth = false;
      }

      if ((*it)->getHeight().optSize != m_optWidth.optSize) {
        hasHeight = false;
      }

      *listener << "{" << (*it)->getWidth().optSize << ", "
                << (*it)->getHeight().optSize << "}";
    }

    *listener << "}";

    return hasWidth && hasHeight;
  }

  virtual void DescribeTo(::std::ostream* os) const {
    *os << "have sizes: " << m_optWidth.optSize << " " << m_optHeight.optSize;
  }

  virtual void DescribeNegationTo(::std::ostream* os) const {
    *os << "have not sizes: " << m_optWidth.optSize << " "
        << m_optHeight.optSize;
  }

 private:
  oap::OptSize m_optWidth;
  oap::OptSize m_optHeight;
};

inline Matcher<const oap::Images&> HaveSizes(const oap::OptSize& optWidth,
                                             const oap::OptSize& optHeight) {
  return MakeMatcher(new HaveSizesMatcher(optWidth, optHeight));
}

TEST_F(OapDataLoaderTests, LoadProcessTest) {
  NiceMock<PngFileMock> pfm1(10, 10);
  NiceMock<PngFileMock> pfm2(20, 20);
  NiceMock<PngFileMock> pfm3(30, 30);

  EXPECT_CALL(pfm1, loadBitmap()).Times(3);
  EXPECT_CALL(pfm2, loadBitmap()).Times(2);
  EXPECT_CALL(pfm3, loadBitmap()).Times(1);

  oap::Images images = {&pfm1, &pfm2, &pfm3};
  oap::DataLoader dataLoader(images);

  EXPECT_THAT(images, HaveSizes(30, 30));
}

TEST_F(OapDataLoaderTests, LoadProcessTest1) {
  NiceMock<PngFileMock> pfm1(10, 10);
  NiceMock<PngFileMock> pfm2(20, 20);
  NiceMock<PngFileMock> pfm3(30, 30);
  NiceMock<PngFileMock> pfm4(30, 30);
  NiceMock<PngFileMock> pfm5(30, 30);
  NiceMock<PngFileMock> pfm6(50, 50);

  EXPECT_CALL(pfm1, loadBitmap()).Times(4);
  EXPECT_CALL(pfm2, loadBitmap()).Times(3);
  EXPECT_CALL(pfm3, loadBitmap()).Times(2);
  EXPECT_CALL(pfm4, loadBitmap()).Times(2);
  EXPECT_CALL(pfm5, loadBitmap()).Times(2);
  EXPECT_CALL(pfm6, loadBitmap()).Times(1);

  oap::Images images = {&pfm1, &pfm2, &pfm3, &pfm4, &pfm5, &pfm6};
  oap::DataLoader dataLoader(images);

  EXPECT_THAT(images, HaveSizes(50, 50));
}

TEST_F(OapDataLoaderTests, LoadProcessTest2) {
  NiceMock<PngFileMock> pfm1(10, 10);
  NiceMock<PngFileMock> pfm2(20, 20);
  NiceMock<PngFileMock> pfm3(30, 30);
  NiceMock<PngFileMock> pfm4(50, 50);
  NiceMock<PngFileMock> pfm5(30, 30);
  NiceMock<PngFileMock> pfm6(40, 40);

  EXPECT_CALL(pfm1, loadBitmap()).Times(4);
  EXPECT_CALL(pfm2, loadBitmap()).Times(3);
  EXPECT_CALL(pfm3, loadBitmap()).Times(2);
  EXPECT_CALL(pfm4, loadBitmap()).Times(1);
  EXPECT_CALL(pfm5, loadBitmap()).Times(2);
  EXPECT_CALL(pfm6, loadBitmap()).Times(2);

  oap::Images images = {&pfm1, &pfm2, &pfm3, &pfm4, &pfm5, &pfm6};
  oap::DataLoader dataLoader(images);

  EXPECT_THAT(images, HaveSizes(50, 50));
}

TEST_F(OapDataLoaderTests, LoadProcessTest3) {
  oap::Images images;
  for (size_t fa = 0; fa < 100; ++fa) {
    images.push_back(new NiceMock<PngFileMock>(fa, fa));
  }
  images.push_back(new NiceMock<PngFileMock>(102, 102));
  for (size_t fa = 0; fa < 100; ++fa) {
    images.push_back(new NiceMock<PngFileMock>(fa, fa));
  }

  oap::DataLoader dataLoader(images, true);

  EXPECT_THAT(images, HaveSizes(102, 102));
}

TEST_F(OapDataLoaderTests, LoadProcessTest4) {
  oap::Images images;
  for (size_t fa = 0; fa < 100; ++fa) {
    images.push_back(new NiceMock<PngFileMock>(fa, fa));
  }
  for (size_t fa = 0; fa < 99; ++fa) {
    images.push_back(new NiceMock<PngFileMock>(fa, fa));
  }

  oap::DataLoader dataLoader(images, true);

  EXPECT_THAT(images, HaveSizes(99, 99));
}
