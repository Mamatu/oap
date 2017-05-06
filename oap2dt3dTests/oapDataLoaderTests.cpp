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
#include "Matrix.h"
#include "HostMatrixUtils.h"
#include "MatrixAPI.h"

#include "ImageMock.h"
#include "PngFileMock.h"

using namespace ::testing;

class OapDataLoaderTests : public testing::Test {
 public:
  virtual void SetUp() {}

  virtual void TearDown() {}

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
  NiceImageMock imageMock("");

  EXPECT_CALL(imageMock, openProtected(_)).Times(1).WillOnce(Return(false));

  oap::Images images;
  images.push_back(&imageMock);

  EXPECT_THROW(oap::DataLoader pngDataLoader(images),
               oap::exceptions::FileNotExist);
}

TEST_F(OapDataLoaderTests, FailVerification) {
  NiceImageMock imageMock("");

  EXPECT_CALL(imageMock, openProtected(_)).Times(1).WillOnce(Return(true));

  EXPECT_CALL(imageMock, isCorrectFormat()).Times(1).WillOnce(Return(false));

  oap::Images images;
  images.push_back(&imageMock);

  EXPECT_THROW(oap::DataLoader dataLoader(images),
               oap::exceptions::NotCorrectFormat);
}

TEST_F(OapDataLoaderTests, Load) {
  NiceImageMock imageMock("");

  EXPECT_CALL(imageMock, openProtected(_)).Times(1).WillOnce(Return(true));

  EXPECT_CALL(imageMock, isCorrectFormat()).Times(1).WillOnce(Return(true));

  EXPECT_CALL(imageMock, loadBitmapProtected()).Times(1);

  EXPECT_CALL(imageMock, freeBitmapProtected()).Times(1);

  ON_CALL(imageMock, getWidth()).WillByDefault(Return(1024));

  ON_CALL(imageMock, getHeight()).WillByDefault(Return(760));

  ON_CALL(imageMock, getOutputWidth()).WillByDefault(Return(1024));

  ON_CALL(imageMock, getOutputHeight()).WillByDefault(Return(760));

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
    images.push_back(new NicePngFileMock(pixels, 4, 2, 2));
    images.push_back(new NicePngFileMock(pixels, 4, 2, 2));
    images.push_back(new NicePngFileMock(pixels, 4, 2, 2));
    images.push_back(new NicePngFileMock(pixels, 4, 2, 2));

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

TEST_F(OapDataLoaderTests, CreateImagesVectorTest) {
  class ImageProxy : public ImageMock {
   public:
    ImageProxy(const std::string& path) : ImageMock(path), m_path(path) {}

    virtual ~ImageProxy() {}

    std::string getPath() const { return m_path; }

   private:
    std::string m_path;
  };

  class DataLoaderProxy : public oap::DataLoader {
   public:
    static oap::Images createImagesVector(const std::string& imageAbsPath,
                                          const std::string& nameBase,
                                          size_t loadCount, size_t count) {
      return oap::DataLoader::createImagesVector<ImageProxy>(
          imageAbsPath, nameBase, loadCount, count);
    }
  };

  size_t count = 1000;

  oap::Images images =
      DataLoaderProxy::createImagesVector("dir1/dir2/", "image_", count, count);

  for (size_t fa = 0; fa < images.size(); ++fa) {
    oap::Image* image = images[fa];
    ImageProxy* imageProxy = dynamic_cast<ImageProxy*>(image);

    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << fa;

    std::string expectedPath = "dir1/dir2/image_" + ss.str();

    EXPECT_EQ(expectedPath, imageProxy->getPath());

    delete images[fa];
  }
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
      if ((*it)->getOutputWidth().optSize != m_optWidth.optSize) {
        hasWidth = false;
      }

      if ((*it)->getOutputHeight().optSize != m_optWidth.optSize) {
        hasHeight = false;
      }

      *listener << "{" << (*it)->getOutputWidth().optSize << ", "
                << (*it)->getOutputHeight().optSize << "}";
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
  NicePngFileMock pfm1(10, 10);
  NicePngFileMock pfm2(20, 20);
  NicePngFileMock pfm3(30, 30);

  EXPECT_CALL(pfm1, loadBitmapProtected()).Times(3);
  EXPECT_CALL(pfm2, loadBitmapProtected()).Times(2);
  EXPECT_CALL(pfm3, loadBitmapProtected()).Times(1);

  oap::Images images = {&pfm1, &pfm2, &pfm3};
  oap::DataLoader dataLoader(images);

  EXPECT_THAT(images, HaveSizes(30, 30));
}

TEST_F(OapDataLoaderTests, LoadProcessTest1) {
  NicePngFileMock pfm1(10, 10);
  NicePngFileMock pfm2(20, 20);
  NicePngFileMock pfm3(30, 30);
  NicePngFileMock pfm4(30, 30);
  NicePngFileMock pfm5(30, 30);
  NicePngFileMock pfm6(50, 50);

  EXPECT_CALL(pfm1, loadBitmapProtected()).Times(4);
  EXPECT_CALL(pfm2, loadBitmapProtected()).Times(3);
  EXPECT_CALL(pfm3, loadBitmapProtected()).Times(2);
  EXPECT_CALL(pfm4, loadBitmapProtected()).Times(2);
  EXPECT_CALL(pfm5, loadBitmapProtected()).Times(2);
  EXPECT_CALL(pfm6, loadBitmapProtected()).Times(1);

  oap::Images images = {&pfm1, &pfm2, &pfm3, &pfm4, &pfm5, &pfm6};
  oap::DataLoader dataLoader(images);

  EXPECT_THAT(images, HaveSizes(50, 50));
}

TEST_F(OapDataLoaderTests, LoadProcessTest2) {
  NicePngFileMock pfm1(10, 10);
  NicePngFileMock pfm2(20, 20);
  NicePngFileMock pfm3(30, 30);
  NicePngFileMock pfm4(50, 50);
  NicePngFileMock pfm5(30, 30);
  NicePngFileMock pfm6(40, 40);

  EXPECT_CALL(pfm1, loadBitmapProtected()).Times(4);
  EXPECT_CALL(pfm2, loadBitmapProtected()).Times(3);
  EXPECT_CALL(pfm3, loadBitmapProtected()).Times(2);
  EXPECT_CALL(pfm4, loadBitmapProtected()).Times(1);
  EXPECT_CALL(pfm5, loadBitmapProtected()).Times(2);
  EXPECT_CALL(pfm6, loadBitmapProtected()).Times(2);

  oap::Images images = {&pfm1, &pfm2, &pfm3, &pfm4, &pfm5, &pfm6};
  oap::DataLoader dataLoader(images);

  EXPECT_THAT(images, HaveSizes(50, 50));
}

TEST_F(OapDataLoaderTests, LoadProcessTest3) {
  oap::Images images;
  for (size_t fa = 0; fa < 100; ++fa) {
    images.push_back(new NicePngFileMock(fa, fa));
  }
  images.push_back(new NicePngFileMock(102, 102));
  for (size_t fa = 0; fa < 100; ++fa) {
    images.push_back(new NicePngFileMock(fa, fa));
  }

  oap::DataLoader dataLoader(images, true);

  EXPECT_THAT(images, HaveSizes(102, 102));
}

TEST_F(OapDataLoaderTests, LoadProcessTest4) {
  oap::Images images;
  for (size_t fa = 0; fa < 100; ++fa) {
    images.push_back(new NicePngFileMock(fa, fa));
  }
  for (size_t fa = 0; fa < 99; ++fa) {
    images.push_back(new NicePngFileMock(fa, fa));
  }

  oap::DataLoader dataLoader(images, true);

  EXPECT_THAT(images, HaveSizes(99, 99));
}
