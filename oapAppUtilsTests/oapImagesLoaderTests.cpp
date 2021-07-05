/*
 * Copyright 2016 - 2021 Marcin Matula
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
#include "ImagesLoader.hpp"
#include "Matrix.hpp"
#include "oapHostComplexMatrixApi.hpp"
#include "MatrixAPI.hpp"

#include "ImageMock.hpp"
#include "PngFileMock.hpp"

using namespace ::testing;

class OapImagesLoaderTests : public testing::Test {
 public:
  virtual void SetUp() {}

  virtual void TearDown() {}

 public:  // ImagesLoaderProxy type
  class ImagesLoaderProxy : public oap::ImagesLoader {
   public:
    static oap::Images m_emptyImages;
    ImagesLoaderProxy() : ImagesLoader(m_emptyImages) {}

    static std::string constructAbsPath(const std::string& basePath) {
      return oap::ImagesLoader::constructAbsPath(basePath);
    }

    static std::string constructImagePath(const std::string& absPath,
                                          const std::string& nameBase,
                                          size_t index) {
      return oap::ImagesLoader::constructImagePath(absPath, nameBase, index);
    }
  };
};

oap::Images OapImagesLoaderTests::ImagesLoaderProxy::m_emptyImages;

TEST_F(OapImagesLoaderTests, LoadFail) {
  NiceImageMock imageMock;

  EXPECT_CALL(imageMock, openProtected(_)).Times(1).WillOnce(Return(false));

  oap::Images images;
  images.push_back(&imageMock);

  EXPECT_THROW(oap::ImagesLoader pngImagesLoader(images),
               oap::exceptions::FileNotExist);
}

TEST_F(OapImagesLoaderTests, FailVerification) {
  NiceImageMock imageMock;

  EXPECT_CALL(imageMock, openProtected(_)).Times(1).WillOnce(Return(true));

  EXPECT_CALL(imageMock, isCorrectFormat()).Times(1).WillOnce(Return(false));

  oap::Images images;
  images.push_back(&imageMock);

  EXPECT_THROW(oap::ImagesLoader dataLoader(images),
               oap::exceptions::NotCorrectFormat);
}

TEST_F(OapImagesLoaderTests, Load) {
  NiceImageMock imageMock;

  EXPECT_CALL(imageMock, openProtected(_)).Times(1).WillOnce(Return(true));

  EXPECT_CALL(imageMock, isCorrectFormat()).Times(1).WillOnce(Return(true));

  EXPECT_CALL(imageMock, loadBitmapProtected()).Times(1);

  EXPECT_CALL(imageMock, freeBitmapProtected()).Times(1);

  ON_CALL(imageMock, getWidth()).WillByDefault(Return(1024));

  ON_CALL(imageMock, getHeight()).WillByDefault(Return(760));

  ON_CALL(imageMock, getOutputWidth()).WillByDefault(Return(1024));

  ON_CALL(imageMock, getOutputHeight()).WillByDefault(Return(760));

  EXPECT_CALL(imageMock, closeProtected()).Times(1);

  oap::Images images;
  images.push_back(&imageMock);

  EXPECT_NO_THROW(oap::ImagesLoader dataLoader(images));
}

TEST_F(OapImagesLoaderTests, Matrix4x4FromImage) {
  const oap::pixel_t pv = oap::Image::getPixelMax();
  oap::pixel_t pixels[16] = {pv, pv, pv, pv, pv, pv, pv, pv,
                             pv, pv, pv, pv, pv, pv, pv, pv};
  try {
    oap::Images images;
    images.push_back(new NicePngFileMock(pixels, 4, 2, 2));
    images.push_back(new NicePngFileMock(pixels, 4, 2, 2));
    images.push_back(new NicePngFileMock(pixels, 4, 2, 2));
    images.push_back(new NicePngFileMock(pixels, 4, 2, 2));

    oap::ImagesLoader dataLoader(images, true);

    math::ComplexMatrix* matrix = dataLoader.createMatrix();

    EXPECT_EQ(4, gColumns (matrix));
    EXPECT_EQ(4, gRows (matrix));

    for (size_t fa = 0; fa < gColumns (matrix); ++fa) {
      for (size_t fb = 0; fb < gRows (matrix); ++fb) {
        const floatt value = GetRe(matrix, fa, fb);
        EXPECT_EQ(1, value);
      }
    }
    oap::chost::DeleteMatrix(matrix);
  } catch (const std::exception& ex) {
    debugException(ex);
  }
}

TEST_F(OapImagesLoaderTests, ContructAbsPathTest) {
  ImagesLoaderProxy dataLoaderProxy;
}

TEST_F(OapImagesLoaderTests, ContructImagePathTest) {
  EXPECT_EQ("abs/image0",
            ImagesLoaderProxy::constructImagePath("abs/", "image", 0));
  EXPECT_EQ("abs/image1",
            ImagesLoaderProxy::constructImagePath("abs/", "image", 1));
  EXPECT_EQ("abs/image2",
            ImagesLoaderProxy::constructImagePath("abs/", "image", 2));
  EXPECT_EQ("abs/image3",
            ImagesLoaderProxy::constructImagePath("abs/", "image", 3));
  EXPECT_EQ("abs/image4",
            ImagesLoaderProxy::constructImagePath("abs/", "image", 4));
  EXPECT_EQ("abs/image5",
            ImagesLoaderProxy::constructImagePath("abs/", "image", 5));
}

TEST_F(OapImagesLoaderTests, CreateImagesVectorTest) {
  class ImageProxy : public ImageMock {
   public:
    ImageProxy(const std::string& path) : ImageMock(path), m_path(path) {}

    virtual ~ImageProxy() {}

    std::string getPath() const { return m_path; }

   private:
    std::string m_path;
  };

  class ImagesLoaderProxy : public oap::ImagesLoader {
   public:
    static oap::Images createImagesVector(const std::string& imageAbsPath,
                                          const std::string& nameBase,
                                          size_t loadCount) {
      return oap::ImagesLoader::createImagesVector<ImageProxy>(
          imageAbsPath, nameBase, loadCount);
    }
  };

  size_t count = 1000;

  oap::Images images =
      ImagesLoaderProxy::createImagesVector("dir1/dir2/", "image_", count);

  for (size_t fa = 0; fa < images.size(); ++fa) {
    oap::Image* image = images[fa];
    ImageProxy* imageProxy = dynamic_cast<ImageProxy*>(image);

    std::string expectedPath = "dir1/dir2/image_" + std::to_string(fa);

    EXPECT_EQ(expectedPath, imageProxy->getPath());

    delete images[fa];
  }
}

class HaveSizesMatcher : public MatcherInterface<const oap::Images&> {
 public:
  HaveSizesMatcher(const oap::ImageSection& optWidth, const oap::ImageSection& optHeight)
      : m_optWidth(optWidth), m_optHeight(optHeight) {}

  virtual bool MatchAndExplain(const oap::Images& images,
                               MatchResultListener* listener) const {
    bool hasWidth = true;
    bool hasHeight = true;

    *listener << "{";

    for (oap::Images::const_iterator it = images.begin(); it != images.end(); ++it) {
      if ((*it)->getOutputWidth().getl() != m_optWidth.getl()) {
        hasWidth = false;
      }

      if ((*it)->getOutputHeight().getl() != m_optWidth.getl()) {
        hasHeight = false;
      }

      *listener << "{" << (*it)->getOutputWidth().getl() << ", "
                << (*it)->getOutputHeight().getl() << "}";
    }

    *listener << "}";

    return hasWidth && hasHeight;
  }

  virtual void DescribeTo(::std::ostream* os) const {
    *os << "have sizes: " << m_optWidth.getl() << " " << m_optHeight.getl();
  }

  virtual void DescribeNegationTo(::std::ostream* os) const {
    *os << "have not sizes: " << m_optWidth.getl() << " "
        << m_optHeight.getl();
  }

 private:
  oap::ImageSection m_optWidth;
  oap::ImageSection m_optHeight;
};

inline Matcher<const oap::Images&> HaveSizes(const oap::ImageSection& optWidth,
                                             const oap::ImageSection& optHeight) {
  return MakeMatcher(new HaveSizesMatcher(optWidth, optHeight));
}

TEST_F(OapImagesLoaderTests, LoadProcessTest) {
  NicePngFileMock pfm1(10, 10);
  NicePngFileMock pfm2(20, 20);
  NicePngFileMock pfm3(30, 30);

  EXPECT_CALL(pfm1, loadBitmapProtected()).Times(3);
  EXPECT_CALL(pfm2, loadBitmapProtected()).Times(2);
  EXPECT_CALL(pfm3, loadBitmapProtected()).Times(1);

  oap::Images images = {&pfm1, &pfm2, &pfm3};
  oap::ImagesLoader dataLoader(images);

  EXPECT_THAT(images, HaveSizes(30, 30));
}

TEST_F(OapImagesLoaderTests, LoadProcessTest1) {
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
  oap::ImagesLoader dataLoader(images);

  EXPECT_THAT(images, HaveSizes(50, 50));
}

TEST_F(OapImagesLoaderTests, LoadProcessTest2) {
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
  oap::ImagesLoader dataLoader(images);

  EXPECT_THAT(images, HaveSizes(50, 50));
}

TEST_F(OapImagesLoaderTests, LoadProcessTest3) {
  oap::Images images;
  for (size_t fa = 0; fa < 100; ++fa) {
    images.push_back(new NicePngFileMock(fa, fa));
  }
  images.push_back(new NicePngFileMock(102, 102));
  for (size_t fa = 0; fa < 100; ++fa) {
    images.push_back(new NicePngFileMock(fa, fa));
  }

  oap::ImagesLoader dataLoader(images, true);

  EXPECT_THAT(images, HaveSizes(102, 102));
}

TEST_F(OapImagesLoaderTests, LoadProcessTest4) {
  oap::Images images;
  for (size_t fa = 0; fa < 100; ++fa) {
    images.push_back(new NicePngFileMock(fa, fa));
  }
  for (size_t fa = 0; fa < 99; ++fa) {
    images.push_back(new NicePngFileMock(fa, fa));
  }

  oap::ImagesLoader dataLoader(images, true);

  EXPECT_THAT(images, HaveSizes(99, 99));
}
