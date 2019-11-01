/*
 * Copyright 2016 - 2019 Marcin Matula
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

#include "Config.h"
#include "Logger.h"
#include "Exceptions.h"
#include "PngFile.h"
#include "BitmapUtils.h"

using namespace ::testing;

class OapPngFileTests : public testing::Test
{
  public:
    OapPngFileTests()
    {
      m_data_path = utils::Config::getPathInOap("oap2dt3d/data/");
      m_images_path = m_data_path + "images_monkey/";
    }

    virtual void SetUp() {}

    virtual void TearDown() {}

    static std::string m_data_path;
    static std::string m_images_path;

    static std::string getImagePath(const std::string& filename)
    {
      return m_images_path + filename;
    }
};

std::string OapPngFileTests::m_data_path;
std::string OapPngFileTests::m_images_path;

TEST_F(OapPngFileTests, SaveImageToFileTest)
{
  EXPECT_NO_THROW(try
  {
    oap::PngFile pngFile(getImagePath("image000.png"));
    oap::PngFile pngFile900(getImagePath("image900.png"));
    oap::PngFile pngFile910(getImagePath("image910.png"));

    EXPECT_TRUE(pngFile.save("/tmp/Oap/truncated_image000.png"));
    EXPECT_TRUE(pngFile900.save("/tmp/Oap/truncated_image900.png"));
    EXPECT_TRUE(pngFile910.save("/tmp/Oap/truncated_image910.png"));
  }
  catch (const std::exception& ex)
  {
    debugException(ex);
    throw;
  });
}

class BitmapsConversionTest : public oap::PngFile
{
  public:
    BitmapsConversionTest(size_t width, size_t height, size_t colorsCount)
      : oap::PngFile("/tmp/Oap/test_data/test_iamge")
    {
      m_bitmap2dTest = createBitmap2D(width, height, colorsCount);
      m_widthTest = width;
      m_heightTest = height;
      m_colorsCountTest = colorsCount;
      for (size_t fa = 0; fa < m_heightTest; ++fa)
      {
        for (size_t fb = 0; fb < m_widthTest * m_colorsCountTest; ++fb)
        {
          m_bitmap2dTest[fa][fb] = fa;
        }
      }
    }

    virtual ~BitmapsConversionTest()
    {
      oap::PngFile::destroyBitmap2d(m_bitmap2dTest, m_heightTest);
    }

    png_byte* createBitmap1dFrom2d(png_bytep* bitmap2d,
                                   const oap::ImageSection& optWidth,
                                   oap::ImageSection& optHeight)
    {
      return oap::PngFile::createBitmap1dFrom2d(bitmap2d, optWidth, optHeight,
             m_colorsCountTest);
    }

    oap::pixel_t* createPixelsVectorFrom1d(png_byte* bitmap1d, size_t width,
                                           size_t height)
    {
      return oap::PngFile::createPixelsVectorFrom1d(bitmap1d, width, height,
             m_colorsCountTest);
    }

    png_bytep* m_bitmap2dTest;
    size_t m_widthTest;
    size_t m_heightTest;
    size_t m_colorsCountTest;

    static void run(size_t width, size_t height, size_t coloursCount)
    {
      BitmapsConversionTest pngFile(width, height, coloursCount);

      oap::ImageSection optWidth(width);
      oap::ImageSection optHeight(height);

      png_byte* buffer1 = pngFile.createBitmap1dFrom2d(pngFile.m_bitmap2dTest,
                          optWidth, optHeight);
      oap::pixel_t* buffer2 =
        pngFile.createPixelsVectorFrom1d(buffer1, width, height);

      std::vector<int> expectedVec;
      std::vector<int> testVec1;
      std::vector<int> testVec2;

      for (size_t fa = 0; fa < pngFile.m_heightTest; ++fa)
      {
        for (size_t fb = 0; fb < pngFile.m_widthTest * pngFile.m_colorsCountTest;
             ++fb)
        {
          testVec1.push_back(pngFile.m_bitmap2dTest[fa][fb]);
        }
      }

      const size_t length = pngFile.m_widthTest * pngFile.m_colorsCountTest;

      for (size_t fa = 0; fa < pngFile.m_heightTest; ++fa)
      {
        for (size_t fb = 0; fb < length; ++fb)
        {
          expectedVec.push_back(fa);
          testVec2.push_back(buffer1[fa * length + fb]);
        }
      }

      EXPECT_EQ(expectedVec.size(), testVec1.size());
      EXPECT_EQ(expectedVec.size(), testVec2.size());

      for (size_t fa = 0; fa < pngFile.m_heightTest; ++fa)
      {
        decltype(expectedVec)::iterator it = expectedVec.begin();
        decltype(expectedVec)::iterator it1 = expectedVec.begin();
        std::advance(it, fa * pngFile.m_widthTest * pngFile.m_colorsCountTest);
        std::advance(it1,
                     (fa + 1) * pngFile.m_widthTest * pngFile.m_colorsCountTest);
        std::vector<int> vec = std::vector<int>(it, it1);
        std::vector<int> vec1 =
          std::vector<int>(pngFile.m_bitmap2dTest[fa],
                           pngFile.m_bitmap2dTest[fa] +
                           pngFile.m_widthTest * pngFile.m_colorsCountTest);
        EXPECT_THAT(vec1, ElementsAreArray(vec));
      }

      EXPECT_THAT(expectedVec, ElementsAreArray(testVec1));
      EXPECT_THAT(expectedVec, ElementsAreArray(testVec2));

      delete[] buffer2;
      delete[] buffer1;
    }
};

TEST_F(OapPngFileTests, Bitmap2DToBitmap1DConversionTest1)
{
  BitmapsConversionTest::run(1, 1, 4);
}
TEST_F(OapPngFileTests, Bitmap2DToBitmap1DConversionTest2)
{
  BitmapsConversionTest::run(2, 2, 4);
}
TEST_F(OapPngFileTests, Bitmap2DToBitmap1DConversionTest3)
{
  BitmapsConversionTest::run(4, 4, 4);
}
TEST_F(OapPngFileTests, Bitmap2DToBitmap1DConversionTest4)
{
  BitmapsConversionTest::run(8, 8, 4);
}
TEST_F(OapPngFileTests, Bitmap2DToBitmap1DConversionTest5)
{
  BitmapsConversionTest::run(12, 12, 4);
}

TEST_F(OapPngFileTests, TwoImagesPixelVectorsTests)
{
  class Test
  {
    public:
      static void run(const std::string& image1, const std::string& image2)
      {
        try
        {
          oap::PngFile pngFile(OapPngFileTests::getImagePath(image1), false);
          oap::PngFile pngFile1(OapPngFileTests::getImagePath(image2), false);

          pngFile.olc();
          pngFile1.olc();

          const std::vector<floatt>& vec = pngFile.getStlFloatVector ();
          const std::vector<floatt>& vec1 = pngFile1.getStlFloatVector ();
          
          EXPECT_EQ(vec.size(), vec1.size());
        }
        catch (const std::exception& ex)
        {
          debugException(ex);
          throw;
        }
      }
  };

  EXPECT_NO_THROW(Test::run("image000.png", "image010.png"));
  EXPECT_NO_THROW(Test::run("image020.png", "image030.png"));
  EXPECT_NO_THROW(Test::run("image010.png", "image030.png"));
  EXPECT_NO_THROW(Test::run("image040.png", "image050.png"));
  EXPECT_NO_THROW(Test::run("image100.png", "image110.png"));
  EXPECT_NO_THROW(Test::run("image670.png", "image680.png"));
}

TEST_F(OapPngFileTests, LoadMnistExamples)
{
  std::string path = utils::Config::getPathInOap("oapNeural/data/text/");
  path = path + "MnistExamples.png";
  oap::PngFile pngFile (path, false);

  pngFile.olc ();

  std::vector<floatt> vec = pngFile.getStlFloatVector ();

  size_t expectedWidth = 557;
  size_t expectedHeight = 326;

  EXPECT_EQ (expectedWidth, pngFile.getOutputWidth().getl());
  EXPECT_EQ (expectedHeight, pngFile.getOutputHeight().getl());

  oap::bitmap::ConnectedPixels cp = oap::bitmap::ConnectedPixels::process1DArray (vec, expectedWidth, expectedHeight, 1);
  oap::bitmap::CoordsSectionVec csVec = cp.getCoordsSectionVec ();

  //pngFile.print (oap::ImageSection(55 - 25, 25), oap::ImageSection(40 - 10, 10));
}
