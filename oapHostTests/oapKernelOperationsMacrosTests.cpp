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
#include "CuProcedures/CuKernelOperationsMacros.h"
#include "MatrixInfo.h"
#include "MatrixAPI.h"
#include "GenericProceduresApi.h"
#include "oapHostMatrixUPtr.h"

class OapKernelOperationsMacrosTests : public testing::Test {
 public:

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }

  uintt gColumns (const math::MatrixInfo& minfo) const
  {
    return minfo.columns ();
  }

  uintt gRows (const math::MatrixInfo& minfo) const
  {
    return minfo.rows ();
  }
};

TEST_F(OapKernelOperationsMacrosTests, CalcSharedMemoryTests)
{
  {
    math::MatrixInfo matrixInfo (true, false, 2, 2);
    math::MatrixInfo kernelInfo (true, false, 2, 2);
    EXPECT_EQ(4, oap::generic::aux::convolve_cache_calculateWidth (matrixInfo, kernelInfo));
    EXPECT_EQ(1, oap::generic::aux::convolve_cache_calculateHeight (matrixInfo, kernelInfo));
  }
  {
    math::MatrixInfo matrixInfo (true, false, 3, 3);
    math::MatrixInfo kernelInfo (true, false, 2, 2);
    EXPECT_EQ(8, oap::generic::aux::convolve_cache_calculateWidth (matrixInfo, kernelInfo));
    EXPECT_EQ(2, oap::generic::aux::convolve_cache_calculateHeight (matrixInfo, kernelInfo));
  }
  {
    math::MatrixInfo matrixInfo (true, false, 4, 4);
    math::MatrixInfo kernelInfo (true, false, 2, 2);
    EXPECT_EQ(12, oap::generic::aux::convolve_cache_calculateWidth (matrixInfo, kernelInfo));
    EXPECT_EQ(3, oap::generic::aux::convolve_cache_calculateHeight (matrixInfo, kernelInfo));
  }
  {
    math::MatrixInfo matrixInfo (true, false, 4, 4);
    math::MatrixInfo kernelInfo (true, false, 3, 3);
    EXPECT_EQ(18, oap::generic::aux::convolve_cache_calculateWidth (matrixInfo, kernelInfo));
    EXPECT_EQ(2, oap::generic::aux::convolve_cache_calculateHeight (matrixInfo, kernelInfo));
  }
  {
    math::MatrixInfo matrixInfo (true, false, 4, 4);
    math::MatrixInfo kernelInfo (true, false, 4, 4);
    EXPECT_EQ(16, oap::generic::aux::convolve_cache_calculateWidth (matrixInfo, kernelInfo));
    EXPECT_EQ(1, oap::generic::aux::convolve_cache_calculateHeight (matrixInfo, kernelInfo));
  }
}

TEST_F(OapKernelOperationsMacrosTests, CalcOutputDimTests)
{
  {
    math::MatrixInfo matrixInfo (true, false, 2, 2);
    math::MatrixInfo kernelInfo (true, false, 2, 2);
    EXPECT_EQ(1, oap::generic::aux::convolve_output_calculateWidth (matrixInfo, kernelInfo));
    EXPECT_EQ(1, oap::generic::aux::convolve_output_calculateHeight (matrixInfo, kernelInfo));
  }
  {
    math::MatrixInfo matrixInfo (true, false, 3, 3);
    math::MatrixInfo kernelInfo (true, false, 2, 2);
    EXPECT_EQ(2, oap::generic::aux::convolve_output_calculateWidth (matrixInfo, kernelInfo));
    EXPECT_EQ(2, oap::generic::aux::convolve_output_calculateHeight (matrixInfo, kernelInfo));
  }
  {
    math::MatrixInfo matrixInfo (true, false, 4, 4);
    math::MatrixInfo kernelInfo (true, false, 2, 2);
    EXPECT_EQ(3, oap::generic::aux::convolve_output_calculateWidth (matrixInfo, kernelInfo));
    EXPECT_EQ(3, oap::generic::aux::convolve_output_calculateHeight (matrixInfo, kernelInfo));
  }
  {
    math::MatrixInfo matrixInfo (true, false, 4, 4);
    math::MatrixInfo kernelInfo (true, false, 3, 3);
    EXPECT_EQ(2, oap::generic::aux::convolve_output_calculateWidth (matrixInfo, kernelInfo));
    EXPECT_EQ(2, oap::generic::aux::convolve_output_calculateHeight (matrixInfo, kernelInfo));
  }
  {
    math::MatrixInfo matrixInfo (true, false, 4, 4);
    math::MatrixInfo kernelInfo (true, false, 4, 4);
    EXPECT_EQ(1, oap::generic::aux::convolve_output_calculateWidth (matrixInfo, kernelInfo));
    EXPECT_EQ(1, oap::generic::aux::convolve_output_calculateHeight (matrixInfo, kernelInfo));
  }
}

TEST_F(OapKernelOperationsMacrosTests, CalcOutputIdxTests)
{
  {
    math::MatrixInfo matrixInfo (true, false, 4, 4);
    math::MatrixInfo kernelInfo (true, false, 2, 2);

    uintt columns = oap::generic::aux::convolve_cache_calculateWidth (matrixInfo, kernelInfo);
    uintt rows = oap::generic::aux::convolve_cache_calculateHeight (matrixInfo, kernelInfo);

    for (uintt threadIndexY = 0; threadIndexY < rows; ++threadIndexY)
    {
      for (uintt threadIndexX = 0; threadIndexX < columns; ++threadIndexX)
      {
      }
    }
  }
}

TEST_F(OapKernelOperationsMacrosTests, ConvolutionCalcParamIdxTest_1)
{
  /*
   *  |M1 M2 M3|
   *  |M4 M5 M6|
   *  |M7 M8 M9|
   *
   *  |K1 K2|
   *  |K3 K4|
   *
   *  Cache:
   *  |K1M1 K2M2 K3M4 K4M5 K1M2 K2M3 K3M5 K4M6|
   *
   */
  math::MatrixInfo matrixInfo (true, false, 3, 3);
  math::MatrixInfo kernelInfo (true, false, 2, 2);

  size_t width = 8;
  size_t height = 2;
  EXPECT_EQ(width, oap::generic::aux::convolve_cache_calculateWidth (matrixInfo, kernelInfo));
  EXPECT_EQ(height, oap::generic::aux::convolve_cache_calculateHeight (matrixInfo, kernelInfo));

  for (size_t idxX = 0; idxX < width; ++idxX)
  {
    for (size_t idxY = 0; idxY < height; ++idxY)
    {
      size_t threadIndexX = idxX;
      size_t threadIndexY = idxY;
      EXPECT_LT(KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_X (kernelInfo, gColumns, gRows), matrixInfo.columns()) << "(" << threadIndexX << ", " << threadIndexY << ")";
      EXPECT_LT(KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_Y (kernelInfo, gColumns, gRows), matrixInfo.rows()) << "(" << threadIndexX << ", " << threadIndexY << ")";
    }
  }
}

TEST_F(OapKernelOperationsMacrosTests, ConvolutionCalcParamIdxTest_2)
{
  math::MatrixInfo matrixInfo (true, false, 5, 5);
  math::MatrixInfo kernelInfo (true, false, 3, 3);

  size_t width = 27;
  size_t height = 3;
  EXPECT_EQ(width, oap::generic::aux::convolve_cache_calculateWidth (matrixInfo, kernelInfo));
  EXPECT_EQ(height, oap::generic::aux::convolve_cache_calculateHeight (matrixInfo, kernelInfo));

  for (size_t idxX = 0; idxX < width; ++idxX)
  {
    for (size_t idxY = 0; idxY < height; ++idxY)
    {
      size_t threadIndexX = idxX;
      size_t threadIndexY = idxY;
      EXPECT_LT(KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_X (kernelInfo, gColumns, gRows), matrixInfo.columns()) << "(" << threadIndexX << ", " << threadIndexY << ")";
      EXPECT_LT(KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_Y (kernelInfo, gColumns, gRows), matrixInfo.rows()) << "(" << threadIndexX << ", " << threadIndexY << ")";
    }
  }
}

TEST_F(OapKernelOperationsMacrosTests, ConvolutionCalcParamIdxTest_3)
{
  math::MatrixInfo matrixInfo (true, false, 4, 4);
  math::MatrixInfo kernelInfo (true, false, 2, 2);

  EXPECT_EQ(12, oap::generic::aux::convolve_cache_calculateWidth (matrixInfo, kernelInfo));
  EXPECT_EQ(3, oap::generic::aux::convolve_cache_calculateHeight (matrixInfo, kernelInfo));

  size_t threadIndexX = 0;
  size_t threadIndexY = 0;
  EXPECT_EQ(0, KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_X (kernelInfo, gColumns, gRows));
  EXPECT_EQ(0, KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_Y (kernelInfo, gColumns, gRows));

  threadIndexX = 1;
  threadIndexY = 0;
  EXPECT_EQ(1, KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_X (kernelInfo, gColumns, gRows));
  EXPECT_EQ(0, KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_Y (kernelInfo, gColumns, gRows));

  threadIndexX = 2;
  threadIndexY = 0;
  EXPECT_EQ(0, KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_X (kernelInfo, gColumns, gRows));
  EXPECT_EQ(1, KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_Y (kernelInfo, gColumns, gRows));

  threadIndexX = 3;
  threadIndexY = 0;
  EXPECT_EQ(1, KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_X (kernelInfo, gColumns, gRows));
  EXPECT_EQ(1, KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_Y (kernelInfo, gColumns, gRows));

  threadIndexX = 4;
  threadIndexY = 0;
  EXPECT_EQ(1, KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_X (kernelInfo, gColumns, gRows));
  EXPECT_EQ(0, KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_Y (kernelInfo, gColumns, gRows));

  threadIndexX = 0;
  threadIndexY = 1;
  EXPECT_EQ(0, KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_X (kernelInfo, gColumns, gRows));
  EXPECT_EQ(1, KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_Y (kernelInfo, gColumns, gRows));
}

TEST_F(OapKernelOperationsMacrosTests, ConvolutionCalcParamIdxTest_4)
{
  math::MatrixInfo matrixInfo (true, false, 5, 5);
  math::MatrixInfo kernelInfo (true, false, 2, 2);

  EXPECT_EQ(16, oap::generic::aux::convolve_cache_calculateWidth (matrixInfo, kernelInfo));
  EXPECT_EQ(4, oap::generic::aux::convolve_cache_calculateHeight (matrixInfo, kernelInfo));

  size_t threadIndexX = 0;
  size_t threadIndexY = 0;
  EXPECT_EQ(0, KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_X (kernelInfo, gColumns, gRows));
  EXPECT_EQ(0, KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_Y (kernelInfo, gColumns, gRows));

  threadIndexX = 1;
  threadIndexY = 1;
  EXPECT_EQ(1, KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_X (kernelInfo, gColumns, gRows));
  EXPECT_EQ(1, KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_Y (kernelInfo, gColumns, gRows));
}

TEST_F(OapKernelOperationsMacrosTests, PoolingCalcParamIdxTest_1)
{
  math::MatrixInfo matrixInfo (true, false, 4, 4);
  math::MatrixInfo kernelInfo (true, false, 2, 2);

  size_t width = 8;
  size_t height = 2;
  EXPECT_EQ(width, oap::generic::aux::pooling_cache_calculateWidth (matrixInfo, kernelInfo));
  EXPECT_EQ(height, oap::generic::aux::pooling_cache_calculateHeight (matrixInfo, kernelInfo));

  for (size_t idxX = 0; idxX < width; ++idxX)
  {
    for (size_t idxY = 0; idxY < height; ++idxY)
    {
      size_t threadIndexX = idxX;
      size_t threadIndexY = idxY;
      EXPECT_LT(KEROPER_POOLING_CALCULATE_PARAM_IDX_X (kernelInfo, gColumns, gRows), matrixInfo.columns()) << "(" << threadIndexX << ", " << threadIndexY << ")";
      EXPECT_LT(KEROPER_POOLING_CALCULATE_PARAM_IDX_Y (kernelInfo, gColumns, gRows), matrixInfo.rows()) << "(" << threadIndexX << ", " << threadIndexY << ")";
    }
  }

  {
    size_t threadIndexX = 7;
    size_t threadIndexY = 1;
    EXPECT_EQ(3, KEROPER_POOLING_CALCULATE_PARAM_IDX_X (kernelInfo, gColumns, gRows));
    EXPECT_EQ(3, KEROPER_POOLING_CALCULATE_PARAM_IDX_Y (kernelInfo, gColumns, gRows));
  }
}

TEST_F(OapKernelOperationsMacrosTests, ConvolutionCalcCacheIdxTests_1)
{
  // M1 M2 M3 M4
  // K1 K2
  //
  //K1M1 K2M2 K1M2 K2M3 K1M3 K2M4
  math::MatrixInfo matrixInfo (true, false, 4, 4);
  math::MatrixInfo kernelInfo (true, false, 2, 2);
 
  uintt threadIndexX = 0;
  uintt threadIndexY = 0;
 
  EXPECT_EQ(0, KEROPER_CONVOLUTION_CALCULATE_CACHE_IDX (matrixInfo, kernelInfo, gColumns, gRows));
}

TEST_F(OapKernelOperationsMacrosTests, ConvolutionCalcCacheIdxTests_2)
{
  math::MatrixInfo matrixInfo (true, false, 4, 4);
  math::MatrixInfo kernelInfo (true, false, 2, 2);
 
  uintt threadIndexX = 1;
  uintt threadIndexY = 0;
 
  EXPECT_EQ(1, KEROPER_CONVOLUTION_CALCULATE_CACHE_IDX (matrixInfo, kernelInfo, gColumns, gRows));
}

TEST_F(OapKernelOperationsMacrosTests, ConvolutionCalcCacheIdxTests_3)
{
  math::MatrixInfo matrixInfo (true, false, 4, 4);
  math::MatrixInfo kernelInfo (true, false, 2, 2);
 
  uintt threadIndexX = 0;
  uintt threadIndexY = 1;
 
  EXPECT_EQ(12, KEROPER_CONVOLUTION_CALCULATE_CACHE_IDX (matrixInfo, kernelInfo, gColumns, gRows));
}

TEST_F(OapKernelOperationsMacrosTests, ConvolutionCalcCacheIdxTests_4)
{
  math::MatrixInfo matrixInfo (true, false, 4, 4);
  math::MatrixInfo kernelInfo (true, false, 2, 2);
 
  uintt threadIndexX = 1;
  uintt threadIndexY = 1;
 
  EXPECT_EQ(13, KEROPER_CONVOLUTION_CALCULATE_CACHE_IDX (matrixInfo, kernelInfo, gColumns, gRows));
}

TEST_F(OapKernelOperationsMacrosTests, ConvolutionCalcCacheIdxTests_5)
{
  math::MatrixInfo matrixInfo (true, false, 4, 4);
  math::MatrixInfo kernelInfo (true, false, 2, 2);
 
  uintt threadIndexX = 2;
  uintt threadIndexY = 0;
 
  EXPECT_EQ(2, KEROPER_CONVOLUTION_CALCULATE_CACHE_IDX (matrixInfo, kernelInfo, gColumns, gRows));
}

TEST_F(OapKernelOperationsMacrosTests, ConvolutionCalcCacheIdxTests_6)
{
  math::MatrixInfo matrixInfo (true, false, 4, 4);
  math::MatrixInfo kernelInfo (true, false, 2, 2);
 
  uintt threadIndexX = 2;
  uintt threadIndexY = 1;
 
  EXPECT_EQ(14, KEROPER_CONVOLUTION_CALCULATE_CACHE_IDX (matrixInfo, kernelInfo, gColumns, gRows));
}

TEST_F(OapKernelOperationsMacrosTests, ConvolutionCalcCacheIdxTests_7)
{
  math::MatrixInfo matrixInfo (true, false, 3, 3);
  math::MatrixInfo kernelInfo (true, false, 2, 2);

  /*
  param =
  {
    1, 1, 1,
    0, 1, 1,
    0, 0, 1,
  };

  kernel =
  {
    1, 0,
    0, 1
  };

  cache = 
  {
    1 0 0 1 1 0 0 1
    0 0 0 0 1 0 0 1
  };
  */

  uintt cacheIdx0 = 0;
  uintt cacheIdx1 = 0;
  uintt cacheIdx2 = 0;

  {
    uintt threadIndexX = 0;
    uintt threadIndexY = 1;
 
    cacheIdx0 = KEROPER_CONVOLUTION_CALCULATE_CACHE_IDX (matrixInfo, kernelInfo, gColumns, gRows);

    EXPECT_EQ (8, cacheIdx0);
  }
  {
    uintt threadIndexX = 2;
    uintt threadIndexY = 0;
 
    cacheIdx1 = KEROPER_CONVOLUTION_CALCULATE_CACHE_IDX (matrixInfo, kernelInfo, gColumns, gRows);

    EXPECT_EQ (2, cacheIdx1);
  }
  {
    uintt threadIndexX = 0;
    uintt threadIndexY = 2;
 
    cacheIdx2 = KEROPER_CONVOLUTION_CALCULATE_CACHE_IDX (matrixInfo, kernelInfo, gColumns, gRows);

    EXPECT_EQ (16, cacheIdx2);
  }

  EXPECT_NE (cacheIdx0, cacheIdx1);
  EXPECT_NE (cacheIdx0, cacheIdx2);
  EXPECT_NE (cacheIdx1, cacheIdx2);
}

TEST_F(OapKernelOperationsMacrosTests, ConvolutionCalcCacheIdxTests_8)
{
  math::MatrixInfo matrixInfo (true, false, 2, 2);
  math::MatrixInfo kernelInfo (true, false, 2, 2);
 
  uintt threadIndexX = 2;
  uintt threadIndexY = 0;
 
  EXPECT_EQ(2, KEROPER_CONVOLUTION_CALCULATE_CACHE_IDX (matrixInfo, kernelInfo, gColumns, gRows));
}

TEST_F(OapKernelOperationsMacrosTests, ConvolutionCreateCacheTest)
{
  floatt paramArray[] =
  {
    1, 1, 1, 0, 0,
    0, 1, 1, 1, 0,
    0, 0, 1, 1, 1,
    0, 0, 1, 1, 0,
    0, 1, 1, 0, 0,
  };

  floatt kernelArray[] =
  {
    1, 0, 1,
    0, 1, 0,
    1, 0, 1
  };

  oap::HostMatrixUPtr outcome = oap::host::NewReMatrix (3, 3);
  oap::HostMatrixUPtr param = oap::host::NewReMatrixCopyOfArray (5, 5, paramArray);
  oap::HostMatrixUPtr kernel = oap::host::NewReMatrixCopyOfArray (3, 3, kernelArray);

  auto pinfo = oap::host::GetMatrixInfo (param);
  auto kinfo = oap::host::GetMatrixInfo (kernel);

  uintt width = oap::generic::aux::convolve_cache_calculateWidth (pinfo, kinfo);
  uintt height = oap::generic::aux::convolve_cache_calculateHeight (pinfo, kinfo);

  EXPECT_EQ(27, width);
  EXPECT_EQ(3, height);

  std::vector<uintt> cache;
  cache.resize (width * height);

  for (uintt threadIndexY = 0; threadIndexY < height; ++threadIndexY)
  {
    for (uintt threadIndexX = 0; threadIndexX < width; ++threadIndexX)
    {
      KEROPER_CACHE_CODE (CONVOLUTION, pinfo, kinfo, cache.data(), gColumns, gRows, GetRe (param, px, py) * GetReIndex (kernel, kidx););
    }
  }

  std::vector<uintt> expected;
  for (size_t y = 0; y < 3; ++y)
  {
    for (size_t x = 0; x < 3; ++x)
    {
      for (size_t ky = 0; ky < 3; ++ky)
      {
        for (size_t kx = 0; kx < 3; ++kx)
        {
          expected.push_back (kernelArray[kx + 3 * ky] * paramArray[(x + kx) + 5 * (y + ky)]);
        }
      }  
    }
  }

  EXPECT_EQ(expected.size (), cache.size ());
  EXPECT_EQ(expected, cache);

  for (uintt idx = 0; idx < cache.size(); ++idx)
  {
    printf ("%d", cache[idx]);
    if (idx < cache.size() - 1)
    {
      printf (", ");
    }
  }
  printf ("\n");
}
