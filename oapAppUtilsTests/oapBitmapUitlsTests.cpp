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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "BitmapUtils.h"

using namespace ::testing;

using BCP = oap::Bitmap_ConnectedPixels;
class OapBitmapUtilsTests : public testing::Test {
 public:
  virtual void SetUp() {}

  virtual void TearDown() {}

  bool find (size_t x, size_t y, const BCP::CoordsSection& coordsSec)
  {
    auto it = coordsSec.coords.find(BCP::Coord (x, y));
    return it != coordsSec.coords.end ();
  }
  
  template<size_t N>
  bool find (int array[][N], size_t x, size_t y, const BCP::CoordsSection& coordsSec)
  {
    auto it = coordsSec.coords.find(BCP::Coord (x, y));
    if (array[x][y] == 1)
    {
      return (it != coordsSec.coords.end ());
    }
    return (it == coordsSec.coords.end ());
  }

  template<size_t N, typename Callback>
  void enumerateArray (int array[][N], size_t width, size_t height, Callback&& callback)
  {
    for (size_t y = 0; y < height; ++y)
    {
      for (size_t x = 0; x < width; ++x)
      {
        callback (x, y);
      }
    }
  }

  template<size_t N>
  void checkArray (int array[][N], size_t width, size_t height, const BCP::CoordsSection& coordsSec)
  {
    enumerateArray (array, width, height, [this, &array, &coordsSec](size_t x, size_t y) { EXPECT_TRUE (find (array, x, y, coordsSec));});
  }
};

TEST_F(OapBitmapUtilsTests, Test_1)
{
  const size_t dim = 4;

  int array[dim][dim] =
  {
    {0, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 0, 1},
    {0, 0, 0, 0},
  };


  oap::Bitmap_ConnectedPixels b_cp = oap::Bitmap_ConnectedPixels::process (array, dim, dim, 0);
  std::vector<std::pair<BCP::Coord, BCP::CoordsSection>> vec = b_cp.getCoordsSectionVec ();


  EXPECT_EQ (2, vec.size());
  ASSERT_EQ (1, vec[0].second.coords.size());
  ASSERT_EQ (1, vec[1].second.coords.size());
  checkArray (array, dim, dim, vec[0].second);
  enumerateArray (array, dim, dim, [this, &array, &vec](size_t x, size_t y)
  {
    if (x == 1 && y == 1)
    {
      EXPECT_TRUE (find (array, x, y, vec[0].second));
    }
    else if (x == 3 && y == 2)
    {
      EXPECT_TRUE (find (array, x, y, vec[1].second));
    }
    else
    {
      EXPECT_FALSE (find (x, y, vec[0].second));
      EXPECT_FALSE (find (x, y, vec[1].second));
    }
  });
}

TEST_F(OapBitmapUtilsTests, Test_2)
{
  const size_t dim = 4;

  int array[dim][dim] =
  {
    {0, 0, 0, 0},
    {0, 1, 1, 0},
    {0, 1, 1, 0},
    {0, 0, 0, 0},
  };


  oap::Bitmap_ConnectedPixels b_cp = oap::Bitmap_ConnectedPixels::process (array, dim, dim, 0);
  std::vector<std::pair<BCP::Coord, BCP::CoordsSection>> vec = b_cp.getCoordsSectionVec ();


  EXPECT_EQ (1, vec.size());
  ASSERT_EQ (4, vec[0].second.coords.size());
  checkArray (array, dim, dim, vec[0].second);
}

TEST_F(OapBitmapUtilsTests, Test_3)
{
  const size_t dim = 10;

  int array[dim][dim] =
  {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 1, 0, 0, 0, 0},
    {0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  };

  using BCP = oap::Bitmap_ConnectedPixels;

  oap::Bitmap_ConnectedPixels b_cp = oap::Bitmap_ConnectedPixels::process (array, dim, dim, 0);
  std::vector<std::pair<BCP::Coord, BCP::CoordsSection>> vec = b_cp.getCoordsSectionVec ();


  EXPECT_EQ (1, vec.size());
  ASSERT_EQ (13, vec[0].second.coords.size());
  checkArray (array, dim, dim, vec[0].second);
}

TEST_F(OapBitmapUtilsTests, Test_4)
{
  const size_t dim = 10;

  int array[dim][dim] =
  {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 1, 0, 1, 1, 1, 0, 1, 0},
    {1, 0, 0, 0, 0, 1, 0, 0, 1, 0},
    {1, 0, 0, 0, 0, 1, 0, 0, 1, 0},
    {1, 1, 1, 0, 0, 1, 0, 0, 1, 0},
    {0, 0, 1, 0, 0, 1, 0, 0, 1, 0},
    {0, 0, 1, 0, 0, 1, 0, 0, 1, 0},
    {1, 1, 1, 0, 0, 1, 0, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  };

  using BCP = oap::Bitmap_ConnectedPixels;

  oap::Bitmap_ConnectedPixels b_cp = oap::Bitmap_ConnectedPixels::process (array, dim, dim, 0);
  std::vector<std::pair<BCP::Coord, BCP::CoordsSection>> vec = b_cp.getCoordsSectionVec ();

  ASSERT_EQ (3, vec.size());
  EXPECT_EQ (13, vec[0].second.coords.size());
  EXPECT_EQ (9, vec[1].second.coords.size());
  EXPECT_EQ (9, vec[2].second.coords.size());

  enumerateArray (array, dim, dim, [this, &array, &vec](size_t x, size_t y)
  {
    if (x < 3)
    {
      EXPECT_TRUE (find (array, x, y, vec[0].second));
    }
    if (x <= 6)
    {
      EXPECT_TRUE (find (array, x, y, vec[1].second));
    }
    if (x > 6)
    {
      EXPECT_TRUE (find (array, x, y, vec[2].second));
    }
  });
}
