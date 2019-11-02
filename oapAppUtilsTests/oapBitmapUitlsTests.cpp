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
using namespace oap::bitmap;

class OapBitmapUtilsTests : public testing::Test {
 public:
  virtual void SetUp() {}

  virtual void TearDown() {}

  bool find (size_t x, size_t y, const CoordsSection& coordsSec)
  {
    auto it = coordsSec.coords.find(Coord (x, y));
    return it != coordsSec.coords.end ();
  }
  
  template<size_t N>
  bool find (int array[][N], size_t x, size_t y, const CoordsSection& coordsSec)
  {
    auto it = coordsSec.coords.find(Coord (x, y));
    if (array[y][x] == 1)
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

  template<size_t N, typename Callback>
  void checkArray (int array[][N], size_t width, size_t height, const CoordsSection& coordsSec, Callback&& callback)
  {
    enumerateArray (array, width, height, [this, &array, &coordsSec, &callback](size_t x, size_t y) { bool o = find (array, x, y, coordsSec); callback (o, x, y); });
  }
};

TEST_F(OapBitmapUtilsTests, TestConnection_1)
{
  class Test_Bitmap_ConnectedPixels : public oap::bitmap::ConnectedPixels
  {
    public:
      Test_Bitmap_ConnectedPixels (size_t width, size_t height) : oap::bitmap::ConnectedPixels (width, height)
      {}

      void connect (const Coord& pcoord, const Coord& ncoord)
      {
        oap::bitmap::ConnectedPixels::connect (pcoord, ncoord);
      }
  };

  Test_Bitmap_ConnectedPixels b_cp (10, 10);

  Coord coord (2, 2);
  b_cp.connect (coord, coord);

  const auto& vec = b_cp.getCoordsSectionVec ();

  ASSERT_EQ (1, vec.size());
  ASSERT_EQ (1, vec[0].second.coords.size());
}

TEST_F(OapBitmapUtilsTests, TestConnection_2)
{
  class Test_Bitmap_ConnectedPixels : public oap::bitmap::ConnectedPixels
  {
    public:
      Test_Bitmap_ConnectedPixels (size_t width, size_t height) : oap::bitmap::ConnectedPixels (width, height)
      {}

      void connect (const Coord& pcoord, const Coord& ncoord)
      {
        oap::bitmap::ConnectedPixels::connect (pcoord, ncoord);
      }
  };

  Test_Bitmap_ConnectedPixels b_cp (10, 10);

  Coord coord1 (2, 2);
  Coord coord2 (2, 3);
  b_cp.connect (coord1, coord2);
  Coord coord3 (3, 2);
  b_cp.connect (coord1, coord3);

  Coord coord4 (3, 3);
  b_cp.connect (coord4, coord4);

  const auto& vec = b_cp.getCoordsSectionVec ();

  ASSERT_EQ (2, vec.size());
  ASSERT_EQ (3, vec[0].second.coords.size());
  ASSERT_EQ (1, vec[1].second.coords.size());
}

TEST_F(OapBitmapUtilsTests, TestConnection_3)
{
  class Test_Bitmap_ConnectedPixels : public oap::bitmap::ConnectedPixels
  {
    public:
      Test_Bitmap_ConnectedPixels (size_t width, size_t height) : oap::bitmap::ConnectedPixels (width, height)
      {}

      void connect (const Coord& pcoord, const Coord& ncoord)
      {
        oap::bitmap::ConnectedPixels::connect (pcoord, ncoord);
      }
  };

  Test_Bitmap_ConnectedPixels b_cp (10, 10);

  Coord coord1 (2, 2);
  Coord coord2 (2, 3);
  b_cp.connect (coord1, coord2);
  Coord coord3 (3, 2);
  b_cp.connect (coord1, coord3);

  Coord coord4 (3, 3);
  Coord coord5 (3, 5);
  b_cp.connect (coord4, coord4);
  b_cp.connect (coord4, coord5);

  Coord coord6 (6, 6);
  b_cp.connect (coord6, coord6);

  const auto& vec = b_cp.getCoordsSectionVec ();

  ASSERT_EQ (3, vec.size());
  ASSERT_EQ (3, vec[0].second.coords.size());
  ASSERT_EQ (2, vec[1].second.coords.size());
  ASSERT_EQ (1, vec[2].second.coords.size());
}

TEST_F(OapBitmapUtilsTests, Test_2DArray_1)
{
  const size_t dim = 4;

  int array[dim][dim] =
  {
    {0, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 0, 1},
    {0, 0, 0, 0},
  };

  oap::bitmap::ConnectedPixels b_cp = oap::bitmap::ConnectedPixels::process2DArray (array, dim, dim, 0);
  std::vector<std::pair<Coord, CoordsSection>> vec = b_cp.getCoordsSectionVec ();

  ASSERT_EQ (2, vec.size());

  ASSERT_EQ (1, vec[0].second.coords.size());
  EXPECT_EQ (1, vec[0].second.section.x.getp1());
  EXPECT_EQ (1, vec[0].second.section.y.getp1());
  EXPECT_EQ (1, vec[0].second.section.x.getp2());
  EXPECT_EQ (1, vec[0].second.section.y.getp2());

  ASSERT_EQ (1, vec[1].second.coords.size());
  EXPECT_EQ (3, vec[1].second.section.x.getp1());
  EXPECT_EQ (2, vec[1].second.section.y.getp1());
  EXPECT_EQ (3, vec[1].second.section.x.getp2());
  EXPECT_EQ (2, vec[1].second.section.y.getp2());

  enumerateArray (array, dim, dim, [this, &array, &vec](size_t x, size_t y)
  {
    if (x == 1 && y == 1)
    {
      EXPECT_TRUE (find (array, x, y, vec[0].second)) << "x: " << x << " y: " << y;
    }
    else if (x == 3 && y == 2)
    {
      EXPECT_TRUE (find (array, x, y, vec[1].second)) << "x: " << x << " y: " << y;
    }
    else
    {
      EXPECT_FALSE (find (x, y, vec[0].second)) << "x: " << x << " y: " << y;
      EXPECT_FALSE (find (x, y, vec[1].second)) << "x: " << x << " y: " << y;
    }
  });
}

TEST_F(OapBitmapUtilsTests, Test_2DArray_2)
{
  const size_t dim = 4;

  int array[dim][dim] =
  {
    {0, 0, 0, 0},
    {0, 1, 1, 0},
    {0, 1, 1, 0},
    {0, 0, 0, 0},
  };


  oap::bitmap::ConnectedPixels b_cp = oap::bitmap::ConnectedPixels::process2DArray (array, dim, dim, 0);
  std::vector<std::pair<Coord, CoordsSection>> vec = b_cp.getCoordsSectionVec ();


  EXPECT_EQ (1, vec.size());
  ASSERT_EQ (4, vec[0].second.coords.size());
  EXPECT_EQ (1, vec[0].second.section.x.getp1());
  EXPECT_EQ (1, vec[0].second.section.y.getp1());
  EXPECT_EQ (2, vec[0].second.section.x.getp2());
  EXPECT_EQ (2, vec[0].second.section.y.getp2());

  checkArray (array, dim, dim, vec[0].second, [](bool b, size_t x, size_t y) { EXPECT_TRUE (b) << "x: " << x << " y: " << y; });
}

TEST_F(OapBitmapUtilsTests, Test_2DArray_3)
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

  oap::bitmap::ConnectedPixels b_cp = oap::bitmap::ConnectedPixels::process2DArray (array, dim, dim, 0);
  std::vector<std::pair<Coord, CoordsSection>> vec = b_cp.getCoordsSectionVec ();


  EXPECT_EQ (1, vec.size());
  ASSERT_EQ (13, vec[0].second.coords.size());
  EXPECT_EQ (3, vec[0].second.section.x.getp1());
  EXPECT_EQ (2, vec[0].second.section.y.getp1());
  EXPECT_EQ (5, vec[0].second.section.x.getp2());
  EXPECT_EQ (7, vec[0].second.section.y.getp2());

  checkArray (array, dim, dim, vec[0].second, [](bool b, size_t x, size_t y) { EXPECT_TRUE (b) << "x: " << x << " y: " << y; });
}

TEST_F(OapBitmapUtilsTests, Test_2DArray_4)
{
  const size_t dim = 10;

  int array[dim][dim] =
  {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 1, 0, 1, 1, 1, 0, 0, 0},
    {0, 1, 1, 0, 1, 0, 1, 0, 0, 0},
    {0, 1, 0, 0, 1, 0, 1, 0, 0, 0},
    {0, 1, 0, 0, 1, 1, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  };

  oap::bitmap::ConnectedPixels b_cp = oap::bitmap::ConnectedPixels::process2DArray (array, dim, dim, 0);
  std::vector<std::pair<Coord, CoordsSection>> vec = b_cp.getCoordsSectionVec ();

  ASSERT_EQ (2, vec.size());
  EXPECT_EQ (6, vec[0].second.coords.size());
  EXPECT_EQ (1, vec[0].second.section.x.getp1());
  EXPECT_EQ (2, vec[0].second.section.y.getp1());
  EXPECT_EQ (2, vec[0].second.section.x.getp2());
  EXPECT_EQ (5, vec[0].second.section.y.getp2());

  EXPECT_EQ (10, vec[1].second.coords.size());
  EXPECT_EQ (4, vec[1].second.section.x.getp1());
  EXPECT_EQ (2, vec[1].second.section.y.getp1());
  EXPECT_EQ (6, vec[1].second.section.x.getp2());
  EXPECT_EQ (5, vec[1].second.section.y.getp2());

  enumerateArray (array, dim, dim, [this, &array, &vec](size_t x, size_t y)
  {
    if (x < 3)
    {
      EXPECT_TRUE (find (array, x, y, vec[0].second)) << "x: " << x << " y: " << y;
    }
    else
    {
      EXPECT_TRUE (find (array, x, y, vec[1].second)) << "x: " << x << " y: " << y;
    }
  });
}

TEST_F(OapBitmapUtilsTests, Test_2DArray_5)
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

  oap::bitmap::ConnectedPixels b_cp = oap::bitmap::ConnectedPixels::process2DArray (array, dim, dim, 0);
  std::vector<std::pair<Coord, CoordsSection>> vec = b_cp.getCoordsSectionVec ();

  ASSERT_EQ (3, vec.size());

  EXPECT_EQ (13, vec[0].second.coords.size());
  EXPECT_EQ (0, vec[0].second.section.x.getp1());
  EXPECT_EQ (1, vec[0].second.section.y.getp1());
  EXPECT_EQ (2, vec[0].second.section.x.getp2());
  EXPECT_EQ (7, vec[0].second.section.y.getp2());

  EXPECT_EQ (9, vec[1].second.coords.size());
  EXPECT_EQ (4, vec[1].second.section.x.getp1());
  EXPECT_EQ (1, vec[1].second.section.y.getp1());
  EXPECT_EQ (6, vec[1].second.section.x.getp2());
  EXPECT_EQ (7, vec[1].second.section.y.getp2());

  EXPECT_EQ (9, vec[2].second.coords.size());
  EXPECT_EQ (7, vec[2].second.section.x.getp1());
  EXPECT_EQ (1, vec[2].second.section.y.getp1());
  EXPECT_EQ (9, vec[2].second.section.x.getp2());
  EXPECT_EQ (7, vec[2].second.section.y.getp2());

  enumerateArray (array, dim, dim, [this, &array, &vec](size_t x, size_t y)
  {
    if (x < 3)
    {
      EXPECT_TRUE (find (array, x, y, vec[0].second)) << "x: " << x << " y: " << y;
    }
    else if (x <= 6)
    {
      EXPECT_TRUE (find (array, x, y, vec[1].second)) << "x: " << x << " y: " << y;
    }
    else if (x > 6)
    {
      EXPECT_TRUE (find (array, x, y, vec[2].second)) << "x: " << x << " y: " << y;
    }
  });
}

TEST_F(OapBitmapUtilsTests, Test_2DArray_6)
{
  const size_t dim = 10;

  int array[dim][dim] =
  {
    {1, 1, 1, 1, 1, 1, 0, 1, 1, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 0, 1, 1, 1, 1, 1, 1, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {1, 1, 1, 1, 1, 0, 0, 0, 0, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {1, 1, 1, 1, 1, 1, 0, 1, 1, 1},
  };

  oap::bitmap::ConnectedPixels b_cp = oap::bitmap::ConnectedPixels::process2DArray (array, dim, dim, 0);
  std::vector<std::pair<Coord, CoordsSection>> vec = b_cp.getCoordsSectionVec ();

  ASSERT_EQ (2, vec.size());

  EXPECT_EQ (24, vec[0].second.coords.size());
  EXPECT_EQ (0, vec[0].second.section.x.getp1());
  EXPECT_EQ (0, vec[0].second.section.y.getp1());
  EXPECT_EQ (5, vec[0].second.section.x.getp2());
  EXPECT_EQ (9, vec[0].second.section.y.getp2());

  EXPECT_EQ (27, vec[1].second.coords.size());
  EXPECT_EQ (2, vec[1].second.section.x.getp1());
  EXPECT_EQ (0, vec[1].second.section.y.getp1());
  EXPECT_EQ (9, vec[1].second.section.x.getp2());
  EXPECT_EQ (9, vec[1].second.section.y.getp2());
}

TEST_F(OapBitmapUtilsTests, Test_1DArray_6)
{
  const size_t dim = 10;

  int array[dim * dim] =
  {
    1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
  };

  oap::bitmap::ConnectedPixels b_cp = oap::bitmap::ConnectedPixels::process1DArray (array, dim, dim, 0);
  std::vector<std::pair<Coord, CoordsSection>> vec = b_cp.getCoordsSectionVec ();

  ASSERT_EQ (2, vec.size());

  EXPECT_EQ (24, vec[0].second.coords.size());
  EXPECT_EQ (0, vec[0].second.section.x.getp1());
  EXPECT_EQ (0, vec[0].second.section.y.getp1());
  EXPECT_EQ (5, vec[0].second.section.x.getp2());
  EXPECT_EQ (9, vec[0].second.section.y.getp2());

  EXPECT_EQ (27, vec[1].second.coords.size());
  EXPECT_EQ (2, vec[1].second.section.x.getp1());
  EXPECT_EQ (0, vec[1].second.section.y.getp1());
  EXPECT_EQ (9, vec[1].second.section.x.getp2());
  EXPECT_EQ (9, vec[1].second.section.y.getp2());
}
