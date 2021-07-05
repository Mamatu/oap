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
#include "gmock/gmock.h"
#include "GraphicUtils.hpp"
#include "Exceptions.hpp"

using namespace ::testing;

class OapGraphicUtilsTests : public testing::Test {
 public:
  virtual void SetUp() {}

  virtual void TearDown() {}

  struct bitmap_t {
    char** array;
    size_t width;
    size_t height;
    size_t colorsCount;
  };

  bitmap_t alloc(size_t height, size_t width, size_t colorsCount,
                 char value) const {
    char** array = new char* [height];
    for (size_t fa = 0; fa < height; ++fa) {
      array[fa] = new char[width * colorsCount];
      memset(array[fa], value, width * colorsCount * sizeof(char));
    }
    bitmap_t bitmap = {array, width, height, colorsCount};
    return bitmap;
  }

  void bitset(bitmap_t bitmap, size_t column, size_t row, char value,
              size_t length) {
    memset(&bitmap.array[row][column], value, length * sizeof(char));
  }

  void dealloc(bitmap_t bitmap) {
    char** array = bitmap.array;
    size_t height = bitmap.height;
    for (size_t fa = 0; fa < height; ++fa) {
      delete[] array[fa];
    }
    delete[] array;
  }

  void print(bitmap_t bitmap) {
    for (size_t fa = 0; fa < bitmap.height; ++fa) {
      for (size_t fb = 0; fb < bitmap.width * bitmap.colorsCount; ++fb) {
        char v = bitmap.array[fa][fb];
        if (fb % bitmap.colorsCount == 0) {
          printf("(");
        }
        printf("%d", v);
        if (fb != bitmap.width * bitmap.colorsCount - 1) {
          if (fb % bitmap.colorsCount == bitmap.colorsCount - 1) {
            printf("),");
          } else {
            printf(",");
          }
        } else {
          printf(")\n");
        }
      }
    }
  }
};

TEST_F(OapGraphicUtilsTests, OptWidthTest) {
  bitmap_t bitmap = alloc(10, 10, 3, 0);

  bitset(bitmap, 8, 3, 1, 9);
  bitset(bitmap, 8, 4, 1, 9);
  bitset(bitmap, 8, 5, 1, 9);
  bitset(bitmap, 8, 6, 1, 9);

  print(bitmap);

  oap::ImageSection optWidth = oap::GetOptWidth<char**, char>(
      bitmap.array, bitmap.width, bitmap.height, bitmap.colorsCount);

  EXPECT_EQ(4, optWidth.getl());
  EXPECT_EQ(2, optWidth.getp());

  dealloc(bitmap);
}


TEST_F(OapGraphicUtilsTests, OptHeightTest) {
  bitmap_t bitmap = alloc(10, 10, 3, 0);

  bitset(bitmap, 8, 3, 1, 9);
  bitset(bitmap, 8, 4, 1, 9);
  bitset(bitmap, 8, 5, 1, 9);
  bitset(bitmap, 8, 6, 1, 9);

  print(bitmap);

  oap::ImageSection optHeight = oap::GetOptHeight<char**, char>(
      bitmap.array, bitmap.width, bitmap.height, bitmap.colorsCount);

  EXPECT_EQ(4, optHeight.getl());
  EXPECT_EQ(3, optHeight.getp());

  dealloc(bitmap);
}
