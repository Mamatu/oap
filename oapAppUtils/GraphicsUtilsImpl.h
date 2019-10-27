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

#ifndef GRAPHICUTILSIMPL_H
#define GRAPHICUTILSIMPL_H

#include <utility>

#include <cstddef>
#include <stdio.h>

#include <memory>
#include "GraphicUtils.h"

namespace oap
{

template <typename T2DArray, typename T>
size_t getBeginColumn(T2DArray bitmap2d, size_t width, size_t height,
                      size_t colorsCount);

template <typename T2DArray, typename T>
size_t getEndColumn(T2DArray bitmap2d, size_t width, size_t height,
                    size_t colorsCount);

template <typename T2DArray, typename T>
size_t getBeginRow(T2DArray bitmap2d, size_t width, size_t height,
                   size_t colorsCount);

template <typename T2DArray, typename T>
size_t getEndRow(T2DArray bitmap2d, size_t width, size_t height,
                 size_t colorsCount);

template <typename T2DArray, typename T>
OptSize GetOptWidth(T2DArray bitmap2d, size_t width, size_t height, size_t colorsCount)
{
  size_t beginC = getBeginColumn<T2DArray, T>(bitmap2d, width, height, colorsCount);
  size_t endC = width;

  size_t optWidth = width;

  if (beginC != width)
  {
    endC = getEndColumn<T2DArray, T>(bitmap2d, width, height, colorsCount);
    optWidth = endC - beginC;
  }
  else
  {
    optWidth = width;
    beginC = 0;
  }

  OptSize optSize(optWidth, beginC);

  return optSize;
}

template <typename T2DArray, typename T>
OptSize GetOptHeight(T2DArray bitmap2d, size_t width, size_t height,
                     size_t colorsCount)
{
  size_t beginR = getBeginRow<T2DArray, T>(bitmap2d, width, height, colorsCount);

  size_t endR = height;

  size_t optHeight = height;

  if (beginR != height)
  {
    endR = getEndRow<T2DArray, T>(bitmap2d, width, height, colorsCount);
    optHeight = endR - beginR;
  }
  else
  {
    optHeight = height;
    beginR = 0;
  }

  OptSize optSize(optHeight, beginR);

  return optSize;
}

size_t getColumn(size_t index, size_t colorsCount);

template <typename T2DArray, typename T>
bool verifyColumn(T2DArray bitmap2d, size_t fa, size_t height,
                  size_t colorsCount, size_t* outColumn)
{
  size_t fb = 0;
  T gbyte = bitmap2d[fb][fa];
  for (; fb < height; fb++)
  {
    T byte = bitmap2d[fb][fa];
    if (gbyte != byte)
    {
      (*outColumn) = getColumn(fa, colorsCount);
      return false;
    }
  }
  return true;
}

template <typename T>
bool isEqual(T* color1, T* color2, size_t colorsCount)
{
  for (size_t fa = 0; fa < colorsCount; ++fa)
  {
    if (color1[fa] != color2[fa])
    {
      return false;
    }
  }
  return true;
}

template <typename T2DArray, typename T>
bool verifyRow(T2DArray bitmap2d, size_t fa, size_t width, size_t colorsCount,
               size_t* outRow)
{
  std::unique_ptr<T[]> gcolorUPtr(new T[colorsCount]);
  std::unique_ptr<T[]> colorUPtr(new T[colorsCount]);
  T* gcolor = gcolorUPtr.get();
  T* color = colorUPtr.get();
  size_t fb = 0;

  for (size_t x = 0; x < colorsCount; ++x)
  {
    gcolor[x] = bitmap2d[fa][fb * colorsCount + x];
  }
  for (; fb < width; fb++)
  {
    for (size_t x = 0; x < colorsCount; ++x)
    {
      color[x] = bitmap2d[fa][fb * colorsCount + x];
    }

    if (!isEqual<T>(gcolor, color, colorsCount))
    {
      (*outRow) = fa;
      return false;
    }
  }
  return true;
}

template <typename T2DArray, typename T>
size_t getBeginColumn(T2DArray bitmap2d, size_t width, size_t height,
                      size_t colorsCount)
{
  size_t column = 0;
  for (size_t fa = 0; fa < width * colorsCount; ++fa)
  {
    if (!verifyColumn<T2DArray, T>(bitmap2d, fa, height, colorsCount,
                                   &column))
    {
      return column;
    }
  }
  return getColumn(width * colorsCount, colorsCount);
}

template <typename T2DArray, typename T>
size_t getEndColumn(T2DArray bitmap2d, size_t width, size_t height,
                    size_t colorsCount)
{
  size_t column = 0;
  for (size_t fa = width * colorsCount - 1; fa >= 1; --fa)
  {
    if (!verifyColumn<T2DArray, T>(bitmap2d, fa, height, colorsCount,
                                   &column))
    {
      return column + 1;
    }
  }
  return 0;
}

template <typename T2DArray, typename T>
size_t getBeginRow(T2DArray bitmap2d, size_t width, size_t height,
                   size_t colorsCount)
{
  size_t row = 0;
  for (size_t fa = 0; fa < height; ++fa)
  {
    if (!verifyRow<T2DArray, T>(bitmap2d, fa, width, colorsCount, &row))
    {
      return row;
    }
  }
  return height;
}

template <typename T2DArray, typename T>
size_t getEndRow(T2DArray bitmap2d, size_t width, size_t height,
                 size_t colorsCount)
{
  size_t row = 0;
  for (size_t fa = height - 1; fa >= 1; --fa)
  {
    if (!verifyRow<T2DArray, T>(bitmap2d, fa, width, colorsCount, &row))
    {
      return row + 1;
    }
  }
  return 0;
}
};

#endif
