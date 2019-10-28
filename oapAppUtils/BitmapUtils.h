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

#ifndef OAP_BITMAP_UTILS_H
#define OAP_BITMAP_UTILS_H

#include <cstddef>
#include <cmath>
#include <stdio.h>
#include <utility>
#include <set>
#include <vector>
#include <map>

#include "Logger.h"

namespace oap
{

class Bitmap_ConnectedPixels
{
  public:
    using Coord = std::pair<int, int>;
    /*struct Coord
    {
      int x;
      int y;

      Coord (int _x, int _y) : x (_x), y (_y)
      {}

      Coord () : x(0), y(0)
      {}

      Coord (const Coord& coord) : x (coord.x), y (coord.y)
      {}

      Coord (Coord&& coord) : x (std::move (coord.x)), y (std::move (coord.y))
      {}

      Coord& operator=(const Coord& coord)
      {
        if (this != &coord)
        {
          x = coord.x;
          y = coord.y;
        }
        return *this;
      }

      Coord& operator=(Coord&& coord)
      {
        if (this != &coord)
        {
          x = std::move (coord.x);
          y = std::move (coord.y);
        }
        return *this;
      }

      bool operator< (const Coord& coord) const
      {
        if (y < coord.y)
        {
          return true;
        }
        else if (y > coord.y)
        {
          return false;
        }
        else if (x < coord.x)
        {
          return true;
        }
        return false;
      }

      bool operator== (const Coord& coord) const
      {
        return x == coord.x && y == coord.y;
      }

      bool operator!= (const Coord& coord) const
      {
        return !((*this) == coord);
      }
    };*/

    using Coords = std::set<Coord>;
    using CoordsMap = std::map<Coord, Coords>;
    using ChildRoot = std::map<Coord, Coord>;

    struct MinMax
    {
      Coord min;
      Coord max;
    };

    struct CoordsSection
    {
      Coords coords;
      MinMax section;
    };

    using CoordsSectionSet = std::map<Coord, CoordsSection>;
    using CoordsSectionVec = std::vector<std::pair<Coord, CoordsSection>>;

    Bitmap_ConnectedPixels (size_t width, size_t height);
    virtual ~Bitmap_ConnectedPixels();

    /**
     * \brief set which contains coords group and their dimension in image
     */
    CoordsSectionSet getCoordsSectionSet () const;

    /**
     * \brief vector which contains coords group and their dimension in image
     */
    CoordsSectionVec getCoordsSectionVec () const;

    /**
     *  \brief Returns groups of connected pixels which are separated by backgroud pixels
     *  \params bgPixel - pixel which determines background
     */
    template <typename T2DArray, typename T>
    static Bitmap_ConnectedPixels process (T2DArray bitmap2d, size_t width, size_t height, T bgPixel);

  protected:
    CoordsMap m_groups;
    ChildRoot m_cr;
    CoordsSectionSet m_css;

    size_t m_width = 0, m_height = 0;

    Coord getRoot (const Coord& coord) const;

    void connect (const Coord& pcoord, const Coord& ncoord);
    bool connectToPixel (size_t x, size_t y, size_t nx, size_t ny);

    void registerIntoGroup (const Coord& coord, std::initializer_list<Coord> coords);

    void removeWithTransfer (const Coord& dst, const Coord& toRemove);

    bool checkTop (size_t x, size_t y);
    bool checkTopLeft (size_t x, size_t y);
    bool checkTopRight (size_t x, size_t y);

    bool checkLeft (size_t x, size_t y);
    bool checkRight (size_t x, size_t y);

    bool checkBottom (size_t x, size_t y);
    bool checkBottomLeft (size_t x, size_t y);
    bool checkBottomRight (size_t x, size_t y);
};

template <typename T2DArray, typename T>
Bitmap_ConnectedPixels Bitmap_ConnectedPixels::process (T2DArray bitmap2d, size_t width, size_t height, T bgPixel)
{
  Bitmap_ConnectedPixels bitmap_cp (width, height);

  for (size_t y = 0; y < height; ++y)
  {
    for (size_t x = 0; x < width; ++x)
    {
      if (bitmap2d[y][x] != bgPixel)
      {
        Coord coord (x, y);
        bitmap_cp.connect (coord, coord);

        bitmap_cp.checkLeft (x, y);
        bitmap_cp.checkTopLeft (x, y);
        bitmap_cp.checkTop (x, y);
        bitmap_cp.checkTopRight (x, y);
        bitmap_cp.checkRight (x, y);
        bitmap_cp.checkBottomLeft (x, y);
        bitmap_cp.checkBottom (x, y);
        bitmap_cp.checkBottomRight (x, y);
      }
    }
  }

  return bitmap_cp;
}

};

#endif
