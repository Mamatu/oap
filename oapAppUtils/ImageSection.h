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

#ifndef OAP_IMAGE_SECTION_H
#define OAP_IMAGE_SECTION_H

#include <algorithm>
#include <cstddef>
#include <stdio.h>
#include <utility>
#include <tuple>

namespace oap
{

struct ImageSection
{
  ImageSection() : section (std::make_pair (0, 0)) {}

  ImageSection(size_t length) : section (std::make_pair (0, length)) {}

  ImageSection(size_t length, size_t point) : section (std::make_pair(point, length)) {}

  static size_t calcLength (size_t p1, size_t p2)
  {
    return p2 - p1 + 1;
  }

  static ImageSection cpp(size_t point1, size_t point2)
  {
    return ImageSection (calcLength (point1, point2), point1);
  }

  std::pair <size_t, size_t> section;

  void setPoint (size_t point)
  {
    section.first = point;
  }

  size_t getPoint () const
  {
    return section.first;
  }

  void setLength (size_t length)
  {
    section.second = length;
  }

  void setLength (size_t p1, size_t p2)
  {
    section.second = calcLength (p1, p2);
  }

  size_t getLength () const
  {
    return section.second;
  }

  void setp (size_t point)
  {
    section.first = point;
  }

  size_t getp () const
  {
    return section.first;
  }

  void setl (size_t length)
  {
    section.second = length;
  }

  void setpp (size_t p1, size_t p2)
  {
    section.first = p1;
    setLength (p1, p2);
  }

  size_t getl () const
  {
    return section.second;
  }

  size_t getp1 () const
  {
    return section.first;
  }

  size_t getp2 () const
  {
    return section.first + section.second - 1;
  }
};

struct RegionSize
{
  size_t width = 0;
  size_t height = 0;

  size_t getSize () const
  {
    return width * height;
  }

  void replaceIfGreater (size_t _width, size_t _height)
  {
    replaceWidthIfGreater (_width);
    replaceHeightIfGreater (_height);
  }

  void replaceIfGreater (const ImageSection& _width, const ImageSection& _height)
  {
    replaceWidthIfGreater (_width.getl ());
    replaceHeightIfGreater (_height.getl ());
  }

  void replaceWidthIfGreater (size_t _width)
  {
    if (width < _width)
    {
      width = _width;
    }
  }

  void replaceHeightIfGreater (size_t _height)
  {
    if (height < _height)
    {
      height = _height;
    }
  }
};

struct ImageRegion
{
  ImageSection x;
  ImageSection y;

  ImageRegion () : x (), y ()
  {}

  ImageRegion (const ImageSection& _x, const ImageSection& _y) : x (_x), y (_y)
  {}

  static ImageRegion create (size_t x1, size_t x2, size_t y1, size_t y2)
  {
    ImageRegion region (ImageSection::cpp (x1, x2), ImageSection::cpp (y1, y2));
    return region;
  }

  size_t getLength () const
  {
    return x.getl () * y.getl ();
  }

  void replaceIfLonger (const ImageSection& _x, const ImageSection& _y)
  {
    if (x.getl() < _x.getl())
    {
      x = _x;
    }
    if (y.getl() < _y.getl())
    {
      y = _y;
    }
  }

  bool extendIf (const ImageRegion& ir, size_t gap)
  {
    size_t cx = x.getp1 () + x.getl () / 2;
    size_t cy = y.getp1 () + y.getl () / 2;

    size_t ir_cx = ir.x.getp1 () + ir.x.getl () / 2;
    size_t ir_cy = ir.y.getp1 () + ir.y.getl () / 2;

    auto tuple = std::make_tuple (cx, cy, this);
    auto tuple1 = std::make_tuple (ir_cx, ir_cy, &ir);

    if (x.getp1() - gap <= ir.x.getp2 () && ir.x.getp1() - gap <= x.getp2 ())
    {
      if (y.getp1() - gap <= ir.y.getp2 () && ir.y.getp1() - gap <= y.getp2 ())
      {
        extend (ir);
        return true;
      }
    }
    return false;
  }

  void extend (const ImageRegion& ir)
  {
    size_t xp1 = std::min (x.getp1 (), ir.x.getp1 ());
    size_t xp2 = std::max (x.getp2 (), ir.x.getp2 ());

    size_t yp1 = std::min (y.getp1 (), ir.y.getp1 ());
    size_t yp2 = std::max (y.getp2 (), ir.y.getp2 ());

    x.setpp (xp1, xp2);
    y.setpp (yp1, yp2);
  }

  /**
   *  \brief Less method for sort. It bases on position of region in image. If region is closer
   *  to left-top corner (0, 0) is more less (where y-axis has higher priority than y-axis)
   */
  bool lessByPosition (const ImageRegion& ir2) const
  {
    if (y.getp1 () == ir2.y.getp1 ())
    {
      return x.getp1 () < ir2.x.getp1 ();
    }

    return (y.getp1 () < ir2.y.getp1());
  }
};

};

#endif
