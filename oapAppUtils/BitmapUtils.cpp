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

#include "BitmapUtils.h"

#include <algorithm>

namespace oap
{
namespace bitmap
{
namespace
{
  inline int x (const Coord& coord)
  {
    return coord.first;
  }

  inline int y (const Coord& coord)
  {
    return coord.second;
  }
}

PatternsSeeker::PatternsSeeker (size_t width, size_t height) :
  m_width (width), m_height (height)
{}

PatternsSeeker::~PatternsSeeker()
{}

CoordsSectionSet PatternsSeeker::getCoordsSectionSet () const
{
  return m_css;
}

CoordsSectionVec PatternsSeeker::getCoordsSectionVec () const
{
  std::vector<std::pair<Coord, CoordsSection>> vec (m_css.begin(), m_css.end());
  return vec;
}

Coord PatternsSeeker::getRoot (const Coord& coord) const
{
  auto it = m_cr.find (coord);
  if (it != m_cr.end ())
  {
    if (it->first != it->second)
    {
      return getRoot (it->second);
    }
    else
    {
      return it->second;
    }
  }
  return coord;
}

void PatternsSeeker::connect (const Coord& coord1, const Coord& coord2)
{
  Coord root1 = getRoot (coord1);
  Coord root2 = getRoot (coord2);
  m_cr [root1] = root2;

  registerIntoGroup (root1, {root1, coord1});
  registerIntoGroup (root2, {root2, coord2});

  removeWithTransfer (root2, root1);
}

void PatternsSeeker::registerIntoGroup (const Coord& root, std::initializer_list<Coord> coords)
{
  auto it = m_css.find (root);
  if (it == m_css.end ())
  {
    int minx = std::min (coords, [](const Coord& c1, const Coord& c2) { return x (c1) < x (c2); }).first;
    int miny = std::min (coords, [](const Coord& c1, const Coord& c2) { return y (c1) < y (c2); }).second;
    int maxx = std::max (coords, [](const Coord& c1, const Coord& c2) { return x (c1) > x (c2); }).first;
    int maxy = std::max (coords, [](const Coord& c1, const Coord& c2) { return y (c1) > y (c2); }).second;

    ImageRegion ir = ImageRegion::create (minx, maxx, miny, maxy);
    m_css[root] = { Coords (coords), ir};
    m_overlapingSize.replaceIfGreater (ir.x, ir.y);
  }
  else
  {
    auto& pair = *it;

    auto minx = pair.second.section.x.getp1();
    auto miny = pair.second.section.y.getp1();

    auto maxx = pair.second.section.x.getp2();
    auto maxy = pair.second.section.y.getp2();

    for (auto it = coords.begin(); it != coords.end(); ++it)
    {
      pair.second.coords.insert (*it);

      if (minx > it->first) { minx = it->first; }
      if (miny > it->second) { miny = it->second; }
      if (maxx < it->first) { maxx = it->first; }
      if (maxy < it->second) { maxy = it->second; }
    }

    pair.second.section.x.cpp (minx, maxx);
    pair.second.section.y.cpp (miny, maxy);
    m_overlapingSize.replaceIfGreater (pair.second.section.x.getl (), pair.second.section.y.getl ());
  }
}

bool PatternsSeeker::connectToPixel (size_t x, size_t y, size_t nx, size_t ny)
{
  Coord pcoord (nx, ny);
  Coord ncoord (x, y);

  auto it = m_cr.find (pcoord);
  if (it != m_cr.end())
  {
    connect (pcoord, ncoord);
    return true;
  }
  return false;
}

void PatternsSeeker::removeWithTransfer (const Coord& dst, const Coord& toRemove)
{
  if (dst == toRemove)
  {
    return;
  }

  CoordsSectionSet::iterator ito = m_css.find (dst);
  logAssert (ito != m_css.end());

  CoordsSectionSet::iterator ifrom = m_css.find (toRemove);
  if (ifrom != m_css.end())
  {
    Coords::iterator biter = ifrom->second.coords.begin();
    Coords::iterator eiter = ifrom->second.coords.end();
  
    size_t minx = std::min (ito->second.section.x.getp1(), ifrom->second.section.x.getp1());
    size_t miny = std::min (ito->second.section.y.getp1(), ifrom->second.section.y.getp1());

    size_t maxx = std::max (ito->second.section.x.getp2(), ifrom->second.section.x.getp2());
    size_t maxy = std::max (ito->second.section.y.getp2(), ifrom->second.section.y.getp2());

    ito->second.section.x.setpp (minx, maxx);
    ito->second.section.y.setpp (miny, maxy);

    m_overlapingSize.replaceIfGreater (ito->second.section.x, ito->second.section.y);

    std::copy (biter, eiter, std::inserter (ito->second.coords, ito->second.coords.end ()));
    m_css.erase (ifrom);
  }
}

bool PatternsSeeker::checkTop (size_t x, size_t y)
{
  if (y == 0)
  {
    return false;
  }

  return connectToPixel (x, y, x, y - 1);
}

bool PatternsSeeker::checkTopLeft (size_t x, size_t y)
{
  if (x == 0 || y == 0)
  {
    return false;
  }

  return connectToPixel (x, y, x - 1, y - 1);
}


bool PatternsSeeker::checkTopRight (size_t x, size_t y)
{
  if (y == 0 || x >= m_width)
  {
    return false;
  }

  return connectToPixel (x, y, x + 1, y - 1);
}

bool PatternsSeeker::checkLeft (size_t x, size_t y)
{
  if (x == 0)
  {
    return false;
  }

  return connectToPixel (x, y, x - 1, y);
}

bool PatternsSeeker::checkRight (size_t x, size_t y)
{
  if (x >= m_width)
  {
    return false;
  }

  return connectToPixel (x, y, x + 1, y);
}

bool PatternsSeeker::checkBottom (size_t x, size_t y)
{
  if (y >= m_height)
  {
    return false;
  }

  return connectToPixel (x, y, x, y + 1);
}

bool PatternsSeeker::checkBottomLeft (size_t x, size_t y)
{
  if (x == 0 || y >= m_height)
  {
    return false;
  }

  return connectToPixel (x, y, x - 1, y + 1);
}


bool PatternsSeeker::checkBottomRight (size_t x, size_t y)
{
  if (y >= m_height || x >= m_width)
  {
    return false;
  }

  return connectToPixel (x, y, x + 1, y + 1);
}

}
}
