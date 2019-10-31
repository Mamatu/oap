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

namespace oap
{
namespace
{
  using BCP = Bitmap_ConnectedPixels;
}

BCP::CoordsSectionSet Bitmap_ConnectedPixels::getCoordsSectionSet () const
{
  return m_css;
}

BCP::CoordsSectionVec Bitmap_ConnectedPixels::getCoordsSectionVec () const
{
  std::vector<std::pair<BCP::Coord, BCP::CoordsSection>> vec (m_css.begin(), m_css.end());
  return vec;
}

BCP::Coord Bitmap_ConnectedPixels::getRoot (const Coord& coord) const
{
  auto it = m_cr.find (coord);
  if (it != m_cr.end () && it->first != it->second)
  {
    return getRoot (it->second);
  }
  return it->second;
}

void Bitmap_ConnectedPixels::connect (const Coord& pcoord, const BCP::Coord& ncoord)
{
  Coord root = getRoot (pcoord);
  Coord nroot = getRoot (ncoord);
  m_cr [nroot] = root;

  auto it = m_css.find (root);
  if (it == m_css.end ())
  {
    int minx = std::min (pcoord.x, ncoord.x);
    int miny = std::min (pcoord.y, ncoord.y);
    int maxx = std::max (pcoord.x, ncoord.x);
    int maxy = std::max (pcoord.y, ncoord.y);
    m_css[root] = { Coords (), {Coord (minx, miny), Coord (maxx, maxy)}};

    m_css[root].coords.insert (pcoord);
    m_css[root].coords.insert (ncoord);
    m_css[root].coords.insert (root);
    m_css[root].coords.insert (nroot);
  }
  else
  {
    auto& pair = m_css[root];
    pair.coords.insert (pcoord);
    pair.coords.insert (ncoord);
    pair.coords.insert (root);
    pair.coords.insert (nroot);
  
    auto& minx = pair.section.min.x;
    auto& miny = pair.section.min.y;

    auto& maxx = pair.section.max.x;
    auto& maxy = pair.section.max.y;

    if (minx > ncoord.x) { minx = ncoord.x; }
    if (miny > ncoord.y) { miny = ncoord.y; }
    if (maxx < ncoord.x) { minx = ncoord.x; }
    if (maxy < ncoord.y) { miny = ncoord.y; }
  }
}

bool Bitmap_ConnectedPixels::connectToPixel (size_t x, size_t y, size_t nx, size_t ny)
{
  Coord pcoord (nx, ny);

  auto it = m_cr.find (pcoord);
  if (it != m_cr.end())
  {
    connect (pcoord, Coord (x, y));
    return true;
  }
  return false;
}

bool Bitmap_ConnectedPixels::checkTop (size_t x, size_t y)
{
  if (y == 0)
  {
    return false;
  }

  return connectToPixel (x, y, x, y - 1);
}

bool Bitmap_ConnectedPixels::checkTopLeft (size_t x, size_t y)
{
  if (x == 0 || y == 0)
  {
    return false;
  }

  return connectToPixel (x, y, x - 1, y - 1);
}


bool Bitmap_ConnectedPixels::checkTopRight (size_t x, size_t y)
{
  if (y == 0 || x >= m_width)
  {
    return false;
  }

  return connectToPixel (x, y, x + 1, y - 1);
}

bool Bitmap_ConnectedPixels::checkLeft (size_t x, size_t y)
{
  if (x == 0)
  {
    return false;
  }

  return connectToPixel (x, y, x - 1, y);
}

bool Bitmap_ConnectedPixels::checkRight (size_t x, size_t y)
{
  if (x >= m_width)
  {
    return false;
  }

  return connectToPixel (x, y, x + 1, y);
}

bool Bitmap_ConnectedPixels::checkBottom (size_t x, size_t y)
{
  if (y >= m_height)
  {
    return false;
  }

  return connectToPixel (x, y, x, y + 1);
}

bool Bitmap_ConnectedPixels::checkBottomLeft (size_t x, size_t y)
{
  if (x == 0 || y >= m_height)
  {
    return false;
  }

  return connectToPixel (x, y, x - 1, y + 1);
}


bool Bitmap_ConnectedPixels::checkBottomRight (size_t x, size_t y)
{
  if (y >= m_height || x >= m_width)
  {
    return false;
  }

  return connectToPixel (x, y, x + 1, y + 1);
}
}
