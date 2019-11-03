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
#include <utility>
#include <set>
#include <stdio.h>
#include <vector>
#include <map>

#include "ImageSection.h"
#include "Math.h"
#include "Logger.h"

namespace oap
{
namespace bitmap
{
inline int pixelFloattToInt (floatt pixel)
{
  return pixel < 0.5 ? 0 : 1;
}

using Coord = std::pair<size_t, size_t>;
using Coords = std::set<Coord>;
using CoordsMap = std::map<Coord, Coords>;
using ChildRoot = std::map<Coord, Coord>;

struct CoordsSection
{
  Coords coords;
  oap::ImageRegion section;
};

using CoordsSectionSet = std::map<Coord, CoordsSection>;
using CoordsSectionVec = std::vector<std::pair<Coord, CoordsSection>>;

class ConnectedPixels
{
  public:

    ConnectedPixels (size_t width, size_t height);
    virtual ~ConnectedPixels();

    /**
     * \brief set which contains coords group and their dimension in image
     */
    CoordsSectionSet getCoordsSectionSet () const;

    /**
     * \brief vector which contains coords group and their dimension in image
     */
    CoordsSectionVec getCoordsSectionVec () const;

    oap::RegionSize getOverlapingPaternSize () const
    {
      return m_overlapingSize;
    }

    template <typename Callback>
    static ConnectedPixels processGeneric (Callback&& callback, size_t width, size_t height);

    template <typename TArray, typename T, typename Callback>
    static ConnectedPixels processGenericArray (TArray array, size_t width, size_t height, T bgPixel, Callback&& getPixel);

    /**
     *  \brief returns groups of connected pixels which are separated by backgroud pixels in 1D bitmap
     *  \params bgpixel - pixel which determines background
     */
    template <typename T1DArray, typename T>
    static ConnectedPixels process1DArray (T1DArray bitmap1D, size_t width, size_t height, T bgPixel);

    /**
     *  \brief returns groups of connected pixels which are separated by backgroud pixels in 2D bitmap
     *  \params bgpixel - pixel which determines background
     */
    template <typename T2DArray, typename T>
    static ConnectedPixels process2DArray (T2DArray bitmap2d, size_t width, size_t height, T bgPixel);

  protected:
    CoordsMap m_groups;
    ChildRoot m_cr;
    CoordsSectionSet m_css;

    size_t m_width = 0, m_height = 0;

    /**
     * \brief Size of minimal region which can overlaping any pattern found in image.
     */
    oap::RegionSize m_overlapingSize;

    size_t m_adjustedWidth = 0, m_adjustedHeight = 0;

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

template <typename Callback>
ConnectedPixels ConnectedPixels::processGeneric (Callback&& callback, size_t width, size_t height)
{
  ConnectedPixels bitmap_cp (width, height);

  for (size_t y = 0; y < height; ++y)
  {
    for (size_t x = 0; x < width; ++x)
    {
      callback (bitmap_cp, x, y);
    }
  }

  return bitmap_cp;
}

template <typename TArray, typename T, typename Callback>
ConnectedPixels ConnectedPixels::processGenericArray (TArray array, size_t width, size_t height, T bgPixel, Callback&& getPixel)
{
  ConnectedPixels bitmap_cp (width, height);

  return ConnectedPixels::processGeneric (
    [&array, width, height, &bgPixel, &getPixel] (ConnectedPixels& bitmap_cp, size_t x, size_t y)
    {
      if (getPixel (array, x, y) != bgPixel)
      {
        Coord coord (x, y);
        bitmap_cp.connect (coord, coord);

        bitmap_cp.checkLeft (x, y);
        bitmap_cp.checkTopLeft (x, y);
        bitmap_cp.checkTop (x, y);
        bitmap_cp.checkTopRight (x, y);
#if 0
        bitmap_cp.checkRight (x, y);
        bitmap_cp.checkBottomLeft (x, y);
        bitmap_cp.checkBottom (x, y);
        bitmap_cp.checkBottomRight (x, y);
#endif
      }
    },
  width, height);
}

template <typename T1DArray, typename T>
ConnectedPixels ConnectedPixels::process1DArray (T1DArray bitmap1D, size_t width, size_t height, T bgPixel)
{
  return ConnectedPixels::processGenericArray (bitmap1D, width, height, bgPixel,
         [&width](T1DArray bitmap1D, size_t x, size_t y) { return bitmap1D[x + width * y];});
}

template <typename T2DArray, typename T>
ConnectedPixels ConnectedPixels::process2DArray (T2DArray bitmap2D, size_t width, size_t height, T bgPixel)
{
  return ConnectedPixels::processGenericArray (bitmap2D, width, height, bgPixel,
         [](T2DArray bitmap2D, size_t x, size_t y) { return bitmap2D[y][x];});
}

namespace
{
auto defaultCallbackNL = [](){};
}

template<typename Bitmap, typename Callback, typename CallbackNL = decltype (defaultCallbackNL)>
void iterateBitmap (Bitmap pixels, const oap::ImageSection& width, const oap::ImageSection& height, size_t stride, Callback&& callback, CallbackNL&& cnl = std::forward<CallbackNL>(defaultCallbackNL))
{
  for (size_t y = 0; y < height.getl(); ++y)
  {
    for (size_t x = 0; x < width.getl(); ++x)
    {
      size_t idx = x + width.getp() + stride * (y + height.getp());
      floatt value = pixels[idx];
      callback (value, x, y);
    }
    cnl ();
  }
  cnl ();
}

template<typename Bitmap, typename Callback, typename CallbackNL = decltype (defaultCallbackNL)>
void iterateBitmap (Bitmap pixels, const oap::ImageSection& width, const oap::ImageSection& height, Callback&& callback, CallbackNL&& cnl = std::forward<CallbackNL>(defaultCallbackNL))
{
  iterateBitmap (pixels, width, height, width.getl (), callback, cnl);
}

template<typename Bitmap>
void printBitmapRegion (Bitmap pixels, const oap::ImageSection& width, const oap::ImageSection& height, size_t stride)
{
  iterateBitmap (pixels, width, height, stride, [](floatt pixel, size_t x, size_t y){ printf ("%d", pixelFloattToInt (pixel)); }, [](){ printf("\n"); });
}

template<typename Bitmap>
void printBitmap (Bitmap pixels, size_t width, size_t height)
{
  iterateBitmap (pixels, width, height, width, [](floatt pixel, size_t x, size_t y){ printf ("%d", pixelFloattToInt (pixel)); }, [](){ printf("\n"); });
}

template<typename Bitmap, typename Callback>
void printBitmap (Bitmap pixels, size_t width, size_t height, Callback&& callback)
{
  iterateBitmap (pixels, width, height, width, [&callback](floatt pixel, size_t x, size_t y){ callback (pixel, x, y); printf ("%d", pixelFloattToInt (pixel)); }, [](){ printf("\n"); });
}

template<typename Bitmap1D, typename T>
void getBitmapFromSection (Bitmap1D& output, const RegionSize& outputSize, const Bitmap1D& bitmap1D, size_t width, size_t height, const CoordsSection& coordsSection, T bgPixel)
{
  const oap::ImageRegion& region = coordsSection.section;

  const size_t subwidth = outputSize.width;

  std::fill (std::begin (output), std::end (output), bgPixel);

  for (const auto& coord : coordsSection.coords)
  {
    T pixel = bitmap1D[coord.first + width * coord.second];

    debugAssert (coord.first >= coordsSection.section.x.getp1 ());
    debugAssert (coord.second >= coordsSection.section.y.getp1 ());

    size_t nx = coord.first - coordsSection.section.x.getp1 ();
    size_t ny = coord.second - coordsSection.section.y.getp1 ();

    output[nx + subwidth * ny] = pixel;
  }
}

inline bool mergeIf (CoordsSection& dst, const CoordsSection& src, size_t gap)
{
  if (dst.section.extendIf (src.section, gap))
  {
    std::copy (src.coords.begin(), src.coords.end(), std::inserter (dst.coords, dst.coords.end()));
    return true;
  }
  return false;
}

template<typename Container, typename Getter>
void mergeIf (Container& container, size_t gap, Getter&& getter)
{
  for (auto it = container.begin(); it != container.end(); ++it)
  {
    for (auto it1 = container.begin(); it1 != container.end();)
    {
      if (it1 != it)
      {
        CoordsSection&& cs = getter (it);
        CoordsSection&& cs1 = getter (it1);

        if (mergeIf (cs, cs1, gap))
        {
          it1 = container.erase (it1);
        }
        else
        {
          ++it1;
        }
      }
      else
      {
        ++it1;
      }
    }
  }
}

template<typename Container, typename Getter, typename CondCallback>
void removeIf (Container& container, Getter&& get, CondCallback&& condition)
{
  for (auto it = container.begin(); it != container.end();)
  {
    CoordsSection&& cs = get (it);
    if (condition (cs))
    {
      it = container.erase (it);
    }
    else
    {
      ++it;
    }
  }
}

template<typename Container, typename Bitmap1D, typename Getter, typename CoordsCond, typename PixelCond>
void removeCoordsIf (Container& container, Getter&& get, const Bitmap1D& bitmap1D, size_t width, size_t height, CoordsCond&& ccond, PixelCond&& pcond)
{
  removeIf (container, get, [&bitmap1D, &width, &height, &ccond, &pcond](const CoordsSection& cs)
  {
    if (ccond(cs))
    {
      return true;
    }

    for (auto it = cs.coords.begin (); it != cs.coords.end (); ++it)
    {
      if (!pcond (bitmap1D[it->first + width * it->second]))
      {
        return false;
      }
    }
    return true;
  });
}

template<typename Container, typename Bitmap1D, typename T, typename Getter>
void removeIfPixelsAreLower (Container& container, Getter&& get, const Bitmap1D& bitmap1D, size_t width, size_t height, T limit)
{
  removeCoordsIf (container, get, bitmap1D, width, height, [](const CoordsSection& cs) { return cs.coords.empty (); }, [&limit](T pixel) { return pixel < limit; });
}

template<typename Bitmap1D, typename T>
void removeIfPixelsAreLower (CoordsSectionSet& set, const Bitmap1D& bitmap1D, size_t width, size_t height, T limit)
{
  removeIfPixelsAreLower<CoordsSectionSet, Bitmap1D, T> (set, [](CoordsSectionSet::iterator it) { return it->second; }, bitmap1D, width, height, limit);
}

template<typename Bitmap1D, typename T>
void removeIfPixelsAreLower (CoordsSectionVec& vec, const Bitmap1D& bitmap1D, size_t width, size_t height, T limit)
{
  removeIfPixelsAreLower<CoordsSectionVec, Bitmap1D, T>(vec, [](CoordsSectionVec::iterator it) { return it->second; }, bitmap1D, width, height, limit);
}

template<typename Container, typename Bitmap1D, typename T, typename Getter>
void removeIfPixelsAreHigher (Container& container, Getter&& get, const Bitmap1D& bitmap1D, size_t width, size_t height, T limit)
{
  removeCoordsIf (container, get, bitmap1D, width, height, [](const CoordsSection& cs) { return cs.coords.empty (); }, [&limit](T pixel) { return pixel > limit; });
}

template<typename Bitmap1D, typename T>
void removeIfPixelsAreHigher (CoordsSectionSet& set, const Bitmap1D& bitmap1D, size_t width, size_t height, T limit)
{
  removeIfPixelsAreHigher<CoordsSectionSet, Bitmap1D, T> (set, [](CoordsSectionSet::iterator it) { return it->second; }, bitmap1D, width, height, limit);
}

template<typename Bitmap1D, typename T>
void removeIfPixelsAreHigher (CoordsSectionVec& vec, const Bitmap1D& bitmap1D, size_t width, size_t height, T limit)
{
  removeIfPixelsAreHigher<CoordsSectionVec, Bitmap1D, T> (vec, [](CoordsSectionVec::iterator it) { return it->second; }, bitmap1D, width, height, limit);
}

inline void mergeIf (CoordsSectionSet& set, size_t gap)
{
  mergeIf (set, gap, [](CoordsSectionSet::iterator it) { return it->second; });
}

inline void mergeIf (CoordsSectionVec& vec, size_t gap)
{
  mergeIf (vec, gap, [](CoordsSectionVec::iterator it) { return it->second; });
}

}
}

#endif
