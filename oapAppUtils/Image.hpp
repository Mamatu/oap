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

#ifndef IMAGE_H
#define IMAGE_H

#include <functional>
#include <string>
#include <vector>

#include "Math.hpp"

#include "BitmapUtils.hpp"
#include "GraphicUtils.hpp"

#include "Logger.hpp"

namespace oap
{

class Image;

namespace
{
  auto defaultFilter = [](oap::bitmap::CoordsSectionVec&, const std::vector<floatt>&, const oap::Image*){};
  using DefaultFilter = decltype (defaultFilter);
}

typedef unsigned int pixel_t;

class Image
{
  public:
    Image(const std::string& path);

    virtual ~Image();

    void open();

    bool isOpened() const;
    bool isLoaded() const;

    bool read(void* buffer, size_t size);

    virtual bool read(void* buffer, size_t repeat, size_t size) = 0;

    void loadBitmap();

    void freeBitmap();

    /**
     * Open, load bitmap and close
     */
    void olc ();

    std::vector<floatt> getStlFloatVector () const;

    /**
     * \brief Prints subimage in boundaries determined by width and height as array of 0 and 1 digit (gray scale)
     */
    void print (const oap::ImageSection& width, const oap::ImageSection& height);

    /**
     * \brief Prints output image as array of 0 and 1 digit (gray scale)
     */
    void print ();

    /**
    * \brief Gets width of load image.
    */
    virtual oap::ImageSection getWidth() const = 0;

    /**
    * \brief Gets height of load image.
    */
    virtual oap::ImageSection getHeight() const = 0;

    /**
    * \brief Forces width of output.
    *
    *  If it is not set, output width (see getOutputWidth) should be
    *  equal to image width (see getWidth()).
    */
    virtual void forceOutputWidth(const oap::ImageSection& optWidth) = 0;

    /**
    * \brief Forces height of output.
    *
    *  If it is not set, output height (see getOutputHeight) should be
    *  equal to image height (see getHeight()).
    */
    virtual void forceOutputHeight(const oap::ImageSection& optHeight) = 0;

    /**
    * \brief Get width of output.
    *
    *  It may vary from getWidth due to truncate redundant elements of image.
    */
    virtual oap::ImageSection getOutputWidth() const = 0;

    /**
    * \brief Get height of output.
    *
    *  It may vary from getHeight due to truncate redundant elements of image.
    */
    virtual oap::ImageSection getOutputHeight() const = 0;

    pixel_t getPixel(unsigned int x, unsigned int y) const;

    size_t getLength() const;

    /**
     *  \brief Returns pixel vector of size equals to size of truncated Image (getOutputWidth() * getOutputHeight())
     */
    bool getPixelsVector(pixel_t* pixels) const;

    /**
     *  \brief Returns floatts vector of size equals to size of truncated Image (getOutputWidth() * getOutputHeight())
     */
    template<typename Vec>
    void getFloattVector(Vec&& vector) const
    {
      const size_t length = getLength();

      std::unique_ptr<pixel_t[]> pixelsUPtr(new pixel_t[length]);

      pixel_t* pixels = pixelsUPtr.get();
      pixel_t max = Image::getPixelMax();

      this->getPixelsVector(pixels);

      for (size_t fa = 0; fa < length; ++fa)
      {
        vector[fa] = oap::Image::convertPixelToFloatt(pixels[fa]);
      }
    }

    using PatternBitmap = std::vector<floatt>;

    /**
     *  \brief This struct is passed to callback from iteratePatterns method
     */
    struct Pattern
    {
      PatternBitmap patternBitmap;

      /**
       *  \brief Size of region which can overlaping any patterns found in image (the best fit)
       */
      oap::RegionSize overlapingRegion;

      /**
       *  \brief Region which descibes pattern (its size and location)
       */
      oap::ImageRegion imageRegion;
    };

    using Patterns = std::vector<Pattern>;

    template<typename T, typename Callback, typename Filter = DefaultFilter>
    void iteratePatterns (T bgPixel, Callback&& callback, Filter&& filter = std::forward<DefaultFilter>(defaultFilter)) const;

    template<typename T, typename Filter = DefaultFilter>
    void getPatterns (Patterns& patterns, T bgPixel, Filter&& filter = std::forward<DefaultFilter>(defaultFilter)) const;

    template<typename T, typename Filter = DefaultFilter>
    Patterns getPatterns (T bgPixel, Filter&& filter = std::forward<DefaultFilter>(defaultFilter)) const;

    void close();

    static pixel_t convertRgbToPixel(unsigned char r, unsigned char g,
                                     unsigned char b);

    static floatt convertPixelToFloatt(pixel_t pixel);

    static floatt convertRgbToFloatt(unsigned char r, unsigned char g,
                                     unsigned char b);

    static pixel_t getPixelMax();

    virtual std::string getSufix() const = 0;

    std::string getFileName() const;
    std::string getFilePath() const;

    bool save(const std::string& prefix, const std::string& path);
    bool save(const std::string& path);

  protected:
    virtual void closeProtected() = 0;

    virtual void loadBitmapProtected() = 0;

    virtual void freeBitmapProtected() = 0;

    virtual void getPixelsVectorProtected(pixel_t* pixels) const = 0;

    virtual bool openProtected(const std::string& path) = 0;

    virtual bool isCorrectFormat() const = 0;

    virtual void onSave(const std::string& path) = 0;
    virtual bool saveProtected(const std::string& path) = 0;

    virtual pixel_t getPixelProtected(unsigned int x, unsigned int y) const = 0;

  private:
    bool m_isOpen;

    std::string m_path;
    std::string m_fileName;
    std::string m_filePath;

    static pixel_t m_MaxPixel;

    bool m_loadedBitmap;
};

template<typename T, typename Callback, typename Filter>
void Image::iteratePatterns (T bgPixel, Callback&& callback, Filter&& filter) const
{
  size_t width = getOutputWidth ().getl ();
  size_t height = getOutputHeight ().getl ();

  std::vector<floatt> vec = getStlFloatVector ();

  oap::bitmap::PatternsSeeker ps = oap::bitmap::PatternsSeeker::process1DArray (vec, width, height, 1);
  oap::bitmap::CoordsSectionVec csVec = ps.getCoordsSectionVec ();

  using Coord = oap::bitmap::Coord;
  using CoordsSection = oap::bitmap::CoordsSection;

  oap::RegionSize rs = ps.getOverlapingPaternSize ();

  const oap::Image* image = this;
  filter (csVec, vec, image);

  for (const auto& pair : csVec)
  {
    PatternBitmap patternBitmap;
    patternBitmap.resize (rs.getSize ());
    oap::bitmap::getBitmapFromSection (patternBitmap, rs, vec, width, height, pair.second, 1.f);

    Pattern pattern = {std::move (patternBitmap), rs, pair.second.section};
    callback (std::move (pattern));
  }
}

template<typename T, typename Filter>
void Image::getPatterns (Patterns& patterns, T bgPixel, Filter&& filter) const
{
  iteratePatterns (bgPixel, [&patterns](Pattern&& pattern) { patterns.push_back (pattern); }, filter);
}

template<typename T, typename Filter>
Image::Patterns Image::getPatterns (T bgPixel, Filter&& filter) const
{
  Patterns patterns;
  getPatterns (patterns, filter);
  return patterns;
}
}

#endif  // IFILE_H
