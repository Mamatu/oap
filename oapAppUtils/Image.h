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

#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <vector>

#include "Math.h"
#include "GraphicUtils.h"

#include "Logger.h"

namespace oap
{

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
    inline void olc ()
    {
      open ();
      loadBitmap ();
      close ();
    }

    inline std::vector<floatt> getStlFloatVector ()
    {
      oap::OptSize size = getOutputHeight().optSize * getOutputWidth().optSize;
      std::vector<floatt> vec;
      vec.reserve (size.optSize);
      getFloattVector (vec.data());
      return vec;
    }

  private:
    template<typename Callback, typename CallbackNL>
    void iterateBitmap (floatt* pixels, const oap::OptSize& width, const oap::OptSize& height, size_t stride, Callback&& callback, CallbackNL&& cnl)
    {
      for (size_t y = 0; y < height.optSize; ++y)
      {
        for (size_t x = 0; x < width.optSize; ++x)
        {
          size_t idx = x + width.begin + stride * (y + height.begin);
          debugAssert (idx < getOutputWidth().optSize * getOutputHeight().optSize);
          floatt value = pixels[idx];
          int pvalue = value > 0.5 ? 1 : 0;
          callback (pvalue, x, y);
        }
        cnl ();
      }
      cnl ();
    }

    inline void printBitmap (floatt* pixels, const oap::OptSize& width, const oap::OptSize& height, size_t stride)
    {
      iterateBitmap (pixels, width, height, stride, [](int pixel, size_t x, size_t y){ printf ("%d", pixel); }, [](){ printf("\n"); });
    }

  public:
    inline void print (const oap::OptSize& width, const oap::OptSize& height)
    {
      size_t rwidth = getOutputWidth().optSize;
      std::unique_ptr<floatt[]> pixels = std::unique_ptr<floatt[]>(new floatt[rwidth * getOutputHeight().optSize]);
      getFloattVector (pixels.get ());

      printBitmap (pixels.get (), width, height, rwidth);
    }

    inline void print ()
    {
      oap::OptSize&& width = getOutputWidth ();
      oap::OptSize&& height = getOutputHeight ();

      print (std::move (width), std::move (height));
    }

    /**
    * \brief Gets width of load image.
    */
    virtual oap::OptSize getWidth() const = 0;

    /**
    * \brief Gets height of load image.
    */
    virtual oap::OptSize getHeight() const = 0;

    /**
    * \brief Forces width of output.
    *
    *  If it is not set, output width (see getOutputWidth) should be
    *  equal to image width (see getWidth()).
    */
    virtual void forceOutputWidth(const oap::OptSize& optWidth) = 0;

    /**
    * \brief Forces height of output.
    *
    *  If it is not set, output height (see getOutputHeight) should be
    *  equal to image height (see getHeight()).
    */
    virtual void forceOutputHeight(const oap::OptSize& optHeight) = 0;

    /**
    * \brief Get width of output.
    *
    *  It may vary from getWidth due to truncate redundant elements of image.
    */
    virtual oap::OptSize getOutputWidth() const = 0;

    /**
    * \brief Get height of output.
    *
    *  It may vary from getHeight due to truncate redundant elements of image.
    */
    virtual oap::OptSize getOutputHeight() const = 0;

    pixel_t getPixel(unsigned int x, unsigned int y) const;

    size_t getLength() const;

    /**
     *  \brief Returns pixel vector of size equals to size of truncated Image (getOutputWidth() * getOutputHeight())
     */
    bool getPixelsVector(pixel_t* pixels) const;

    /**
     *  \brief Returns floatts vector of size equals to size of truncated Image (getOutputWidth() * getOutputHeight())
     */
    void getFloattVector(floatt* vector) const;

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
}

#endif  // IFILE_H
