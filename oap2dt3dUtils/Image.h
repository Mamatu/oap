/*
 * Copyright 2016, 2017 Marcin Matula
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

#include "Math.h"
#include "GraphicUtils.h"

namespace oap {

typedef unsigned int pixel_t;

class Image {
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

  bool getPixelsVector(pixel_t* pixels) const;

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

 protected:
  virtual void closeProtected() = 0;

  virtual void loadBitmapProtected() = 0;

  virtual void freeBitmapProtected() = 0;

  virtual void getPixelsVectorProtected(pixel_t* pixels) const = 0;

  virtual bool openProtected(const std::string& path) = 0;

  virtual bool isCorrectFormat() const = 0;

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
