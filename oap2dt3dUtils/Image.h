/*
 * Copyright 2016 Marcin Matula
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

#include <png.h>
#include "Math.h"

namespace oap {

typedef unsigned int pixel_t;

class Image {
 public:
  Image();

  virtual ~Image();

  void open(const char* path);

  bool read(void* buffer, size_t size);

  virtual bool read(void* buffer, size_t repeat, size_t size) = 0;

  virtual void loadBitmap() = 0;

  virtual void freeBitmap() = 0;

  virtual size_t getWidth() const = 0;

  virtual size_t getHeight() const = 0;

  pixel_t getPixel(unsigned int x, unsigned int y) const;

  size_t getLength() const;

  virtual void getPixelsVector(pixel_t* pixels) const = 0;

  virtual void close() = 0;

  static pixel_t convertRgbToPixel(unsigned char r, unsigned char g,
                                   unsigned char b);

  static floatt convertPixelToFloatt(pixel_t pixel);

  static floatt convertRgbToFloatt(unsigned char r, unsigned char g,
                                   unsigned char b);

  static pixel_t getPixelMax();

 protected:
  virtual bool openInternal(const char* path) = 0;

  virtual bool isPngInternal() const = 0;

  virtual pixel_t getPixelInternal(unsigned int x, unsigned int y) const = 0;

 private:
  static pixel_t m_MaxPixel;
};
}

#endif  // IFILE_H
