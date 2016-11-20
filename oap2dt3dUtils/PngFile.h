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

#ifndef PNGFILE_H
#define PNGFILE_H

#include <stdio.h>

#include "IPngFile.h"

namespace oap {
class PngFile : public IPngFile {
 public:
  PngFile();

  virtual ~PngFile();

  virtual bool read(void* buffer, size_t repeat, size_t size);

  virtual void loadBitmap();

  virtual void freeBitmap();

  virtual void close();

  virtual size_t getWidth() const;

  virtual size_t getHeight() const;

  virtual pixel_t* newPixelsVector() const;

 protected:
  virtual bool openInternal(const char* path);

  virtual bool isPngInternal() const;

  virtual pixel_t getPixelInternal(unsigned int x, unsigned int y) const;

 private:
  void copy2dBitmapTo1d(png_bytep* bitmap2d, png_byte* bitmap1d, size_t width,
                        size_t height) const;

  void copyToPixelsVector(oap::pixel_t* pixels, png_byte* bitmap1d,
                          size_t width, size_t height);

  FILE* m_fp;
  png_structp m_png_ptr;
  png_infop m_info_ptr;
  png_bytep* m_bitmap2d;
  png_byte* m_bitmap1d;
  oap::pixel_t* m_pixels;
};
}

#endif  // PNGFILE_H
