/*
 * Copyright 2016 - 2018 Marcin Matula
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

#include <png.h>

#include <stdio.h>
#include "Image.h"

namespace oap {
class PngFile : public Image {
 public:
  PngFile(const std::string& path, bool truncateImage = true);

  virtual ~PngFile();

  virtual bool read(void* buffer, size_t repeat, size_t size);

  virtual oap::OptSize getWidth() const;

  virtual oap::OptSize getHeight() const;

  virtual void forceOutputWidth(const oap::OptSize& optWidth);

  virtual void forceOutputHeight(const oap::OptSize& optHeight);

  virtual oap::OptSize getOutputWidth() const;

  virtual oap::OptSize getOutputHeight() const;

  virtual std::string getSufix() const;

  bool save(const std::string& path);

  bool save(const std::string& prefix, const std::string& path);

 protected:
  virtual void closeProtected();

  virtual void loadBitmapProtected();

  void loadBitmapBuffers();

  void convertRawdataToBitmap1D();

  void destroyTmpData();

  void setAutomaticDestroyTmpData(bool destroyTmp);

  virtual void freeBitmapProtected();

  virtual void getPixelsVectorProtected(pixel_t* pixels) const;

  virtual bool openProtected(const std::string& path);

  bool isCorrectFormat() const;

  virtual pixel_t getPixelProtected(unsigned int x, unsigned int y) const;

  png_bytep* copyBitmap(const OptSize& width, const OptSize& height);

  png_bytep* createBitmap2D(size_t width, size_t height,
                            size_t colorsCount) const;

  void destroyBitmap2d(png_bytep* bitmap2d, size_t height) const;

  png_byte* createBitmap1dFrom2d(png_bytep* bitmap2d, const OptSize& optWidth,
                                 const OptSize& optHeight, size_t colorsCount);

  oap::pixel_t* createPixelsVectorFrom1d(png_byte* bitmap1d, size_t width,
                                         size_t height, size_t colorsCount);

  template <typename T>
  void destroyBuffer(T* buffer) {
    if (buffer != NULL) {
      delete[] buffer;
    }
  }

 private:
  void calculateColorsCount();

  void calculateOutputSizes(size_t width, size_t height);

  void destroyBitmap1d();

  void destroyPixels();

  oap::OptSize m_optWidth;
  oap::OptSize m_optHeight;

  FILE* m_fp;
  png_structp m_png_ptr;
  png_infop m_info_ptr;
  png_bytep* m_bitmap2d;
  png_byte* m_bitmap1d;
  oap::pixel_t* m_pixels;

  bool m_destroyTmp;

  size_t m_colorsCount;
  bool m_truncateImage;
};
}

#endif  // PNGFILE_H
