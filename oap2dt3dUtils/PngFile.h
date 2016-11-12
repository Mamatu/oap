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

  virtual unsigned int getWidth() const;

  virtual unsigned int getHeight() const;

 protected:
  virtual bool openInternal(const char* path);

  virtual bool isPngInternal() const;

  virtual Pixel getPixelInternal(unsigned int x, unsigned int y) const;

 private:
  FILE* m_fp;
  png_structp m_png_ptr;
  png_infop m_info_ptr;
  png_bytep* m_bitmap;
};
}

#endif  // PNGFILE_H
