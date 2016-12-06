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

#include "Image.h"
#include "Exceptions.h"

namespace oap {

Image::Image() {}

Image::~Image() {}

bool Image::read(void* buffer, size_t size) { return read(buffer, 1, size); }

void Image::open(const char* path) {
  if (openInternal(path) == false) {
    throw oap::exceptions::FileNotExist(path);
  }

  if (isPngInternal() == false) {
    close();
    throw oap::exceptions::FileIsNotPng(path);
  }
}

pixel_t Image::getPixel(unsigned int x, unsigned int y) const {
  unsigned int height = getHeight();
  unsigned int width = getWidth();
  if (x >= width) {
    throw exceptions::OutOfRange(x, width);
  }
  if (y >= height) {
    throw exceptions::OutOfRange(y, height);
  }
  return getPixelInternal(x, y);
}

size_t Image::getLength() const { return getWidth() * getHeight(); }

pixel_t Image::m_MaxPixel = Image::getPixelMax();

pixel_t Image::convertRgbToPixel(unsigned char r, unsigned char g,
                                    unsigned char b) {
  pixel_t rgb = r;
  rgb = rgb << 8;
  rgb |= g;
  rgb = rgb << 8;
  rgb |= b;
  return rgb;
}

floatt Image::convertPixelToFloatt(pixel_t pixel) {
  return static_cast<floatt>(pixel) / static_cast<floatt>(Image::m_MaxPixel);
}

floatt Image::convertRgbToFloatt(unsigned char r, unsigned char g,
                                    unsigned char b) {
  pixel_t pixel = convertRgbToPixel(r, g, b);
  return convertPixelToFloatt(pixel);
}

pixel_t Image::getPixelMax() { return convertRgbToPixel(255, 255, 255); }
}
