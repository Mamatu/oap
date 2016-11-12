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

#include "IPngFile.h"
#include "Exceptions.h"

namespace oap {

IPngFile::IPngFile() {}

IPngFile::~IPngFile() {}

bool IPngFile::read(void* buffer, size_t size) { return read(buffer, 1, size); }

void IPngFile::open(const char* path) {
  if (openInternal(path) == false) {
    throw oap::exceptions::FileNotExist(path);
  }

  if (isPngInternal() == false) {
    close();
    throw oap::exceptions::FileIsNotPng(path);
  }
}

Pixel IPngFile::getPixel(unsigned int x, unsigned int y) const {
  unsigned int height = getHeight();
  unsigned int width = getWidth();
  if (x >= width || y >= height) {
    throw exceptions::OutOfRange(x, y, width, height);
  }
  return getPixelInternal(x, y);
}
}
