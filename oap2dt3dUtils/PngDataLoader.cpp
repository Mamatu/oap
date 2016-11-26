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

#include "PngDataLoader.h"
#include "Exceptions.h"

namespace oap {

PngDataLoader::PngDataLoader(IPngFile* ifile, const std::string& path)
    : m_ifile(ifile) {
  openAndLoad(path);
}

PngDataLoader::PngDataLoader(IPngFile* ifile) : m_ifile(ifile) { load(); }

PngDataLoader::~PngDataLoader() {
  if (m_ifile != NULL) {
    m_ifile->freeBitmap();
  }
}

void PngDataLoader::openAndLoad(const std::string& path) {
  m_ifile->open(path.c_str());

  load();
}

void PngDataLoader::load() {
  m_ifile->loadBitmap();

  m_ifile->close();
}

oap::pixel_t PngDataLoader::getPixel(unsigned int x, unsigned int y) const {
  return m_ifile->getPixel(x, y);
}

void PngDataLoader::getPixelsVector(oap::pixel_t* pixels) const {
  m_ifile->getPixelsVector(pixels);
}

void PngDataLoader::getFloattVector(floatt* vector) const {
  const size_t length = getLength();
  pixel_t* pixels = new pixel_t[length];
  pixel_t max = IPngFile::getPixelMax();
  m_ifile->getPixelsVector(pixels);
  for (size_t fa = 0; fa < length; ++fa) {
    vector[fa] = pixels[fa] / max;
  }
  delete[] pixels;
}

size_t PngDataLoader::getWidth() const { return m_ifile->getWidth(); }

size_t PngDataLoader::getHeight() const { return m_ifile->getHeight(); }

size_t PngDataLoader::getLength() const { return m_ifile->getLength(); }
}
