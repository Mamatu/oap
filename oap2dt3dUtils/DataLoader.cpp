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

#include "DataLoader.h"
#include "Exceptions.h"
#include "PngFile.h"

namespace oap {

DataLoader::DataLoader(Image* ifile, const std::string& path, bool deallocateIFile)
    : m_ifile(ifile), m_deallocateIFile(deallocateIFile) {
  openAndLoad(path);
}

DataLoader::DataLoader(Image* ifile)
    : m_ifile(ifile), m_deallocateIFile(false) {
  load();
}

DataLoader::~DataLoader() {
  if (m_ifile != NULL) {
    m_ifile->freeBitmap();
  }
  if (m_deallocateIFile) {
    delete m_ifile;
  }
}

void DataLoader::openAndLoad(const std::string& path) {
  m_ifile->open(path.c_str());

  load();
}

void DataLoader::load() {
  m_ifile->loadBitmap();

  m_ifile->close();
}

oap::pixel_t DataLoader::getPixel(unsigned int x, unsigned int y) const {
  return m_ifile->getPixel(x, y);
}

void DataLoader::getPixelsVector(oap::pixel_t* pixels) const {
  m_ifile->getPixelsVector(pixels);
}

void DataLoader::getFloattVector(floatt* vector) const {
  const size_t length = getLength();
  pixel_t* pixels = new pixel_t[length];
  pixel_t max = Image::getPixelMax();
  m_ifile->getPixelsVector(pixels);
  for (size_t fa = 0; fa < length; ++fa) {
    vector[fa] = oap::Image::convertPixelToFloatt(pixels[fa]);
  }
  delete[] pixels;
}

size_t DataLoader::getWidth() const { return m_ifile->getWidth(); }

size_t DataLoader::getHeight() const { return m_ifile->getHeight(); }

size_t DataLoader::getLength() const { return m_ifile->getLength(); }
}
