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
#include "DeviceMatrixModules.h"
#include "Exceptions.h"
#include "PngFile.h"
#include "Config.h"

#include <sstream>

namespace oap {

DataLoader::DataLoader(const Images& images, bool dealocateImages)
    : m_images(images), m_deallocateImages(dealocateImages) {
  load();
}

DataLoader::~DataLoader() { destroyImages(); }

size_t DataLoader::getWidth() const { return m_images[0]->getWidth(); }

size_t DataLoader::getHeight() const { return m_images[0]->getHeight(); }

size_t DataLoader::getLength() const { return m_images[0]->getLength(); }

math::Matrix* DataLoader::createMatrix(const Images& images) {
  const size_t refLength = images[0]->getLength();
  floatt* floatsvec = new floatt[refLength];

  math::Matrix* hostMatrix = host::NewReMatrix(images.size(), refLength);

  for (size_t fa = 0; fa < images.size(); ++fa) {
    Image* it = images[fa];
    const size_t length = it->getLength();
    if (refLength != length) {
      delete[] floatsvec;
      host::DeleteMatrix(hostMatrix);
      throw oap::exceptions::NotIdenticalLengths(refLength, length);
    }
    it->getFloattVector(floatsvec);
    host::SetReVector(hostMatrix, fa, floatsvec, refLength);
  }

  delete[] floatsvec;

  return hostMatrix;
}

math::Matrix* DataLoader::createMatrix() const {
  return DataLoader::createMatrix(m_images);
}

math::Matrix* DataLoader::createDeviceMatrix() const {
  math::Matrix* host = createMatrix();
  math::Matrix* device = device::NewDeviceMatrixCopy(host);
  host::DeleteMatrix(host);
  return device;
}

ArnUtils::MatrixInfo DataLoader::createMatrixInfo() const {
  const uintt width = m_images.size();
  const uintt height = m_images[0]->getLength();

  return ArnUtils::MatrixInfo(true, false, width, height);
}

std::string DataLoader::constructAbsPath(const std::string& dirPath) {
  std::string path;

  if (dirPath[0] != '/') {
    path = utils::Config::getPathInOap(dirPath.c_str());
  } else {
    path = dirPath;
  }

  if (path[path.length() - 1] != '/') {
    path = path + '/';
  }

  return path;
}

std::string DataLoader::constructImagePath(const std::string& absPath,
                                           const std::string& nameBase,
                                           size_t index, size_t count) {
  std::string imagePath = absPath;
  imagePath = imagePath + nameBase;

  std::stringstream sstream;

  size_t width = 0;
  size_t temp = count;
  while (temp >= 10) {
    ++width;
    temp = temp / 10;
  }

  if (width > 0) {
    sstream.width(width);
    sstream.fill('0');
  }

  sstream << index;

  imagePath = imagePath + sstream.str();

  return imagePath;
}

void DataLoader::load() {
  for (size_t fa = 0; fa < m_images.size(); ++fa) {
    Image* image = m_images[fa];

    image->open();

    image->loadBitmap();

    image->close();
  }
}

void DataLoader::destroyImages() {
  for (size_t fa = 0; fa < m_images.size(); ++fa) {
    if (m_images[fa] != NULL) {
      m_images[fa]->freeBitmap();
    }
    if (m_deallocateImages) {
      delete m_images[fa];
    }
  }
  m_images.clear();
}
}
