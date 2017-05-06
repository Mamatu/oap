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
#include "Config.h"

#include "HostMatrixUtils.h"

#include <sstream>
#include <functional>
#include <memory>

namespace oap {

DataLoader::DataLoader(const Images& images, bool dealocateImages,
                       bool frugalMode)
    : m_images(images),
      m_deallocateImages(dealocateImages),
      m_frugalMode(frugalMode),
      m_matrixFileDir("/tmp") {
  load();
}

DataLoader::~DataLoader() { cleanImageStuff(); }

math::Matrix* DataLoader::createMatrix() {
  return createMatrix(0, m_images.size());
}

math::Matrix* DataLoader::createMatrix(uintt index, uintt length) {
  const size_t refLength = m_images[0]->getLength();
  std::unique_ptr<floatt[]> floatsvecUPtr(new floatt[refLength]);
  floatt* floatsvec = floatsvecUPtr.get();

  math::Matrix* hostMatrix = host::NewReMatrix(length, refLength);

  try {
    for (size_t fa = index; fa < index + length; ++fa) {
      loadColumnVector(hostMatrix, fa, floatsvec, fa);
    }
  } catch (const oap::exceptions::NotIdenticalLengths&) {
    host::DeleteMatrix(hostMatrix);
    cleanImageStuff();
    throw;
  }

  return hostMatrix;
}

math::Matrix* DataLoader::createColumnVector(size_t index) {
  const size_t refLength = m_images[0]->getLength();
  std::unique_ptr<floatt[]> floatsvecUPtr(new floatt[refLength]);
  floatt* floatsvec = floatsvecUPtr.get();

  math::Matrix* hostMatrix = host::NewReMatrix(1, refLength);

  try {
    loadColumnVector(hostMatrix, 0, floatsvec, index);
  } catch (const oap::exceptions::NotIdenticalLengths&) {
    host::DeleteMatrix(hostMatrix);
    cleanImageStuff();
    throw;
  }

  return hostMatrix;
}

math::Matrix* DataLoader::createRowVector(size_t index) {
  createDataMatrixFiles();

  return host::ReadRowVector(m_file, index);
}

math::MatrixInfo DataLoader::getMatrixInfo() const {
  const uintt width = m_images.size();
  const uintt height = m_images[0]->getLength();

  return math::MatrixInfo(true, false, width, height);
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

void DataLoader::loadColumnVector(math::Matrix* matrix, size_t column, floatt* vec,
                               size_t imageIndex) {
  const size_t refLength = m_images[0]->getLength();

  Image* it = m_images[imageIndex];
  const size_t length = it->getLength();

  if (refLength != length) {
    throw oap::exceptions::NotIdenticalLengths(refLength, length);
  }

  if (m_frugalMode) {
    loadImage(it);
  }

  it->getFloattVector(vec);
  if (m_frugalMode) {
    it->freeBitmap();
  }

  host::SetReVector(matrix, column, vec, refLength);
}

void DataLoader::load() {
  oap::OptSize optWidth;
  oap::OptSize optHeight;
  executeLoadProcess(optWidth, optHeight, 0, m_images.size());
}

void DataLoader::executeLoadProcess(const oap::OptSize& optWidthRef,
                                    const oap::OptSize& optHeightRef,
                                    size_t begin, size_t end) {
  oap::OptSize refOptWidth = optWidthRef;
  oap::OptSize refOptHeight = optHeightRef;

  std::function<void(Image*, const oap::OptSize&)> setWidthFunc =
      &Image::forceOutputWidth;

  std::function<void(Image*, const oap::OptSize&)> setHeightFunc =
      &Image::forceOutputHeight;

  bool needreload = false;
  bool previousneedreload = false;

  auto verifyReloadConds = [&needreload, &previousneedreload](
      std::function<void(Image*, const oap::OptSize&)>& setter,
      oap::Image* image, oap::OptSize& refOptSize, oap::OptSize& imageOptSize) {
    if (refOptSize.optSize == 0) {
      refOptSize = imageOptSize;
    } else if (imageOptSize.optSize < refOptSize.optSize) {
      setter(image, refOptSize);
      needreload = true;
    } else if (imageOptSize.optSize > refOptSize.optSize) {
      previousneedreload = true;
      refOptSize = imageOptSize;
    }
  };

  for (size_t fa = begin; fa < end; ++fa) {
    Image* image = m_images[fa];

    loadImage(image);

    oap::OptSize imageOptWidth = image->getOutputWidth();

    oap::OptSize imageOptHeight = image->getOutputHeight();

    needreload = false;
    previousneedreload = false;

    verifyReloadConds(setWidthFunc, image, refOptWidth, imageOptWidth);
    verifyReloadConds(setHeightFunc, image, refOptHeight, imageOptHeight);

    if (needreload) {
      image->freeBitmap();
      loadImage(image);
    }

    if (previousneedreload) {
      freeBitmaps(begin, fa);
      forceOutputSizes(refOptWidth, refOptHeight, begin, fa);
      executeLoadProcess(refOptWidth, refOptHeight, begin, fa);
    }
    if (m_frugalMode) {
      image->freeBitmap();
    }
  }
}

void DataLoader::loadImage(oap::Image* image) const {
  image->open();
  image->loadBitmap();
  image->close();
}

void DataLoader::freeBitmaps(size_t begin, size_t end) {
  for (size_t fa = begin; fa < end; ++fa) {
    m_images[fa]->freeBitmap();
  }
}

void DataLoader::forceOutputSizes(const oap::OptSize& width,
                                  const oap::OptSize& height, size_t begin,
                                  size_t end) {
  for (size_t fa = begin; fa < end; ++fa) {
    m_images[fa]->forceOutputWidth(width);
    m_images[fa]->forceOutputHeight(height);
  }
}

void DataLoader::cleanImageStuff() {
  for (oap::Image* image : m_images) {
    if (image != NULL) {
      image->freeBitmap();
    }
    if (m_deallocateImages) {
      delete image;
    }
  }
  m_images.clear();
}

void DataLoader::createDataMatrixFiles() {
  if (m_file.length() == 0) {
    math::Matrix* matrix = createMatrix();

    std::string filePath = m_matrixFileDir;

    filePath += "/dataloader_matrix";

    filePath += std::to_string(getId());

    bool status = host::WriteMatrix(filePath, matrix);

    if (status == true) {
      m_file = filePath;
    }

    host::DeleteMatrix(matrix);
  }
}
}
