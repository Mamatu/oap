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

#ifndef DATALOADER_H
#define DATALOADER_H

#include <string>
#include <vector>

#include "ArnoldiUtils.h"

#include "Image.h"
#include "Math.h"
#include "Matrix.h"
#include "Exceptions.h"

namespace oap {

typedef std::vector<oap::Image*> Images;

class DataLoader {
 public:
  DataLoader(const Images& images, bool dealocateImages = false,
             bool frugalMode = true);

  template <typename T, typename DL>
  static DL* createDataLoader(const std::string& dirPath,
                              const std::string& nameBase, size_t loadCount,
                              size_t count, bool frugalMode) {
    const std::string& imageBasePath = constructAbsPath(dirPath);
    oap::Images images =
        createImagesVector<T>(imageBasePath, nameBase, loadCount, count);

    return new DL(images, true, frugalMode);
  }

  template <typename T, typename DL>
  static DL* createDataLoader(const std::string& dirPath,
                              const std::string& nameBase, size_t loadCount,
                              bool frugalMode = true) {
    const std::string& imageBasePath = constructAbsPath(dirPath);
    oap::Images images =
        createImagesVector<T>(imageBasePath, nameBase, loadCount, loadCount);

    return new DL(images, true, frugalMode);
  }

  template <typename T>
  static oap::DataLoader* createDataLoader(const std::string& dirPath,
                                           const std::string& nameBase,
                                           size_t loadCount, size_t count,
                                           bool frugalMode) {
    return DataLoader::createDataLoader<T, oap::DataLoader>(
        dirPath, nameBase, loadCount, count, frugalMode);
  }

  template <typename T>
  static oap::DataLoader* createDataLoader(const std::string& dirPath,
                                           const std::string& nameBase,
                                           size_t loadCount,
                                           bool frugalMode = true) {
    return DataLoader::createDataLoader<T, oap::DataLoader>(
        dirPath, nameBase, loadCount, frugalMode);
  }

  virtual ~DataLoader();

  /**
   * @brief Creates matrix from sets of pngDataLoader
   * @return matrix in host space
   */
  math::Matrix* createMatrix();

  math::Matrix* createVector(size_t index);

  /**
   * @brief Creates Matrxinfo from set of pngDataLoader
   * @return
   */
  ArnUtils::MatrixInfo createMatrixInfo() const;

 protected:
  static std::string constructAbsPath(const std::string& basePath);

  static std::string constructImagePath(const std::string& absPath,
                                        const std::string& nameBase,
                                        size_t index, size_t count);

  template <typename T>
  static oap::Images createImagesVector(const std::string& imageAbsPath,
                                        const std::string& nameBase,
                                        size_t loadCount, size_t count) {
    oap::Images images;

    for (size_t fa = 0; fa < loadCount; ++fa) {
      const std::string& imagePath =
          constructImagePath(imageAbsPath, nameBase, fa, count);

      Image* image = new T(imagePath);

      images.push_back(image);
    }

    return images;
  }

 private:
  Images m_images;
  bool m_deallocateImages;
  bool m_frugalMode;

  void loadVector(math::Matrix* matrix, size_t column, floatt* vec, size_t imageIndex);

  void load();
  void executeLoadProcess(const oap::OptSize& optWidthRef,
                          const oap::OptSize& optHeightRef, size_t begin,
                          size_t end);
  void loadImage(oap::Image* iamge) const;
  void freeBitmaps(size_t begin, size_t end);
  void forceOutputSizes(const oap::OptSize& width, const oap::OptSize& height,
                        size_t begin, size_t end);

  void cleanImageStuff();
};
}
#endif  // PNGLOADER_H
