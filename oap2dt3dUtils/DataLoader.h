/*
 * Copyright 2016, 2017 Marcin Matula
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

#include "MatrixInfo.h"

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

  class Info {

      std::string m_dirPath;
      std::string m_nameBase;
      size_t m_loadFilesCount;
      bool m_frugalMode;

    public:
      Info(const std::string& dirPath, const std::string& nameBase, size_t loadFilesCount, bool frugalMode) :
        m_dirPath(dirPath), m_nameBase(nameBase), m_loadFilesCount(loadFilesCount), m_frugalMode(frugalMode) {}

      friend class DataLoader;
  };

  /**
   * @brief Creates specified DataLoader instance and loads
   *        images from specified path. Images in dir should have
   *        name contained nameBase_N.type, where nameBase is some
   *        string of chars, N - is number (without padding),
   *        type - is type specified by T parameter.
   *
   *        For example: in directory contained 100 images whose
   *        nameBase is "image" and T is png than images should be:
   *        image_0.png
   *        image_1.png
   *        image_2.png
   *        .
   *        .
   *        .
   *        image_99.png
   *
   * @param T               derived class of Image class
   * @param DL              derived class of or DataLoader class
   * @param dirPath         path to directory with images
   * @param nameBase        core of file name
   * @param loadFilesCount  count of images to load
   * @param frugalMode      if true images content will be loaded
   *                        and free after use, otherwise image contant
   *                        will be loaded one time and kept in memory
   *
   */
  template <typename T, typename DL = oap::DataLoader>
  static DL* createDataLoader(const std::string& dirPath, const std::string& nameBase,
                              size_t loadFilesCount, bool frugalMode = true)
  {
    const std::string& imageBasePath = constructAbsPath(dirPath);
    oap::Images images =
        createImagesVector<T>(imageBasePath, nameBase, loadFilesCount);

    return new DL(images, true, frugalMode);
  }

  /**
   * @brief Creates DataLoader from information from Info instance.
   */
  template <typename T, typename DL = oap::DataLoader>
  static DL* createDataLoader(const oap::DataLoader::Info& info)
  {
    return createDataLoader<T, DL>(info.m_dirPath, info.m_nameBase,
                            info.m_loadFilesCount, info.m_frugalMode);
  }

  virtual ~DataLoader();

  /**
   * @brief Creates matrix from sets of pngDataLoader
   * @return matrix in host space
   */
  math::Matrix* createMatrix();

  math::Matrix* createMatrix(uintt index, uintt length);

  math::Matrix* createColumnVector(size_t index);

  math::Matrix* createRowVector(size_t index);

  /**
   * @brief Gets Matrxinfo from set of pngDataLoader
   * @return
   */
  math::MatrixInfo getMatrixInfo() const;

 protected:
  static std::string constructAbsPath(const std::string& basePath);

  static std::string constructImagePath(const std::string& absPath,
                                        const std::string& nameBase,
                                        size_t index);

  template <typename T>
  static oap::Images createImagesVector(const std::string& imageAbsPath,
                                        const std::string& nameBase,
                                        size_t loadFilesCount) {
    oap::Images images;

    for (size_t fa = 0; fa < loadFilesCount; ++fa) {
      std::string imagePath =
          constructImagePath(imageAbsPath, nameBase, fa);

      Image* image = new T(imagePath);

      images.push_back(image);
    }

    return images;
  }

 protected:
  size_t getImagesCount() const;

  oap::Image* getImage(size_t index) const;

 private:
  Images m_images;

  bool m_deallocateImages;
  bool m_frugalMode;

  std::string m_matrixFileDir;
  std::string m_file;

  void loadColumnVector(math::Matrix* matrix, size_t column, floatt* vec,
                        size_t imageIndex);

  void load();
  void executeLoadProcess(const oap::OptSize& optWidthRef,
                          const oap::OptSize& optHeightRef, size_t begin,
                          size_t end);
  void loadImage(oap::Image* iamge) const;
  void freeBitmaps(size_t begin, size_t end);
  void forceOutputSizes(const oap::OptSize& width, const oap::OptSize& height,
                        size_t begin, size_t end);

  void cleanImageStuff();

  void createDataMatrixFiles();

  uintptr_t getId() const { return reinterpret_cast<uintptr_t>(this); }
};
}
#endif  // PNGLOADER_H
