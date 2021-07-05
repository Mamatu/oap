/*
 * Copyright 2016 - 2021 Marcin Matula
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

#ifndef OAP_IMAGES_LOADER_H
#define OAP_IMAGES_LOADER_H

#include <string>
#include <vector>

#include "MatrixInfo.hpp"

#include "Image.hpp"
#include "Math.hpp"
#include "Matrix.hpp"
#include "Exceptions.hpp"

namespace oap {

typedef std::vector<oap::Image*> Images;

class ImagesLoader {
 public:
  ImagesLoader (const Images& images, bool dealocateImages = false, bool lazyMode = true);

  class Info {

      std::string m_dirPath;
      std::string m_nameBase;
      size_t m_loadFilesCount;
      bool m_lazyMode;

    public:
      Info() : m_dirPath(""), m_nameBase(""), m_loadFilesCount(0), m_lazyMode(false)
      {}

      Info(const std::string& dirPath, const std::string& nameBase, size_t loadFilesCount, bool lazyMode) :
        m_dirPath(dirPath), m_nameBase(nameBase), m_loadFilesCount(loadFilesCount), m_lazyMode(lazyMode) {}

      bool isValid() const
      {
        return !m_dirPath.empty() && !m_nameBase.empty() && m_loadFilesCount > 0;
      }

      friend class ImagesLoader;
  };

  /**
   * @brief Creates specified ImagesLoader instance and loads
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
   * @param DL              derived class of or ImagesLoader class
   * @param dirPath         path to directory with images
   * @param nameBase        core of file name
   * @param loadFilesCount  count of images to load
   * @param lazyMode      if true images content will be loaded
   *                        and free after use, otherwise image contant
   *                        will be loaded one time and kept in memory
   *
   */
  template <typename T, typename DL = oap::ImagesLoader>
  static DL* createImagesLoader(const std::string& dirPath, const std::string& nameBase,
                              size_t loadFilesCount, bool lazyMode = true)
  {
    const std::string& imageBasePath = constructAbsPath(dirPath);
    oap::Images images =
        createImagesVector<T>(imageBasePath, nameBase, loadFilesCount);

    return new DL(images, true, lazyMode);
  }

  /**
   * @brief Creates ImagesLoader from information from Info instance.
   */
  template <typename T, typename DL = oap::ImagesLoader>
  static DL* createImagesLoader(const oap::ImagesLoader::Info& info)
  {
    return createImagesLoader<T, DL>(info.m_dirPath, info.m_nameBase,
                            info.m_loadFilesCount, info.m_lazyMode);
  }

  virtual ~ImagesLoader();

  /**
   * @brief Creates matrix from sets of pngImagesLoader
   * @return matrix in host space
   */
  math::ComplexMatrix* createMatrix();

  math::ComplexMatrix* createMatrix(uintt index, uintt length);

  math::ComplexMatrix* createSubMatrix(uintt cindex, uintt rindex, uintt columns, uintt rows);

  math::ComplexMatrix* createColumnVector(size_t index);

  math::ComplexMatrix* createRowVector(size_t index);

  /**
   * @brief Gets Matrxinfo from set of pngImagesLoader
   * @return
   */
  math::MatrixInfo getMatrixInfo() const;

 protected:
  static std::string constructAbsPath (const std::string& basePath);

  static std::string constructImagePath (const std::string& absPath, const std::string& nameBase, size_t index);

  template <typename T>
  static oap::Images createImagesVector (const std::string& imageAbsPath, const std::string& nameBase, size_t loadFilesCount)
  {
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
  bool m_lazyMode;

  std::string m_matrixFileDir;
  std::string m_file;

  void loadColumnVector(math::ComplexMatrix* matrix, size_t column, floatt* vec,
                        size_t imageIndex);

  void load();
  void executeLoadProcess (const oap::ImageSection& optWidthRef, const oap::ImageSection& optHeightRef, size_t begin, size_t end);
  void loadImage(oap::Image* iamge) const;
  void freeBitmaps(size_t begin, size_t end);
  void forceOutputSizes(const oap::ImageSection& width, const oap::ImageSection& height, size_t begin, size_t end);

  void cleanImageStuff();

  void createDataMatrixFiles();

  uintptr_t getId() const { return reinterpret_cast<uintptr_t>(this); }
};
}
#endif  // PNGLOADER_H
