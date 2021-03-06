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

#ifndef DEVICEDATALOADER_H
#define DEVICEDATALOADER_H

#include "ImagesLoader.h"

namespace oap {

class DeviceImagesLoader : public ImagesLoader {
 public:
  DeviceImagesLoader(const Images& images, bool dealocateImages = false,
                   bool frugalMode = true);

  virtual ~DeviceImagesLoader();

  /**
   * @brief Creates device matrix from set of pngImagesLoader
   * @return matrix in device space
   */
  math::ComplexMatrix* createDeviceMatrix();

  math::ComplexMatrix* createDeviceRowVector(size_t index);
  math::ComplexMatrix* getDeviceRowVector(size_t index, math::ComplexMatrix* dmatrix);

  math::ComplexMatrix* createDeviceColumnVector(size_t index);
  math::ComplexMatrix* getDeviceColumnVector(size_t index, math::ComplexMatrix* dmatrix);
  
  math::ComplexMatrix* createDeviceSubMatrix(uintt cindex, uintt rindex, uintt columns, uintt rows);
  math::ComplexMatrix* getDeviceSubMatrix(uintt cindex, uintt rindex, uintt columns, uintt rows, math::ComplexMatrix* dmatrix);
};

}

#endif  // DEVICEDATALOADER_H
