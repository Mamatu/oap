/*
 * Copyright 2016 - 2019 Marcin Matula
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

#include "DataLoader.h"

namespace oap {

class DeviceDataLoader : public DataLoader {
 public:
  DeviceDataLoader(const Images& images, bool dealocateImages = false,
                   bool frugalMode = true);

  virtual ~DeviceDataLoader();

  /**
   * @brief Creates device matrix from set of pngDataLoader
   * @return matrix in device space
   */
  math::Matrix* createDeviceMatrix();

  math::Matrix* createDeviceRowVector(size_t index);
  math::Matrix* getDeviceRowVector(size_t index, math::Matrix* dmatrix);

  math::Matrix* createDeviceColumnVector(size_t index);
  math::Matrix* getDeviceColumnVector(size_t index, math::Matrix* dmatrix);
  
  math::Matrix* createDeviceSubMatrix(uintt cindex, uintt rindex, uintt columns, uintt rows);
  math::Matrix* getDeviceSubMatrix(uintt cindex, uintt rindex, uintt columns, uintt rows, math::Matrix* dmatrix);
};

}

#endif  // DEVICEDATALOADER_H
