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

  math::Matrix* createDeviceVector(size_t index);
};
}

#endif  // DEVICEDATALOADER_H