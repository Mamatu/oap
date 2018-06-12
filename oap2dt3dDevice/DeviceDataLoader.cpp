/*
 * Copyright 2016 - 2018 Marcin Matula
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

#include "DeviceDataLoader.h"
#include "oapCudaMatrixUtils.h"

namespace oap {
DeviceDataLoader::DeviceDataLoader(const Images& images, bool dealocateImages,
                                   bool frugalMode)
    : DataLoader(images, dealocateImages, frugalMode) {}

DeviceDataLoader::~DeviceDataLoader() {}

math::Matrix* DeviceDataLoader::createDeviceMatrix() {
  math::Matrix* host = createMatrix();
  math::Matrix* device = oap::cuda::NewDeviceMatrixCopy(host);
  oap::host::DeleteMatrix(host);
  return device;
}

math::Matrix* DeviceDataLoader::createDeviceRowVector(size_t index) {
  math::Matrix* host = createRowVector(index);
  math::Matrix* device = oap::cuda::NewDeviceMatrixCopy(host);
  oap::host::DeleteMatrix(host);
  return device;
}

math::Matrix* DeviceDataLoader::getDeviceRowVector(size_t index, math::Matrix* dmatrix)
{
  math::Matrix* host = createRowVector(index);
  oap::cuda::CopyHostMatrixToDeviceMatrix(dmatrix, host);
  oap::host::DeleteMatrix(host);
  return dmatrix;
}

math::Matrix* DeviceDataLoader::createDeviceColumnVector(size_t index) {
  math::Matrix* host = createColumnVector(index);
  math::Matrix* device = oap::cuda::NewDeviceMatrixCopy(host);
  oap::host::DeleteMatrix(host);
  return device;
}

math::Matrix* DeviceDataLoader::getDeviceColumnVector(size_t index, math::Matrix* dmatrix)
{
  math::Matrix* host = createColumnVector(index);
  oap::cuda::CopyHostMatrixToDeviceMatrix(dmatrix, host);
  oap::host::DeleteMatrix(host);
  return dmatrix;
}
}
