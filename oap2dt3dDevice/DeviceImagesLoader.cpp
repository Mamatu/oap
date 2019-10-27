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

#include "DeviceImagesLoader.h"

#include "oapCudaMatrixUtils.h"
#include "oapHostMatrixUtils.h"

namespace oap {
DeviceImagesLoader::DeviceImagesLoader(const Images& images, bool dealocateImages,
                                   bool frugalMode)
    : ImagesLoader(images, dealocateImages, frugalMode) {}

DeviceImagesLoader::~DeviceImagesLoader() {}

math::Matrix* DeviceImagesLoader::createDeviceMatrix() {
  math::Matrix* host = createMatrix();
  math::Matrix* device = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(host);
  oap::host::DeleteMatrix(host);
  return device;
}

math::Matrix* DeviceImagesLoader::createDeviceRowVector(size_t index) {
  math::Matrix* host = createRowVector(index);
  math::Matrix* device = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(host);
  oap::host::DeleteMatrix(host);
  return device;
}

math::Matrix* DeviceImagesLoader::getDeviceRowVector(size_t index, math::Matrix* dmatrix)
{
  math::Matrix* host = createRowVector(index);
  oap::cuda::CopyHostMatrixToDeviceMatrix(dmatrix, host);
  oap::host::DeleteMatrix(host);
  return dmatrix;
}

math::Matrix* DeviceImagesLoader::createDeviceColumnVector(size_t index) {
  math::Matrix* host = createColumnVector(index);
  math::Matrix* device = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(host);
  oap::host::DeleteMatrix(host);
  return device;
}

math::Matrix* DeviceImagesLoader::getDeviceColumnVector(size_t index, math::Matrix* dmatrix)
{
  math::Matrix* host = createColumnVector(index);
  oap::cuda::CopyHostMatrixToDeviceMatrix(dmatrix, host);
  oap::host::DeleteMatrix(host);
  return dmatrix;
}

math::Matrix* DeviceImagesLoader::createDeviceSubMatrix(uintt cindex, uintt rindex, uintt columns, uintt rows)
{
  math::Matrix* host = createSubMatrix (cindex, rindex, columns, rows);
  math::Matrix* device = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(host);
  oap::host::DeleteMatrix(host);
  return device;
}

math::Matrix* DeviceImagesLoader::getDeviceSubMatrix(uintt cindex, uintt rindex, uintt columns, uintt rows, math::Matrix* dmatrix)
{
  uintt columns1 = oap::cuda::GetColumns (dmatrix);
  uintt rows1 = oap::cuda::GetRows (dmatrix);

  math::Matrix* host = createSubMatrix (cindex, rindex, columns, rows);

  if (columns != columns1 || rows != rows1)
  {
    oap::cuda::DeleteDeviceMatrix (dmatrix);
    dmatrix = oap::cuda::NewDeviceMatrixHostRef (host);
  }

  oap::cuda::CopyHostMatrixToDeviceMatrix (dmatrix, host);
  oap::host::DeleteMatrix(host);

  return dmatrix;
}

}
