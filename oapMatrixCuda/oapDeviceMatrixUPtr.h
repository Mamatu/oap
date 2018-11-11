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

#ifndef OAP_DEVICEMATRIXUPTR_H
#define OAP_DEVICEMATRIXUPTR_H

#include "Math.h"
#include "oapCudaMatrixUtils.h"

#include "oapMatrixSPtr.h"

namespace oap {
  class DeviceMatrixUPtr : public oap::MatrixUniquePtr {
    public:
      DeviceMatrixUPtr(math::Matrix* matrix = nullptr) : oap::MatrixUniquePtr(matrix,
        [this](const math::Matrix* matrix) { debugInfo("Destroy: DeviceMatrixUPtr = %p matrix = %p", this, matrix); oap::cuda::DeleteDeviceMatrix(matrix); })
      {
        debugInfo("Create: DeviceMatrixUPtr = %p matrix = %p", this, matrix);
      }
  };

  class DeviceMatricesUPtr : public oap::MatricesUniquePtr {
    public:
      DeviceMatricesUPtr(math::Matrix** matrices, unsigned int count) :
        oap::MatricesUniquePtr(matrices, deleters::MatricesDeleter(count, oap::cuda::DeleteDeviceMatrix)) {}

      DeviceMatricesUPtr(std::initializer_list<math::Matrix*> matrices) :
        oap::MatricesUniquePtr(matrices, deleters::MatricesDeleter(smartptr_utils::getElementsCount(matrices), oap::cuda::DeleteDeviceMatrix)) {}
  };

  template<template<typename, typename> class Container>
  DeviceMatricesUPtr makeDeviceMatricesUPtr(const Container<math::Matrix*, std::allocator<math::Matrix*> >& matrices) {
    return smartptr_utils::makeSmartPtr<DeviceMatricesUPtr>(matrices);
  }
}

#endif
