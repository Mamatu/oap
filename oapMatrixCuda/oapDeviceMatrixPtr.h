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

#ifndef OAP_DEVICEMATRIXPTR_H
#define OAP_DEVICEMATRIXPTR_H

#include "Math.h"
#include "oapCudaMatrixUtils.h"

#include "oapMatrixSPtr.h"

namespace oap {

  /**
   * @brief Shared pointer for cuda matrix type
   *
   * Examples of use: oapDeviceTests/oapDeviceMatrixPtrTests.cpp
   */
  class DeviceMatrixPtr : public oap::MatrixSharedPtr {
    public:
      DeviceMatrixPtr(math::Matrix* matrix = nullptr) : oap::MatrixSharedPtr(matrix,
      [this](const math::Matrix* matrix) { logTrace("Destroy: DeviceMatrixPtr = %p matrix = %p", this, matrix); oap::cuda::DeleteDeviceMatrix(matrix); })
      {
        logTrace("Create: DeviceMatrixPtr = %p matrix = %p", this, matrix);
      }
  };

  /**
   * @brief Shared pointer which points into array of cuda matrix pointers
   *
   * This class creates its own matrices array which contains copied pointers.
   * If array was allocated dynamically must be deallocated.
   *
   * Examples of use: oapDeviceTests/oapDeviceMatrixPtrTests.cpp
   */
  class DeviceMatricesPtr : public oap::MatricesSharedPtr {
    public:
      DeviceMatricesPtr(math::Matrix** matrices, unsigned int count) :
        oap::MatricesSharedPtr (matrices, count, oap::cuda::DeleteDeviceMatrix) {}

      DeviceMatricesPtr(std::initializer_list<math::Matrix*> matrices) :
        oap::MatricesSharedPtr (matrices, oap::cuda::DeleteDeviceMatrix) {}

    private:
      DeviceMatricesPtr(math::Matrix** matrices, unsigned int count, bool bCopyArray) :
        oap::MatricesSharedPtr (matrices, count, oap::cuda::DeleteDeviceMatrix, bCopyArray) {}

      template<class SmartPtr, template<typename, typename> class Container, typename T>
      friend SmartPtr smartptr_utils::makeSmartPtr(const Container<T, std::allocator<T> >& container);
  };

  template<template<typename, typename> class Container>
  DeviceMatricesPtr makeDeviceMatricesPtr(const Container<math::Matrix*, std::allocator<math::Matrix*> >& matrices) {
    return smartptr_utils::makeSmartPtr<DeviceMatricesPtr>(matrices);
  }

  template<template<typename> class Container>
  DeviceMatricesPtr makeDeviceMatricesPtr(const Container<math::Matrix*>& matrices) {
    return smartptr_utils::makeSmartPtr<DeviceMatricesPtr>(matrices);
  }

  inline DeviceMatricesPtr makeDeviceMatricesPtr(math::Matrix** array, size_t count) {
    return smartptr_utils::makeSmartPtr<DeviceMatricesPtr>(array, count);
  }
}

#endif
