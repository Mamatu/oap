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

  /**
   * @brief Unique pointer for cuda matrix type
   *
   * Examples of use: oapDeviceTests/oapDeviceMatrixUPtrTests.cpp
   */
  class DeviceMatrixUPtr : public oap::MatrixUniquePtr {
    public:
      DeviceMatrixUPtr(math::Matrix* matrix = nullptr, bool bDeallocate = true) :
        oap::MatrixUniquePtr(matrix, oap::cuda::DeleteDeviceMatrix, bDeallocate)
      {
      }
  };

  /**
   * @brief Unique pointer which points into array of cuda matrix pointers
   *
   * This class creates its own matrices array which contains copied pointers.
   * If array was allocated dynamically must be deallocated.
   *
   * Examples of use: oapDeviceTests/oapDeviceMatrixUPtrTests.cpp
   */
  class DeviceMatricesUPtr : public oap::MatricesUniquePtr {
    public:
      DeviceMatricesUPtr(math::Matrix** matrices, unsigned int count) :
        oap::MatricesUniquePtr(matrices, count, oap::cuda::DeleteDeviceMatrix) {}

      DeviceMatricesUPtr(std::initializer_list<math::Matrix*> matrices) :
        oap::MatricesUniquePtr(matrices, oap::cuda::DeleteDeviceMatrix) {}
  };

  template<template<typename, typename> class Container>
  DeviceMatricesUPtr makeDeviceMatricesUPtr(const Container<math::Matrix*, std::allocator<math::Matrix*> >& matrices) {
    return smartptr_utils::makeSmartPtr<DeviceMatricesUPtr>(matrices);
  }

  template<template<typename> class Container>
  DeviceMatricesUPtr makeDeviceMatricesUPtr(const Container<math::Matrix*>& matrices) {
    return smartptr_utils::makeSmartPtr<DeviceMatricesUPtr>(matrices);
  }

  inline DeviceMatricesUPtr makeDeviceMatricesUPtr(math::Matrix** array, size_t count) {
    return smartptr_utils::makeSmartPtr<DeviceMatricesUPtr>(array, count);
  }
}

#endif
