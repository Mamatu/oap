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

#ifndef OAP_DEVICEMATRIXPTR_H
#define OAP_DEVICEMATRIXPTR_H

#include "Math.h"
#include "oapCudaMatrixUtils.h"

#include "oapMatrixSPtr.h"

namespace oap {

  /**
   * @brief Shared pointer for cuda matrix type
   *
   * Examples of use: oapDeviceTests/oapDeviceComplexMatrixPtrTests.cpp
   */
  class DeviceComplexMatrixPtr : public oap::ComplexMatrixSharedPtr {
    public:
      DeviceComplexMatrixPtr(math::ComplexMatrix* matrix = nullptr) : oap::ComplexMatrixSharedPtr(matrix,
      [this](const math::ComplexMatrix* matrix) { logTrace("Destroy: DeviceComplexMatrixPtr = %p matrix = %p", this, matrix); oap::cuda::DeleteDeviceMatrix(matrix); })
      {
        logTrace("Create: DeviceComplexMatrixPtr = %p matrix = %p", this, matrix);
      }
  };

  /**
   * @brief Shared pointer which points into array of cuda matrix pointers
   *
   * This class creates its own matrices array which contains copied pointers.
   * If array was allocated dynamically must be deallocated.
   *
   * Examples of use: oapDeviceTests/oapDeviceComplexMatrixPtrTests.cpp
   */
  class DeviceComplexMatricesPtr : public oap::ComplexMatricesSharedPtr {
    public:
      DeviceComplexMatricesPtr(math::ComplexMatrix** matrices, unsigned int count) :
        oap::ComplexMatricesSharedPtr (matrices, count, oap::cuda::DeleteDeviceMatrix) {}

      DeviceComplexMatricesPtr(std::initializer_list<math::ComplexMatrix*> matrices) :
        oap::ComplexMatricesSharedPtr (matrices, oap::cuda::DeleteDeviceMatrix) {}

    private:
      DeviceComplexMatricesPtr(math::ComplexMatrix** matrices, unsigned int count, bool bCopyArray) :
        oap::ComplexMatricesSharedPtr (matrices, count, oap::cuda::DeleteDeviceMatrix, bCopyArray) {}

      template<class SmartPtr, template<typename, typename> class Container, typename T>
      friend SmartPtr smartptr_utils::makeSmartPtr(const Container<T, std::allocator<T> >& container);
  };

  template<template<typename, typename> class Container>
  DeviceComplexMatricesPtr makeDeviceComplexMatricesPtr(const Container<math::ComplexMatrix*, std::allocator<math::ComplexMatrix*> >& matrices) {
    return smartptr_utils::makeSmartPtr<DeviceComplexMatricesPtr>(matrices);
  }

  template<template<typename> class Container>
  DeviceComplexMatricesPtr makeDeviceComplexMatricesPtr(const Container<math::ComplexMatrix*>& matrices) {
    return smartptr_utils::makeSmartPtr<DeviceComplexMatricesPtr>(matrices);
  }

  inline DeviceComplexMatricesPtr makeDeviceComplexMatricesPtr(math::ComplexMatrix** array, size_t count) {
    return smartptr_utils::makeSmartPtr<DeviceComplexMatricesPtr>(array, count);
  }
}

#endif
