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

#ifndef OAP_DEVICEMATRIXUPTR_H
#define OAP_DEVICEMATRIXUPTR_H

#include "Math.h"
#include "oapCudaMatrixUtils.h"

#include "oapMatrixSPtr.h"

namespace oap {

  /**
   * @brief Unique pointer for cuda matrix type
   *
   * Examples of use: oapDeviceTests/oapDeviceComplexMatrixUPtrTests.cpp
   */
  class DeviceComplexMatrixUPtr : public oap::MatrixUniquePtr {
    public:
      DeviceComplexMatrixUPtr(math::ComplexMatrix* matrix = nullptr, bool bDeallocate = true) :
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
   * Examples of use: oapDeviceTests/oapDeviceComplexMatrixUPtrTests.cpp
   */
  class DeviceComplexMatricesUPtr : public oap::MatricesUniquePtr {
    public:
      DeviceComplexMatricesUPtr(math::ComplexMatrix** matrices, unsigned int count) :
        oap::MatricesUniquePtr(matrices, count, oap::cuda::DeleteDeviceMatrix) {}

      DeviceComplexMatricesUPtr(std::initializer_list<math::ComplexMatrix*> matrices) :
        oap::MatricesUniquePtr(matrices, oap::cuda::DeleteDeviceMatrix) {}

    private:
      DeviceComplexMatricesUPtr(math::ComplexMatrix** matrices, unsigned int count, bool bCopyArray) :
        oap::MatricesUniquePtr (matrices, count, oap::cuda::DeleteDeviceMatrix, bCopyArray) {}

      template<class SmartPtr, template<typename, typename> class Container, typename T>
      friend SmartPtr smartptr_utils::makeSmartPtr(const Container<T, std::allocator<T> >& container);
  };

  template<template<typename, typename> class Container>
  DeviceComplexMatricesUPtr makeDeviceComplexMatricesUPtr(const Container<math::ComplexMatrix*, std::allocator<math::ComplexMatrix*> >& matrices) {
    return smartptr_utils::makeSmartPtr<DeviceComplexMatricesUPtr>(matrices);
  }

  template<template<typename> class Container>
  DeviceComplexMatricesUPtr makeDeviceComplexMatricesUPtr(const Container<math::ComplexMatrix*>& matrices) {
    return smartptr_utils::makeSmartPtr<DeviceComplexMatricesUPtr>(matrices);
  }

  inline DeviceComplexMatricesUPtr makeDeviceComplexMatricesUPtr(math::ComplexMatrix** array, size_t count) {
    return smartptr_utils::makeSmartPtr<DeviceComplexMatricesUPtr>(array, count);
  }
}

#endif
