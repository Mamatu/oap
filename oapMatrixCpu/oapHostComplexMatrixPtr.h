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

#ifndef OAP_HOSTMATRIXPTR_H
#define OAP_HOSTMATRIXPTR_H

#include "oapHostMatrixUtils.h"
#include "Math.h"

#include "oapMatrixSPtr.h"

namespace oap {

  /**
   * @brief Shared pointer for host matrix type
   *
   * Examples of use: oapHostTests/oapHostComplexMatrixPtrTests.cpp
   */
  class HostComplexMatrixPtr : public oap::ComplexMatrixSharedPtr {
    public:
      HostComplexMatrixPtr(math::ComplexMatrix* matrix = nullptr) : oap::ComplexMatrixSharedPtr(matrix,
        [this](const math::ComplexMatrix* matrix) { logTrace("Destroy: HostComplexMatrixPtr = %p matrix = %p", this, matrix); oap::host::DeleteMatrix(matrix); })
      {
        logTrace("Create: HostComplexMatrixPtr = %p matrix = %p", this, matrix);
      }
  };

  /**
   * @brief Shared pointer which points into array of host matrix pointers
   *
   * This class creates its own matrices array which contains copied pointers.
   * If array was allocated dynamically must be deallocated.
   *
   * Examples of use: oapHostTests/oapHostComplexMatrixPtrTests.cpp
   */
  class HostComplexMatricesPtr : public oap::ComplexMatricesSharedPtr {
    public:
      HostComplexMatricesPtr(math::ComplexMatrix** matrices, unsigned int count) :
        oap::ComplexMatricesSharedPtr (matrices, count, oap::host::DeleteMatrix) {}

      HostComplexMatricesPtr(std::initializer_list<math::ComplexMatrix*> matrices) :
        oap::ComplexMatricesSharedPtr (matrices, oap::host::DeleteMatrix) {}

      HostComplexMatricesPtr(size_t count) :
        oap::ComplexMatricesSharedPtr (count, oap::host::DeleteMatrix) {}

    private:
      HostComplexMatricesPtr(math::ComplexMatrix** matrices, unsigned int count, bool bCopyArray) :
        oap::ComplexMatricesSharedPtr (matrices, count, oap::host::DeleteMatrix, bCopyArray) {}

      template<class SmartPtr, template<typename, typename> class Container, typename T>
      friend SmartPtr smartptr_utils::makeSmartPtr(const Container<T, std::allocator<T> >& container);
  };

  template<template<typename, typename> class Container>
  HostComplexMatricesPtr makeHostComplexMatricesPtr(const Container<math::ComplexMatrix*, std::allocator<math::ComplexMatrix*> >& matrices) {
    return smartptr_utils::makeSmartPtr<HostComplexMatricesPtr>(matrices);
  }

  template<template<typename> class Container>
  HostComplexMatricesPtr makeHostComplexMatricesPtr(const Container<math::ComplexMatrix*>& matrices) {
    return smartptr_utils::makeSmartPtr<HostComplexMatricesPtr>(matrices);
  }

  inline HostComplexMatricesPtr makeHostComplexMatricesPtr(math::ComplexMatrix** array, size_t count) {
    return smartptr_utils::makeSmartPtr<HostComplexMatricesPtr>(array, count);
  }
}

#endif
