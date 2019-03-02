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

#ifndef OAP_HOSTMATRIXPTR_H
#define OAP_HOSTMATRIXPTR_H

#include "oapHostMatrixUtils.h"
#include "Math.h"

#include "oapMatrixSPtr.h"

namespace oap {

  /**
   * @brief Smart pointer for host matrix type
   *
   * Examples of use: oapHostTests/oapMatrixPtrTests.cpp
   */
  class HostMatrixPtr : public oap::MatrixSharedPtr {
    public:
      HostMatrixPtr(math::Matrix* matrix = nullptr) : oap::MatrixSharedPtr(matrix,
        [this](const math::Matrix* matrix) { debugInfo("Destroy: HostMatrixPtr = %p matrix = %p", this, matrix); oap::host::DeleteMatrix(matrix); })
      {
        debugInfo("Create: HostMatrixPtr = %p matrix = %p", this, matrix);
      }
  };

  /**
   * @brief Smart pointer which points into array of host matrix pointers
   *
   * This class creates its own matrices array which contains copied pointers.
   * If array was allocated dynamically must be deallocated.
   *
   * Examples of use: oapHostTests/oapMatrixPtrTests.cpp
   */
  class HostMatricesPtr : public oap::MatricesSharedPtr {
    public:
      HostMatricesPtr(math::Matrix** matrices, unsigned int count) :
        oap::MatricesSharedPtr (matrices, count, oap::host::DeleteMatrix) {}

      HostMatricesPtr(std::initializer_list<math::Matrix*> matrices) :
        oap::MatricesSharedPtr (matrices, oap::host::DeleteMatrix) {}

      HostMatricesPtr(size_t count) :
        oap::MatricesSharedPtr (count, oap::host::DeleteMatrix) {}
  };

  template<template<typename, typename> class Container>
  HostMatricesPtr makeHostMatricesPtr(const Container<math::Matrix*, std::allocator<math::Matrix*> >& matrices) {
    return smartptr_utils::makeSmartPtr<HostMatricesPtr>(matrices);
  }

  template<template<typename> class Container>
  HostMatricesPtr makeHostMatricesPtr(const Container<math::Matrix*>& matrices) {
    return smartptr_utils::makeSmartPtr<HostMatricesPtr>(matrices);
  }

  inline HostMatricesPtr makeHostMatricesPtr(math::Matrix** array, size_t count) {
    return smartptr_utils::makeSmartPtr<HostMatricesPtr>(array, count);
  }
}

#endif
