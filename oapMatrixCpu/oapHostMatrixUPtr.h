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

#ifndef OAP_HOSTMATRIXUPTR_H
#define OAP_HOSTMATRIXUPTR_H

#include "oapHostMatrixUtils.h"
#include "Math.h"

#include "oapMatrixSPtr.h"

namespace oap {

  /**
   * @brief Unique pointer for host matrix type
   *
   * Examples of use: oapHostTests/oapHostMatrixUPtrTests.cpp
   */
  class HostMatrixUPtr : public oap::MatrixUniquePtr {
    public:
      HostMatrixUPtr (math::ComplexMatrix* matrix, bool bDeallocate = true) :
        oap::MatrixUniquePtr (matrix, oap::host::DeleteMatrix, bDeallocate)
      {
      }
  };

  /**
   * @brief Unique pointer which points into array of host matrix pointers
   *
   * This class creates its own matrices array which contains copied pointers.
   * If array was allocated dynamically must be deallocated.
   *
   * Examples of use: oapHostTests/oapHostMatrixUPtrTests.cpp
   */
  class HostMatricesUPtr : public oap::MatricesUniquePtr {
    public:
      HostMatricesUPtr(math::ComplexMatrix** matrices, unsigned int count) :
        oap::MatricesUniquePtr(matrices, count, oap::host::DeleteMatrix) {}

      HostMatricesUPtr(size_t count) :
        oap::MatricesUniquePtr(count, oap::host::DeleteMatrix) {}

      HostMatricesUPtr(std::initializer_list<math::ComplexMatrix*> matrices) :
        oap::MatricesUniquePtr(matrices, oap::host::DeleteMatrix) {}

    private:
      HostMatricesUPtr(math::ComplexMatrix** matrices, unsigned int count, bool bCopyArray) :
        oap::MatricesUniquePtr (matrices, count, oap::host::DeleteMatrix, bCopyArray) {}

      template<class SmartPtr, template<typename, typename> class Container, typename T>
      friend SmartPtr smartptr_utils::makeSmartPtr(const Container<T, std::allocator<T> >& container);
  };

  template<template<typename, typename> class Container>
  HostMatricesUPtr makeHostMatricesUPtr(const Container<math::ComplexMatrix*, std::allocator<math::ComplexMatrix*> >& matrices) {
    return smartptr_utils::makeSmartPtr<HostMatricesUPtr>(matrices);
  }

  template<template<typename> class Container>
  HostMatricesUPtr makeHostMatricesUPtr(const Container<math::ComplexMatrix*>& matrices) {
    return smartptr_utils::makeSmartPtr<HostMatricesUPtr>(matrices);
  }

  inline HostMatricesUPtr makeHostMatricesUPtr(math::ComplexMatrix** array, size_t count) {
    return smartptr_utils::makeSmartPtr<HostMatricesUPtr>(array, count);
  }
}

#endif
