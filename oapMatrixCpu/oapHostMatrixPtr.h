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
  class HostMatrixPtr : public oap::MatrixSharedPtr {
    public:
      HostMatrixPtr(math::Matrix* matrix = nullptr) : oap::MatrixSharedPtr(matrix, oap::host::DeleteMatrix) {}
  };

  class HostMatricesPtr : public oap::MatricesSharedPtr {
    public:
      HostMatricesPtr(math::Matrix** matrices, unsigned int count) :
        oap::MatricesSharedPtr(matrices, deleters::MatricesDeleter(count, oap::host::DeleteMatrix)) {}

      HostMatricesPtr(std::initializer_list<math::Matrix*> matrices) :
        oap::MatricesSharedPtr(matrices, deleters::MatricesDeleter(smartptr_utils::getElementsCount(matrices), oap::host::DeleteMatrix)) {}

      HostMatricesPtr(size_t count) :
        oap::MatricesSharedPtr(count, deleters::MatricesDeleter(count, oap::host::DeleteMatrix)) {}
  };

  template<template<typename, typename> class Container>
  HostMatricesPtr makeHostMatricesPtr(const Container<math::Matrix*, std::allocator<math::Matrix*> >& matrices) {
    return smartptr_utils::makeSmartPtr<HostMatricesPtr>(matrices);
  }
}

#endif
