/*
 * Copyright 2016, 2017 Marcin Matula
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
#include "DeviceMatrixModules.h"

#include "oapMatrixSPtr.h"

namespace oap {
  class DeviceMatrixPtr : public oap::MatrixSharedPtr {
    public:
      DeviceMatrixPtr(math::Matrix* matrix) : oap::MatrixSharedPtr(matrix, device::DeleteDeviceMatrix) {}
  };

  class DeviceMatricesPtr : public oap::MatricesSharedPtr {
    public:
      DeviceMatricesPtr(math::Matrix** matrices, unsigned int count) :
        oap::MatricesSharedPtr(matrices, deleters::MatricesDeleter(count, device::DeleteDeviceMatrix)) {}

      DeviceMatricesPtr(std::initializer_list<math::Matrix*> matrices) :
        oap::MatricesSharedPtr(matrices, deleters::MatricesDeleter(smartptr_utils::getElementsCount(matrices), device::DeleteDeviceMatrix)) {}
  };

  template<template<typename, typename> class Container>
  DeviceMatricesPtr makeDeviceMatricesPtr(const Container<math::Matrix*, std::allocator<math::Matrix*> >& matrices) {
    return smartptr_utils::makeSmartPtr<DeviceMatricesPtr>(matrices);
  }
}

#endif
