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

#ifndef OAPDEVICEMATRIXUPTR_H
#define OAPDEVICEMATRIXUPTR_H

#include "oapMatrixUPtr.h"
#include "Math.h"
#include "DeviceMatrixModules.h"


namespace oap {
  class DeviceMatrixUPtr : public oap::MatrixUPtr {
    public:
      DeviceMatrixUPtr(math::Matrix* matrix) : oap::MatrixUPtr(matrix, device::DeleteDeviceMatrix) {}
  };

  class DeviceMatricesUPtr : public oap::MatricesUPtr {
    public:
      DeviceMatricesUPtr(math::Matrix** matrices, unsigned int count) : oap::MatricesUPtr(matrices, deleters::MatricesDeleter(count, device::DeleteDeviceMatrix)) {}

      DeviceMatricesUPtr(std::initializer_list<math::Matrix*> matrices) :
        oap::MatricesUPtr(matrices, deleters::MatricesDeleter(smartptr_utils::getElementsCount(matrices), device::DeleteDeviceMatrix)) {}
  };

  template<template<typename, typename> class Container>
  DeviceMatricesUPtr makeDeviceMatricesUPtr(const Container<math::Matrix*, std::allocator<math::Matrix*> >& matrices) {
    return smartptr_utils::makeSmartPtr<DeviceMatricesUPtr>(matrices);
  }
}

#endif
