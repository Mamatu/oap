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

#ifndef OAP_DEVICE_MATRIX_PTR_H
#define OAP_DEVICE_MATRIX_PTR_H

#include "oapDeviceSmartPtr.h"

namespace oap
{
/**
 * @brief unique pointer for host matrix type
 *
 * examples of use: oaphosttests/oapDeviceMatrixPtrtests.cpp
 */
class DeviceMatrixPtr : public oap::MatrixSharedPtr
{
  public:
    DeviceMatrixPtr(DeviceMatrixPtr&& orig) = default;
    DeviceMatrixPtr(const DeviceMatrixPtr& orig) = default;
    DeviceMatrixPtr& operator=(DeviceMatrixPtr&& orig) = default;
    DeviceMatrixPtr& operator=(const DeviceMatrixPtr& orig) = default;

    DeviceMatrixPtr (math::Matrix* matrix = nullptr, bool deallocate = true) :
      oap::MatrixSharedPtr (matrix, [this](const oap::math::Matrix* matrix) { device::DeleteMatrixWrapper (matrix, this); }, deallocate)
    {}

    virtual ~DeviceMatrixPtr () = default;

    operator math::Matrix*() const {
      return this->get();
    }
};

/**
 * @brief unique pointer which points into array of host matrix pointers
 *
 * this class creates its own matrices array which contains copied pointers.
 * if array was allocated dynamically must be deallocated.
 *
 * examples of use: oaphosttests/oapDeviceMatrixPtrtests.cpp
 */
class DeviceMatricesPtr : public oap::MatricesSharedPtr {
  public:
    DeviceMatricesPtr(DeviceMatricesPtr&& orig) = default;
    DeviceMatricesPtr(const DeviceMatricesPtr& orig) = default;
    DeviceMatricesPtr& operator=(DeviceMatricesPtr&& orig) = default;
    DeviceMatricesPtr& operator=(const DeviceMatricesPtr& orig) = default;

    DeviceMatricesPtr(oap::math::Matrix** matrices, size_t count, bool deallocate = true) :
      oap::MatricesSharedPtr (matrices, count, [this](oap::math::Matrix** matrices, size_t count) {device::DeleteMatricesWrapper<oap::math::Matrix> (matrices, count, this);}, deallocate)
    {}

    DeviceMatricesPtr(std::initializer_list<oap::math::Matrix*> matrices, bool deallocate = true) :
      oap::MatricesSharedPtr (matrices, [this](oap::math::Matrix** matrices, size_t count) {device::DeleteMatricesWrapper<oap::math::Matrix> (matrices, count, this);}, deallocate)
    {}

    DeviceMatricesPtr(size_t count, bool deallocate = true) :
      oap::MatricesSharedPtr (count, [this](oap::math::Matrix** matrices, size_t count) {device::DeleteMatricesWrapper<oap::math::Matrix> (matrices, count, this);}, deallocate)
    {}

    virtual ~DeviceMatricesPtr () = default;

    operator math::Matrix**() const {
      return this->get();
    }
};

template<template<typename, typename> class Container>
DeviceMatricesPtr makeDeviceMatricesPtr(const Container<math::Matrix*, std::allocator<math::Matrix*> >& matrices) {
  return smartptr_utils::makeSmartPtr<DeviceMatricesPtr>(matrices);
}

template<template<typename> class Container>
DeviceMatricesPtr makeDeviceMatricesPtr(const Container<math::Matrix*>& matrices) {
  return smartptr_utils::makeSmartPtr<DeviceMatricesPtr>(matrices);
}

//inline DeviceMatricesPtr makeDeviceMatricesPtr(oap::math::Matrix** array, size_t count) {
//  return smartptr_utils::makeSmartPtr<DeviceMatricesPtr, oap::math::Matrix*>(array, count);
//}
}

#endif
