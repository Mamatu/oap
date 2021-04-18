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

#ifndef OAP_DEVICE_MATRIX_UPTR_H
#define OAP_DEVICE_MATRIX_UPTR_H

#include "oapDeviceSmartPtr.h"

namespace oap
{
/**
 * @brief unique pointer for host matrix type
 *
 * examples of use: oaphosttests/oapDeviceMatrixUPtrtests.cpp
 */
class DeviceMatrixUPtr : public oap::MatrixUniquePtr
{
  public:
    DeviceMatrixUPtr(DeviceMatrixUPtr&& orig) = default;
    DeviceMatrixUPtr(const DeviceMatrixUPtr& orig) = default;
    DeviceMatrixUPtr& operator=(DeviceMatrixUPtr&& orig) = default;
    DeviceMatrixUPtr& operator=(const DeviceMatrixUPtr& orig) = default;

    DeviceMatrixUPtr (math::Matrix* matrix = nullptr, bool deallocate = true) :
      oap::MatrixUniquePtr (matrix, [this](const oap::math::Matrix* matrix) { device::DeleteMatrixWrapper (this, matrix); }, deallocate)
    {}

    virtual ~DeviceMatrixUPtr () = default;

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
 * examples of use: oaphosttests/oapDeviceMatrixUPtrtests.cpp
 */
class DeviceMatricesUPtr : public oap::MatricesUniquePtr {
  public:
    DeviceMatricesUPtr(DeviceMatricesUPtr&& orig) = default;
    DeviceMatricesUPtr(const DeviceMatricesUPtr& orig) = default;
    DeviceMatricesUPtr& operator=(DeviceMatricesUPtr&& orig) = default;
    DeviceMatricesUPtr& operator=(const DeviceMatricesUPtr& orig) = default;

    DeviceMatricesUPtr(oap::math::Matrix** matrices, size_t count, bool deallocate = true) :
      oap::MatricesUniquePtr (matrices, count, [this](oap::math::Matrix** matrices, size_t count) {device::DeleteMatricesWrapper<oap::math::Matrix> (matrices, count, this);}, deallocate)
    {}

    DeviceMatricesUPtr(std::initializer_list<oap::math::Matrix*> matrices, bool deallocate = true) :
      oap::MatricesUniquePtr (matrices,  [this](oap::math::Matrix** matrices, size_t count) {device::DeleteMatricesWrapper<oap::math::Matrix> (matrices, count, this);}, deallocate)
    {}

    DeviceMatricesUPtr(size_t count, bool deallocate = true) :
      oap::MatricesUniquePtr (count, [this](oap::math::Matrix** matrices, size_t count) {device::DeleteMatricesWrapper<oap::math::Matrix> (matrices, count, this);}, deallocate)
    {}

    template<typename Container>
    static DeviceMatricesUPtr make (const Container& container, bool deallocate = true) {
      return MatricesSPtrWrapper::make<DeviceMatricesUPtr> (container, deallocate);
    }

    virtual ~DeviceMatricesUPtr () = default;

    operator math::Matrix**() const {
      return this->get();
    }
};

}

#endif
