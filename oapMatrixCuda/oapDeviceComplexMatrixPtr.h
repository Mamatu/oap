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

#ifndef OAP_DEVICE_COMPLEX_MATRIX_PTR_H
#define OAP_DEVICE_COMPLEX_MATRIX_PTR_H

#include "oapDeviceSmartPtr.h"

namespace oap
{
/**
 * @brief unique pointer for host matrix type
 *
 * examples of use: oaphosttests/oapDeviceComplexMatrixPtrtests.cpp
 */
class DeviceComplexMatrixPtr : public oap::ComplexMatrixSharedPtr
{
  public:
    DeviceComplexMatrixPtr(DeviceComplexMatrixPtr&& orig) = default;
    DeviceComplexMatrixPtr(const DeviceComplexMatrixPtr& orig) = default;
    DeviceComplexMatrixPtr& operator=(DeviceComplexMatrixPtr&& orig) = default;
    DeviceComplexMatrixPtr& operator=(const DeviceComplexMatrixPtr& orig) = default;

    DeviceComplexMatrixPtr (math::ComplexMatrix* matrix = nullptr, bool deallocate = true) :
      oap::ComplexMatrixSharedPtr (matrix, [this](const oap::math::ComplexMatrix* matrix) { device::DeleteMatrixWrapper (matrix, this); }, deallocate)
    {}

    virtual ~DeviceComplexMatrixPtr () = default;

    operator math::ComplexMatrix*() const {
      return this->get();
    }
};

/**
 * @brief unique pointer which points into array of host matrix pointers
 *
 * this class creates its own matrices array which contains copied pointers.
 * if array was allocated dynamically must be deallocated.
 *
 * examples of use: oaphosttests/oapDeviceComplexMatrixPtrtests.cpp
 */
class DeviceComplexMatricesPtr : public oap::ComplexMatricesSharedPtr {
  public:
    DeviceComplexMatricesPtr(DeviceComplexMatricesPtr&& orig) = default;
    DeviceComplexMatricesPtr(const DeviceComplexMatricesPtr& orig) = default;
    DeviceComplexMatricesPtr& operator=(DeviceComplexMatricesPtr&& orig) = default;
    DeviceComplexMatricesPtr& operator=(const DeviceComplexMatricesPtr& orig) = default;

    DeviceComplexMatricesPtr(oap::math::ComplexMatrix** matrices, size_t count, bool deallocate = true) :
      oap::ComplexMatricesSharedPtr (matrices, count, [this](oap::math::ComplexMatrix** matrices, size_t count) {device::DeleteMatricesWrapper<oap::math::ComplexMatrix> (matrices, count, this);}, deallocate)
    {}

    DeviceComplexMatricesPtr(std::initializer_list<oap::math::ComplexMatrix*> matrices, bool deallocate = true) :
      oap::ComplexMatricesSharedPtr (matrices, [this](oap::math::ComplexMatrix** matrices, size_t count) {device::DeleteMatricesWrapper<oap::math::ComplexMatrix> (matrices, count, this);}, deallocate)
    {}

    DeviceComplexMatricesPtr(size_t count, bool deallocate = true) :
      oap::ComplexMatricesSharedPtr (count, [this](oap::math::ComplexMatrix** matrices, size_t count) {device::DeleteMatricesWrapper<oap::math::ComplexMatrix> (matrices, count, this);}, deallocate)
    {}

    template<typename Container>
    static DeviceComplexMatricesPtr make (const Container& container, bool deallocate = true) {
      return MatricesSPtrWrapper::make<DeviceComplexMatricesPtr> (container, deallocate);
    }

    virtual ~DeviceComplexMatricesPtr () = default;

    operator math::ComplexMatrix**() const {
      return this->get();
    }
};

}

#endif
