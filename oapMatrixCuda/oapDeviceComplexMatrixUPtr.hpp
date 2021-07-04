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

#ifndef OAP_DEVICE_COMPLEX_MATRIX_UPTR_H
#define OAP_DEVICE_COMPLEX_MATRIX_UPTR_H

#include "oapDeviceSmartPtr.hpp"

namespace oap
{
/**
 * @brief unique pointer for host matrix type
 *
 * examples of use: oaphosttests/oapDeviceComplexMatrixUPtrtests.cpp
 */
class DeviceComplexMatrixUPtr : public oap::ComplexMatrixUniquePtr
{
  public:
    DeviceComplexMatrixUPtr(DeviceComplexMatrixUPtr&& orig) = default;
    DeviceComplexMatrixUPtr(const DeviceComplexMatrixUPtr& orig) = default;
    DeviceComplexMatrixUPtr& operator=(DeviceComplexMatrixUPtr&& orig) = default;
    DeviceComplexMatrixUPtr& operator=(const DeviceComplexMatrixUPtr& orig) = default;

    DeviceComplexMatrixUPtr (math::ComplexMatrix* matrix = nullptr, bool deallocate = true) :
      oap::ComplexMatrixUniquePtr (matrix, [this](const oap::math::ComplexMatrix* matrix) { device::DeleteMatrixWrapper<oap::math::ComplexMatrix> (matrix, this); }, deallocate)
    {}

    virtual ~DeviceComplexMatrixUPtr () = default;

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
 * examples of use: oaphosttests/oapDeviceComplexMatrixUPtrtests.cpp
 */
class DeviceComplexMatricesUPtr : public oap::ComplexMatricesUniquePtr {
  public:

    DeviceComplexMatricesUPtr(DeviceComplexMatricesUPtr&& orig) = default;
    DeviceComplexMatricesUPtr(const DeviceComplexMatricesUPtr& orig) = default;
    DeviceComplexMatricesUPtr& operator=(DeviceComplexMatricesUPtr&& orig) = default;
    DeviceComplexMatricesUPtr& operator=(const DeviceComplexMatricesUPtr& orig) = default;

    DeviceComplexMatricesUPtr(oap::math::ComplexMatrix** matrices, size_t count, bool deallocate = true) :
      oap::ComplexMatricesUniquePtr (matrices, count, [this](oap::math::ComplexMatrix** matrices, size_t count) {device::DeleteMatricesWrapper<oap::math::ComplexMatrix> (matrices, count, this);}, deallocate)
    {}

    DeviceComplexMatricesUPtr(std::initializer_list<oap::math::ComplexMatrix*> matrices, bool deallocate = true) :
      oap::ComplexMatricesUniquePtr (matrices, [this](oap::math::ComplexMatrix** matrices, size_t count) {device::DeleteMatricesWrapper<oap::math::ComplexMatrix> (matrices, count, this);}, deallocate)
    {}

    DeviceComplexMatricesUPtr(size_t count, bool deallocate = true) :
      oap::ComplexMatricesUniquePtr (count, [this](oap::math::ComplexMatrix** matrices, size_t count) {device::DeleteMatricesWrapper<oap::math::ComplexMatrix> (matrices, count, this);}, deallocate)
    {}

    template<typename Container>
    static DeviceComplexMatricesUPtr make (const Container& container, bool deallocate = true) {
      return MatricesSPtrWrapper::make<DeviceComplexMatricesUPtr> (container, deallocate);
    }

    virtual ~DeviceComplexMatricesUPtr () = default;

    operator math::ComplexMatrix**() const {
      return this->get();
    }
};

}

#endif
