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

#ifndef OAP_HOST_MATRIX_PTR_H
#define OAP_HOST_MATRIX_PTR_H

#include "oapHostSmartPtr.hpp"

namespace oap
{

/**
 * @brief unique pointer for host matrix type
 *
 * examples of an use: oaphosttests/oapHostMatrixPtrtests.cpp
 */
class HostMatrixPtr : public oap::MatrixSharedPtr
{
  public:
    HostMatrixPtr(HostMatrixPtr&& orig) = default;
    HostMatrixPtr(const HostMatrixPtr& orig) = default;
    HostMatrixPtr& operator=(HostMatrixPtr&& orig) = default;
    HostMatrixPtr& operator=(const HostMatrixPtr& orig) = default;

    HostMatrixPtr (math::Matrix* matrix = nullptr, bool deallocate = true) :
      oap::MatrixSharedPtr (matrix, [this](const oap::math::Matrix* matrix) { host::DeleteMatrixWrapper (matrix, this); }, deallocate)
    {}

    virtual ~HostMatrixPtr () = default;

    operator math::Matrix*() const {
      return this->get();
    }
};

/**
 * @brief unique pointer for consthost matrix type
 */
class ConstHostMatrixPtr : public oap::ConstMatrixSharedPtr
{
  public:
    ConstHostMatrixPtr(ConstHostMatrixPtr&& orig) = default;
    ConstHostMatrixPtr(const ConstHostMatrixPtr& orig) = default;
    ConstHostMatrixPtr& operator=(ConstHostMatrixPtr&& orig) = default;
    ConstHostMatrixPtr& operator=(const ConstHostMatrixPtr& orig) = default;

    ConstHostMatrixPtr (const math::Matrix* matrix = nullptr, bool deallocate = true) :
      oap::ConstMatrixSharedPtr (matrix, [this](const oap::math::Matrix* matrix) { host::DeleteMatrixWrapper (matrix, this); }, deallocate)
    {}

    ConstHostMatrixPtr (math::Matrix* matrix = nullptr, bool deallocate = true) :
      oap::ConstMatrixSharedPtr (matrix, [this](const oap::math::Matrix* matrix) { host::DeleteMatrixWrapper (matrix, this); }, deallocate)
    {}

    virtual ~ConstHostMatrixPtr () = default;

    operator const math::Matrix*() const {
      return this->get();
    }
};

/**
 * @brief unique pointer which points into array of host matrix pointers
 *
 * this class creates its own matrices array which contains copied pointers.
 * if array was allocated dynamically must be deallocated.
 *
 * examples of use: oaphosttests/oapHostMatrixPtrtests.cpp
 */
class HostMatricesPtr : public oap::MatricesSharedPtr {
  public:
    HostMatricesPtr(HostMatricesPtr&& orig) = default;
    HostMatricesPtr(const HostMatricesPtr& orig) = default;
    HostMatricesPtr& operator=(HostMatricesPtr&& orig) = default;
    HostMatricesPtr& operator=(const HostMatricesPtr& orig) = default;

    HostMatricesPtr(oap::math::Matrix** matrices, size_t count, bool deallocate = true) :
      oap::MatricesSharedPtr (matrices, count, [this](oap::math::Matrix** matrices, size_t count) {host::DeleteMatricesWrapper<oap::math::Matrix>(matrices, count, this);}, deallocate)
    {}

    HostMatricesPtr(std::initializer_list<oap::math::Matrix*> matrices, bool deallocate = true) :
      oap::MatricesSharedPtr (matrices, [this](oap::math::Matrix** matrices, size_t count) {host::DeleteMatricesWrapper<oap::math::Matrix>(matrices, count, this);}, deallocate)
    {}

    HostMatricesPtr(size_t count, bool deallocate = true) :
      oap::MatricesSharedPtr (count, [this](oap::math::Matrix** matrices, size_t count) {host::DeleteMatricesWrapper<oap::math::Matrix>(matrices, count, this);}, deallocate)
    {}

    template<typename Container>
    static HostMatricesPtr make (const Container& container, bool deallocate = true) {
      return MatricesSPtrWrapper::make<HostMatricesPtr> (container, deallocate);
    }

    virtual ~HostMatricesPtr () = default;

    operator math::Matrix**() const {
      return this->get();
    }
};

}
#endif
