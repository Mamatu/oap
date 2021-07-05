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

#ifndef OAP_HOST_MATRIX_UPTR_H
#define OAP_HOST_MATRIX_UPTR_H

#include "oapHostSmartPtr.hpp"

namespace oap
{
/**
 * @brief unique pointer for host matrix type
 *
 * examples of use: oaphosttests/oapHostMatrixUPtrtests.cpp
 */
class HostMatrixUPtr : public oap::MatrixUniquePtr
{
  public:
    HostMatrixUPtr(HostMatrixUPtr&& orig) = default;
    HostMatrixUPtr(const HostMatrixUPtr& orig) = default;
    HostMatrixUPtr& operator=(HostMatrixUPtr&& orig) = default;
    HostMatrixUPtr& operator=(const HostMatrixUPtr& orig) = default;

    HostMatrixUPtr (math::Matrix* matrix = nullptr, bool deallocate = true) :
      oap::MatrixUniquePtr (matrix, [this](const oap::math::Matrix* matrix) { host::DeleteMatrixWrapper (matrix, this); }, deallocate)
    {}

    virtual ~HostMatrixUPtr() = default;

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
 * examples of use: oaphosttests/oapHostMatrixUPtrtests.cpp
 */
class HostMatricesUPtr : public oap::MatricesUniquePtr {
  public:
    HostMatricesUPtr(HostMatricesUPtr&& orig) = default;
    HostMatricesUPtr(const HostMatricesUPtr& orig) = default;
    HostMatricesUPtr& operator=(HostMatricesUPtr&& orig) = default;
    HostMatricesUPtr& operator=(const HostMatricesUPtr& orig) = default;

    HostMatricesUPtr(oap::math::Matrix** matrices, size_t count, bool deallocate = true) :
      oap::MatricesUniquePtr (matrices, count, [this](oap::math::Matrix** matrices, size_t count) {host::DeleteMatricesWrapper<oap::math::Matrix>(matrices, count, this);}, deallocate)
    {}

    HostMatricesUPtr(std::initializer_list<oap::math::Matrix*> matrices, bool deallocate = true) :
      oap::MatricesUniquePtr (matrices, [this](oap::math::Matrix** matrices, size_t count) {host::DeleteMatricesWrapper<oap::math::Matrix>(matrices, count, this);}, deallocate)
    {}

    HostMatricesUPtr(size_t count, bool deallocate = true) :
      oap::MatricesUniquePtr (count, [this](oap::math::Matrix** matrices, size_t count) {host::DeleteMatricesWrapper<oap::math::Matrix>(matrices, count, this);}, deallocate)
    {}

    template<typename Container>
    static HostMatricesUPtr make (const Container& container, bool deallocate = true) {
      return MatricesSPtrWrapper::make<HostMatricesUPtr> (container, deallocate);
    }

    virtual ~HostMatricesUPtr () = default;

    operator math::Matrix**() const {
      return this->get();
    }
};

}

#endif
