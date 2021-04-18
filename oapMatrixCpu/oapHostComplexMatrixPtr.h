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

#ifndef OAP_HOST_COMPLEX_MATRIX_PTR_H
#define OAP_HOST_COMPLEX_MATRIX_PTR_H

#include "oapHostSmartPtr.h"

namespace oap
{
/**
 * @brief unique pointer for host matrix type
 *
 * examples of use: oaphosttests/oapHostComplexMatrixPtrtests.cpp
 */
class HostComplexMatrixPtr : public ComplexMatrixSharedPtr
{
  public:
    HostComplexMatrixPtr(HostComplexMatrixPtr&& orig) = default;
    HostComplexMatrixPtr(const HostComplexMatrixPtr& orig) = default;
    HostComplexMatrixPtr& operator=(HostComplexMatrixPtr&& orig) = default;
    HostComplexMatrixPtr& operator=(const HostComplexMatrixPtr& orig) = default;

    HostComplexMatrixPtr (math::ComplexMatrix* matrix = nullptr, bool deallocate = true) :
      ComplexMatrixSharedPtr (matrix, [this](const math::ComplexMatrix* matrix) { host::DeleteMatrixWrapper (matrix, this); }, deallocate)
    {}

    virtual ~HostComplexMatrixPtr () = default;

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
 * examples of use: oaphosttests/oapHostComplexMatrixPtrtests.cpp
 */
class HostComplexMatricesPtr : public ComplexMatricesSharedPtr {
  public:
    HostComplexMatricesPtr(HostComplexMatricesPtr&& orig) = default;
    HostComplexMatricesPtr(const HostComplexMatricesPtr& orig) = default;
    HostComplexMatricesPtr& operator=(HostComplexMatricesPtr&& orig) = default;
    HostComplexMatricesPtr& operator=(const HostComplexMatricesPtr& orig) = default;

    HostComplexMatricesPtr(math::ComplexMatrix** matrices, size_t count, bool deallocate = true) :
      ComplexMatricesSharedPtr (matrices, count, [this] (oap::math::ComplexMatrix** matrices, size_t count) { host::DeleteMatricesWrapper<math::ComplexMatrix>(matrices, count, this); }, deallocate)
    {}

    HostComplexMatricesPtr(std::initializer_list<math::ComplexMatrix*> matrices, bool deallocate = true) :
      ComplexMatricesSharedPtr (matrices, [this] (oap::math::ComplexMatrix** matrices, size_t count) { host::DeleteMatricesWrapper<math::ComplexMatrix>(matrices, count, this); }, deallocate)
    {}

    HostComplexMatricesPtr(size_t count, bool deallocate = true) :
      ComplexMatricesSharedPtr (count, [this] (oap::math::ComplexMatrix** matrices, size_t count, bool deallocate = true) { host::DeleteMatricesWrapper<math::ComplexMatrix>(matrices, count, this); }, deallocate)
    {}

    template<typename Container>
    static HostComplexMatricesPtr make (const Container& container, bool deallocate = true) {
      return MatricesSPtrWrapper::make<HostComplexMatricesPtr> (container, deallocate);
    }

    virtual ~HostComplexMatricesPtr () = default;

    operator math::ComplexMatrix**() const {
      return this->get();
    }
};

template<template<typename, typename> class Container>
HostComplexMatricesPtr makeHostComplexMatricesPtr(const Container<math::ComplexMatrix*, std::allocator<math::ComplexMatrix*> >& matrices) {
  return smartptr_utils::makeSmartPtr<HostComplexMatricesPtr>(matrices);
}

template<template<typename> class Container>
HostComplexMatricesPtr makeHostComplexMatricesPtr(const Container<math::ComplexMatrix*>& matrices) {
  return smartptr_utils::makeSmartPtr<HostComplexMatricesPtr>(matrices);
}

}

#endif
