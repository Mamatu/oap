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

#ifndef OAP_HOST_COMPLEX_MATRIX_UPTR_H
#define OAP_HOST_COMPLEX_MATRIX_UPTR_H

#include "oapHostSmartPtr.hpp"

namespace oap {
  /**
   * @brief unique pointer for host matrix type
   *
   * examples of use: oaphosttests/oapHostComplexMatrixUPtrtests.cpp
   */
  class HostComplexMatrixUPtr : public oap::ComplexMatrixUniquePtr {
    public:
      HostComplexMatrixUPtr(HostComplexMatrixUPtr&& orig) = default;
      HostComplexMatrixUPtr(const HostComplexMatrixUPtr& orig) = default;
      HostComplexMatrixUPtr& operator=(HostComplexMatrixUPtr&& orig) = default;
      HostComplexMatrixUPtr& operator=(const HostComplexMatrixUPtr& orig) = default;

      HostComplexMatrixUPtr (math::ComplexMatrix* matrix = nullptr, bool deallocate = true) :
        oap::ComplexMatrixUniquePtr (matrix, [this](const oap::math::ComplexMatrix* matrix) { host::DeleteMatrixWrapper (matrix, this); }, deallocate)
      {}

      virtual ~HostComplexMatrixUPtr () = default;

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
   * examples of use: oaphosttests/oapHostComplexMatrixUPtrtests.cpp
   */
  class HostComplexMatricesUPtr : public oap::ComplexMatricesUniquePtr {
    public:
      HostComplexMatricesUPtr(HostComplexMatricesUPtr&& orig) = default;
      HostComplexMatricesUPtr(const HostComplexMatricesUPtr& orig) = default;
      HostComplexMatricesUPtr& operator=(HostComplexMatricesUPtr&& orig) = default;
      HostComplexMatricesUPtr& operator=(const HostComplexMatricesUPtr& orig) = default;

      template<typename Container>
      static HostComplexMatricesUPtr make (const Container& container) {
        return oap::ComplexMatricesUniquePtr::make<HostComplexMatricesUPtr> (container);
      }

      HostComplexMatricesUPtr(oap::math::ComplexMatrix** matrices, size_t count, bool deallocate = true) :
        oap::ComplexMatricesUniquePtr (matrices, count, [this] (oap::math::ComplexMatrix** matrices, size_t count) { host::DeleteMatricesWrapper<math::ComplexMatrix>(matrices, count, this); }, deallocate)
      {}

      HostComplexMatricesUPtr(std::initializer_list<oap::math::ComplexMatrix*> matrices, bool deallocate = true) :
        oap::ComplexMatricesUniquePtr (matrices, [this] (oap::math::ComplexMatrix** matrices, size_t count) { host::DeleteMatricesWrapper<math::ComplexMatrix>(matrices, count, this); }, deallocate)
      {}

      HostComplexMatricesUPtr(size_t count, bool deallocate = true) :
        oap::ComplexMatricesUniquePtr (count, [this] (oap::math::ComplexMatrix** matrices, size_t count) { host::DeleteMatricesWrapper<math::ComplexMatrix>(matrices, count, this); }, deallocate)
      {}

      virtual ~HostComplexMatricesUPtr () = default;

      operator math::ComplexMatrix**() const {
        return this->get();
      }
  };
}

#endif
