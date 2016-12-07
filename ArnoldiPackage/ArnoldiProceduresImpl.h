/*
 * Copyright 2016 Marcin Matula
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

#ifndef ARNOLDIPROCEDURESIMPL_H
#define ARNOLDIPROCEDURESIMPL_H

#include "ArnoldiProcedures.h"

class CuHArnoldiDefault : public CuHArnoldi {
 public:
  /**
 * @brief Set device matrix to calculate its eigenvalues and eigenvectors.
 * @param A
 */
  void setMatrix(math::Matrix* A) { m_A = A; }

 protected:
  void multiply(math::Matrix* m_w, math::Matrix* m_v,
                CuHArnoldi::MultiplicationType mt);

  virtual bool checkEigenvalue(floatt value, uint index) { return true; }

  virtual bool checkEigenvector(math::Matrix* vector, uint index) {
    return true;
  }

 private:
  math::Matrix* m_A;
};

class CuHArnoldiCallback : public CuHArnoldi {
 public:
  typedef void (*MultiplyFunc)(math::Matrix* m_w, math::Matrix* m_v,
                               void* userData,
                               CuHArnoldi::MultiplicationType mt);

  void setCallback(MultiplyFunc multiplyFunc, void* userData);

 protected:
  void multiply(math::Matrix* m_w, math::Matrix* m_v,
                CuHArnoldi::MultiplicationType mt);

  virtual bool checkEigenvalue(floatt value, uint index) { return true; }

  virtual bool checkEigenvector(math::Matrix* vector, uint index) {
    return true;
  }

 private:
  MultiplyFunc m_multiplyFunc;
  void* m_userData;
};
#endif  // ARNOLDIPROCEDURESIMPL_H