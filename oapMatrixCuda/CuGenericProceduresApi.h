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

#ifndef OAP_CU_GENERIC_PROCEDURES_API_H
#define OAP_CU_GENERIC_PROCEDURES_API_H

#include "CuProceduresApi.h"

namespace oap
{

class CuGenericProceduresApi
{
 public:
  CuGenericProceduresApi(CuProceduresApi* cuApi);
  virtual ~CuGenericProceduresApi();

  CuGenericProceduresApi(const CuGenericProceduresApi&) = delete;
  CuGenericProceduresApi(CuGenericProceduresApi&&) = delete;
  CuGenericProceduresApi& operator=(const CuGenericProceduresApi&) = delete;
  CuGenericProceduresApi& operator=(CuGenericProceduresApi&&) = delete;

  template<typename Matrices>
  void add (Matrices& output, const Matrices& params1, floatt value)
  {
    m_cuApi->v2_add (output, params1, value);
  }

  template<typename Matrices>
  void add (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_add (output, params1, params2);
  }

  template<typename Matrices>
  void subtract (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_subtract (output, params1, params2);
  }

  template<typename Matrices>
  void dotProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_dotProduct (output, params1, params2);
  }

  template<typename Matrices>
  void multiply (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_multiply (output, params1, params2);
  }

  template<typename Matrices>
  void hadamardProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_hadamardProduct (output, params1, params2);
  }

  template<typename Matrices>
  void hadamardProductVec (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_hadamardProductVec (output, params1, params2);
  }

  template<typename Matrices>
  void tensorProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_tensorProduct (output, params1, params2);
  }

  template<typename Matrices>
  void transpose (Matrices& output, const Matrices& params1)
  {
    m_cuApi->v2_transpose (output, params1);
  }

  template<typename Matrices>
  void sigmoid (Matrices& output, Matrices& params)
  {
    m_cuApi->v2_sigmoid (output, params);
  }

  template<typename Matrices>
  void dsigmoid (Matrices& output, Matrices& params)
  {
    m_cuApi->v2_dsigmoid (output, params);
  }

  template<typename Matrices>
  void multiplyDSigmoid (Matrices& output, Matrices& params)
  {
    m_cuApi->v2_multiplyDSigmoid (output, params);
  }

  template<typename Matrices>
  void linear (Matrices& output, Matrices& params)
  {
    m_cuApi->v2_linear (output, params);
  }

  template<typename Matrices>
  void dlinear (Matrices& output, Matrices& params)
  {
    m_cuApi->v2_dlinear (output, params);
  }

  template<typename Matrices>
  void tanh (Matrices& output, Matrices& params)
  {
    m_cuApi->v2_tanh (output, params);
  }

  template<typename Matrices>
  void dtanh (Matrices& output, Matrices& params)
  {
    m_cuApi->v2_dtanh (output, params);
  }

  template<typename Matrices>
  void sin (Matrices& output, Matrices& params)
  {
    m_cuApi->v2_sin (output, params);
  }

  template<typename Matrices>
  void dsin (Matrices& output, Matrices& params)
  {
    m_cuApi->v2_dsin (output, params);
  }

  template<typename Matrices>
  void multiplyDSin (Matrices& output, Matrices& params)
  {
    m_cuApi->v2_multiplyDSin (output, params);
  }

  template<typename Matrices>
  void relu (Matrices& output, Matrices& params)
  {
    m_cuApi->v2_relu (output, params);
  }

  template<typename Matrices>
  void drelu (Matrices& output, Matrices& params)
  {
    m_cuApi->v2_drelu (output, params);
  }

  template<typename Matrices>
  void prelu (Matrices& output, Matrices& params)
  {
    m_cuApi->v2_prelu (output, params);
  }

  template<typename Matrices>
  void dprelu (Matrices& output, Matrices& params)
  {
    m_cuApi->v2_dprelu (output, params);
  }

  template<typename Matrices>
  void softplus (Matrices& output, Matrices& params)
  {
    m_cuApi->v2_softplus (output, params);
  }

  template<typename Matrices>
  void dsoftplus (Matrices& output, Matrices& params)
  {
    m_cuApi->v2_dsoftplus (output, params);
  }
  private:
    CuProceduresApi* m_cuApi;
};

}

#endif /* MATRIXPROCEDURES_H */
