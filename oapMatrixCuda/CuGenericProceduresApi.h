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
#include "oapProcedures.h"

namespace oap
{

class CuGenericProceduresApi : public oap::generic::MultiMatricesProcedures
{
 public:
  using Matrices = oap::generic::MultiMatricesProcedures::Matrices;
  CuGenericProceduresApi(CuProceduresApi* cuApi);
  virtual ~CuGenericProceduresApi();

  CuGenericProceduresApi(const CuGenericProceduresApi&) = delete;
  CuGenericProceduresApi(CuGenericProceduresApi&&) = delete;
  CuGenericProceduresApi& operator=(const CuGenericProceduresApi&) = delete;
  CuGenericProceduresApi& operator=(CuGenericProceduresApi&&) = delete;

  void add (Matrices& output, const Matrices& params1, floatt value);

  void add (Matrices& output, const Matrices& params1, const Matrices& params2);

  void subtract (Matrices& output, const Matrices& params1, const Matrices& params2);

  void dotProduct (Matrices& output, const Matrices& params1, const Matrices& params2);

  void multiply (Matrices& output, const Matrices& params1, const Matrices& params2);

  void hadamardProduct (Matrices& output, const Matrices& params1, const Matrices& params2);

  void hadamardProductVec (Matrices& output, const Matrices& params1, const Matrices& params2);

  void tensorProduct (Matrices& output, const Matrices& params1, const Matrices& params2);

  void transpose (Matrices& output, const Matrices& params1);

  void sigmoid (Matrices& output, const Matrices& params);

  void dsigmoid (Matrices& output, const Matrices& params);

  void multiplyDSigmoid (Matrices& output, const Matrices& params);

  void linear (Matrices& output, const Matrices& params);

  void dlinear (Matrices& output, const Matrices& params);

  void tanh (Matrices& output, const Matrices& params);

  void dtanh (Matrices& output, const Matrices& params);

  void sin (Matrices& output, const Matrices& params);

  void dsin (Matrices& output, const Matrices& params);

  void multiplyDSin (Matrices& output, const Matrices& params);

  void relu (Matrices& output, const Matrices& params);

  void drelu (Matrices& output, const Matrices& params);

  void prelu (Matrices& output, const Matrices& params);

  void dprelu (Matrices& output, const Matrices& params);

  void softplus (Matrices& output, const Matrices& params);

  void dsoftplus (Matrices& output, const Matrices& params);

  private:
    CuProceduresApi* m_cuApi;
};

}

#endif /* MATRIXPROCEDURES_H */
