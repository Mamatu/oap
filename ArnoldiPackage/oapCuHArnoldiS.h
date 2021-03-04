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

#ifndef OAP_CUH_ARNOLDI_S_H
#define OAP_CUH_ARNOLDI_S_H

#include "GenericCoreApi.h"
#include "MatrixInfo.h"
#include "Matrix.h"

#include "ArnoldiUtils.h"
#include "oapMatricesContext.h"

namespace oap { namespace generic {

class CuHArnoldiS : public oap::generic::MatricesContext
{
  public:

    CuHArnoldiS ():
      m_w(NULL),
      m_f(NULL),
      m_f1(NULL),
      m_vh(NULL),
      m_h(NULL),
      m_s(NULL),
      m_vs(NULL),
      m_V(NULL),
      m_transposeV(NULL),
      m_V1(NULL),
      m_V2(NULL),
      m_H(NULL),
      m_HC(NULL),
      m_triangularH(NULL),
      m_H2(NULL),
      m_I(NULL),
      m_v(NULL),
      m_QT(NULL),
      m_Q1(NULL),
      m_Q2(NULL),
      m_R1(NULL),
      m_R2(NULL),
      m_HO(NULL),
      m_Q(NULL),
      m_QT1(NULL),
      m_QT2(NULL),
      m_hostV(NULL),
      m_QJ(NULL),
      m_q(NULL),
      m_EV(NULL)
    {}

    std::vector<EigenPair> m_unwanted;
    math::ComplexMatrix* m_w;
    math::ComplexMatrix* m_f;
    math::ComplexMatrix* m_f1;
    math::ComplexMatrix* m_vh;
    math::ComplexMatrix* m_h;
    math::ComplexMatrix* m_s;
    math::ComplexMatrix* m_vs;
    math::ComplexMatrix* m_V;
    math::ComplexMatrix* m_transposeV;
    math::ComplexMatrix* m_V1;
    math::ComplexMatrix* m_V2;
    math::ComplexMatrix* m_H;
    math::ComplexMatrix* m_HC;
    math::ComplexMatrix* m_triangularH;
    math::ComplexMatrix* m_H2;
    math::ComplexMatrix* m_I;
    math::ComplexMatrix* m_v;
    math::ComplexMatrix* m_QT;
    math::ComplexMatrix* m_Q1;
    math::ComplexMatrix* m_Q2;
    math::ComplexMatrix* m_R1;
    math::ComplexMatrix* m_R2;
    math::ComplexMatrix* m_HO;
    math::ComplexMatrix* m_Q;
    math::ComplexMatrix* m_QT1;
    math::ComplexMatrix* m_QT2;
    math::ComplexMatrix* m_QJ;
    math::ComplexMatrix* m_q;
    math::ComplexMatrix* m_EV;

    math::ComplexMatrix* m_hostV;

    floatt m_previousFValue;
    floatt m_FValue;

    uint m_transposeVcolumns;
    uint m_hrows;
    uint m_scolumns;
    uint m_vscolumns;
    uint m_vsrows;
    uint m_vrows;
    uint m_qrows;
    uint m_Hcolumns;
    uint m_Hrows;
    uint m_triangularHcolumns;
    uint m_Qrows;
    uint m_Qcolumns;
};

}}

#endif
