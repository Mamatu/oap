/*
 * Copyright 2016 - 2019 Marcin Matula
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

namespace oap { namespace generic {

class CuHArnoldiS
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
      m_v1(NULL),
      m_v2(NULL),
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
      m_GT(NULL),
      m_G(NULL),
      m_EV(NULL)
    {}

    std::vector<EigenPair> m_unwanted;
    math::Matrix* m_w;
    math::Matrix* m_f;
    math::Matrix* m_f1;
    math::Matrix* m_vh;
    math::Matrix* m_h;
    math::Matrix* m_s;
    math::Matrix* m_vs;
    math::Matrix* m_V;
    math::Matrix* m_transposeV;
    math::Matrix* m_V1;
    math::Matrix* m_V2;
    math::Matrix* m_H;
    math::Matrix* m_HC;
    math::Matrix* m_triangularH;
    math::Matrix* m_H2;
    math::Matrix* m_I;
    math::Matrix* m_v;
    math::Matrix* m_v1;
    math::Matrix* m_v2;
    math::Matrix* m_QT;
    math::Matrix* m_Q1;
    math::Matrix* m_Q2;
    math::Matrix* m_R1;
    math::Matrix* m_R2;
    math::Matrix* m_HO;
    math::Matrix* m_Q;
    math::Matrix* m_QT1;
    math::Matrix* m_QT2;
    math::Matrix* m_QJ;
    math::Matrix* m_q;
    math::Matrix* m_GT;
    math::Matrix* m_G;
    math::Matrix* m_EV;

    math::Matrix* m_hostV;

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
