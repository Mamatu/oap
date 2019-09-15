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

#ifndef OAP_GENERIC_ARNOLDI_API_H
#define OAP_GENERIC_ARNOLDI_API_H

#include "GenericCoreApi.h"
#include "MatrixInfo.h"

namespace oap { namespace generic {

namespace {
inline void aux_swapPointers(math::Matrix** a, math::Matrix** b)
{
   math::Matrix* temp = *b;
  *b = *a;
  *a = temp;
}
}

template<typename Arnoldi, typename Api, typename GetReValue, typename GetImValue>
void iram_fVplusfq(Arnoldi& ar, uintt k, Api& api, GetReValue&& getReValue, GetImValue&& getImValue)
{
  floatt reqm_k = getReValue (ar.m_Q, ar.m_Qcolumns * (ar.m_Qrows - 1) + k);
  floatt imqm_k = 0;

  if (ar.m_matrixInfo.isIm)
  {
    traceFunction();
    imqm_k = getImValue(ar.m_Q, ar.m_Qcolumns * (ar.m_Qrows - 1) + k);
  }

  floatt reBm_k = getReValue(ar.m_H, ar.m_Hcolumns * (k + 1) + k);
  floatt imBm_k = 0;

  if (ar.m_matrixInfo.isIm)
  {
    traceFunction();
    imBm_k = getImValue (ar.m_H, ar.m_Hcolumns * (k + 1) + k);
  }

  api.getVector(ar.m_v, ar.m_vrows, ar.m_V, k);
  api.multiplyConstant(ar.m_f1, ar.m_v, reBm_k, imBm_k);
  api.setZeroMatrix(ar.m_v);

  api.multiplyConstant(ar.m_f, ar.m_f, reqm_k, imqm_k);
  api.add(ar.m_f, ar.m_f1, ar.m_f);

  api.magnitude(ar.m_FValue, ar.m_f);
}

namespace
{
/**
 * Inputs: ar.m_I ar.m_H
 * Outputs: ar.m_Q1, ar.m_R1, ar.m_H
 */
template<typename Arnoldi, typename Api>
void _qr (Arnoldi& ar, Api& api)
{
  api.QRGR (ar.m_Q1, ar.m_R1, ar.m_I, ar.m_Q, ar.m_R2, ar.m_G, ar.m_GT);
  //api.QRHT (ar.m_Q1, ar.m_R1, ar.m_I, ar.m_HT_V, ar.m_HT_VT, ar.m_HT_P, ar.m_HT_VVT);
}

}

/**
 * Inputs: ar.m_H
 * Outputs: ar.m_Q1, ar.m_R1, ar.m_H
 */
template<typename Arnoldi, typename Api>
void qrIteration (Arnoldi& ar, Api& api)
{
  math::Matrix* tempI = ar.m_I;
  ar.m_I = ar.m_H;

  _qr (ar, api);

  ar.m_I = tempI;
}

template<typename Arnoldi, typename Api>
void shiftedQRIteration (Arnoldi& ar, Api& api, size_t idx)
{
  api.setDiagonal (ar.m_I, ar.m_unwanted[idx].re(), ar.m_unwanted[idx].im());
  api.substract (ar.m_I, ar.m_H, ar.m_I);

  _qr (ar, api);
}

namespace iram_shiftedQRIteration
{

template<typename Arnoldi>
math::Matrix* getQ (Arnoldi& ar)
{
  return ar.m_Q1;  
}

template<typename Arnoldi>
math::Matrix* getR (Arnoldi& ar)
{
  return ar.m_R1;  
}

/**
 * Inputs: ar.m_H
 * Outputs: ar.m_Q1, ar.m_R1, ar.m_H
 */
template<typename Arnoldi, typename Api>
void proc (Arnoldi& ar, Api& api)
{
  //debugAssert (!ar.m_unwanted.empty());

  api.setIdentity (ar.m_QJ);
  api.setIdentity (ar.m_Q);

  for (uint fa = 0; fa < ar.m_unwanted.size(); ++fa)
  {
    oap::generic::shiftedQRIteration (ar, api, fa);

    api.conjugateTranspose (ar.m_QT, ar.m_Q1);
    api.dotProduct (ar.m_HO, ar.m_H, ar.m_Q1);
    api.dotProduct (ar.m_H, ar.m_QT, ar.m_HO);
    api.dotProduct (ar.m_Q, ar.m_QJ, ar.m_Q1);
    aux_swapPointers (&ar.m_Q, &ar.m_QJ);
  }

  if (ar.m_unwanted.size() % 2 != 0)
  {
    aux_swapPointers(&ar.m_Q, &ar.m_QJ);
  }
}

}

template<typename Arnoldi, typename NewKernelMatrix>
void allocStage1 (Arnoldi& ar, const math::MatrixInfo& matrixInfo, NewKernelMatrix&& newKernelMatrix)
{
  ar.m_vrows = matrixInfo.m_matrixDim.rows;
  ar.m_w = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, matrixInfo.m_matrixDim.rows);
  ar.m_v = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, matrixInfo.m_matrixDim.rows);
  ar.m_v1 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, matrixInfo.m_matrixDim.rows);
  ar.m_v2 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, matrixInfo.m_matrixDim.rows);
  ar.m_f = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, matrixInfo.m_matrixDim.rows);
  ar.m_f1 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, matrixInfo.m_matrixDim.rows);
  ar.m_vh = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, matrixInfo.m_matrixDim.rows);
  ar.m_vs = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, matrixInfo.m_matrixDim.rows);
}

template<typename Arnoldi, typename NewKernelMatrix, typename NewHostMatrix>
void allocStage2 (Arnoldi& ar, const math::MatrixInfo& matrixInfo, uint k, NewKernelMatrix&& newKernelMatrix, NewHostMatrix&& newHostMatrix)
{
  ar.m_V = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, matrixInfo.m_matrixDim.rows);
  ar.m_V1 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, matrixInfo.m_matrixDim.rows);
  ar.m_V2 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, matrixInfo.m_matrixDim.rows);
  ar.m_EV = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, matrixInfo.m_matrixDim.rows);
  ar.m_transposeV = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, matrixInfo.m_matrixDim.rows, k);

  ar.m_hostV = newHostMatrix (matrixInfo.isRe, matrixInfo.isIm, k, matrixInfo.m_matrixDim.rows);
}

template<typename Arnoldi, typename NewKernelMatrix>
void allocStage3 (Arnoldi& ar, const math::MatrixInfo& matrixInfo, uint k, NewKernelMatrix&& newKernelMatrix)
{
  traceFunction();
  ar.m_h = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  ar.m_s = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  ar.m_H = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_G = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_GT = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_HO = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_triangularH = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_Q1 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_Q2 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_QT = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_R1 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_R2 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_QJ = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_I = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_Q = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_QT1 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_QT2 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_q = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);

  ar.m_HT_V = newKernelMatrix (matrixInfo.isRe, matrixInfo.isIm, 1, k);
  ar.m_HT_VT = newKernelMatrix (matrixInfo.isRe, matrixInfo.isIm, k, 1);
  ar.m_HT_P = newKernelMatrix (matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_HT_VVT = newKernelMatrix (matrixInfo.isRe, matrixInfo.isIm, k, k);
}

template<typename Arnoldi, typename DeleteKernelMatrix>
void deallocStage1 (Arnoldi& ar, DeleteKernelMatrix&& deleteKernelMatrix)
{
  deleteKernelMatrix (ar.m_w);
  deleteKernelMatrix (ar.m_v);
  deleteKernelMatrix (ar.m_v1);
  deleteKernelMatrix (ar.m_v2);
  deleteKernelMatrix (ar.m_f);
  deleteKernelMatrix (ar.m_f1);
  deleteKernelMatrix (ar.m_vh);
  deleteKernelMatrix (ar.m_vs);
}

template<typename Arnoldi, typename DeleteKernelMatrix, typename DeleteHostMatrix>
void deallocStage2 (Arnoldi& ar, DeleteKernelMatrix&& deleteKernelMatrix, DeleteHostMatrix&& deleteHostMatrix)
{
  deleteKernelMatrix (ar.m_V);
  deleteKernelMatrix (ar.m_V1);
  deleteKernelMatrix (ar.m_V2);
  deleteKernelMatrix (ar.m_EV);
  deleteKernelMatrix (ar.m_transposeV);

  deleteHostMatrix (ar.m_hostV);
}

template<typename Arnoldi, typename DeleteKernelMatrix>
void deallocStage3(Arnoldi& ar, DeleteKernelMatrix&& deleteKernelMatrix)
{
  deleteKernelMatrix (ar.m_h);
  deleteKernelMatrix (ar.m_s);
  deleteKernelMatrix (ar.m_H);
  deleteKernelMatrix (ar.m_G);
  deleteKernelMatrix (ar.m_GT);
  deleteKernelMatrix (ar.m_HO);
  deleteKernelMatrix (ar.m_triangularH);
  deleteKernelMatrix (ar.m_Q1);
  deleteKernelMatrix (ar.m_Q2);
  deleteKernelMatrix (ar.m_QT);
  deleteKernelMatrix (ar.m_R1);
  deleteKernelMatrix (ar.m_R2);
  deleteKernelMatrix (ar.m_QJ);
  deleteKernelMatrix (ar.m_I);
  deleteKernelMatrix (ar.m_Q);
  deleteKernelMatrix (ar.m_QT1);
  deleteKernelMatrix (ar.m_QT2);
  deleteKernelMatrix (ar.m_q);

  deleteKernelMatrix (ar.m_HT_V);
  deleteKernelMatrix (ar.m_HT_VT);
  deleteKernelMatrix (ar.m_HT_P);
  deleteKernelMatrix (ar.m_HT_VVT);
}

}}

#endif
