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
#include "oapMatricesContext.h"

#include "ArnoldiUtils.h"

namespace oap {

enum class QRType
{
  NONE = -1,
  QRGR, // givens rotations
  QRHT, // housholder reflection
};

enum class VecMultiplicationType
{
  TYPE_EIGENVECTOR,
  TYPE_WV
};

enum class InitVVectorType
{
  RANDOM,
  FIRST_VALUE_IS_ONE
};

namespace generic {

namespace {
inline void aux_swapPointers(math::Matrix** a, math::Matrix** b)
{
   math::Matrix* temp = *b;
  *b = *a;
  *a = temp;
}
}

template<typename Arnoldi, typename Api, typename VecMultiplication>
void iram_executeInit (Arnoldi& ar, Api& api, VecMultiplication&& multiply)
{
  multiply (ar.m_w, ar.m_v, api, VecMultiplicationType::TYPE_WV);
  api.setVector (ar.m_V, 0, ar.m_v, ar.m_vrows);
  api.transpose (ar.m_transposeV, ar.m_V);
  api.dotProduct (ar.m_h, ar.m_transposeV, ar.m_w);
  api.dotProduct (ar.m_vh, ar.m_V, ar.m_h);
  api.substract (ar.m_f, ar.m_w, ar.m_vh);
  api.setVector (ar.m_H, 0, ar.m_h, 1);
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
template<typename Api>
void _qr (math::Matrix* Q, math::Matrix* R, math::Matrix* H, const math::MatrixInfo& hinfo, oap::generic::MatricesContext& context, const std::string& memType, Api& api, QRType qrtype)
{
   auto getter = context.getter();
  if (qrtype == QRType::QRGR)
  {
    math::Matrix* aux_Q = getter.useMatrix (hinfo, memType);
    math::Matrix* aux_R = getter.useMatrix (hinfo, memType);
    math::Matrix* aux_G = getter.useMatrix (hinfo, memType);
    math::Matrix* aux_GT = getter.useMatrix (hinfo, memType);
    api.QRGR (Q, R, H, aux_Q, aux_R, aux_G, aux_GT);
  }
  else if (qrtype == QRType::QRHT)
  {
    math::Matrix* aux_V = getter.useMatrix (hinfo.isRe, hinfo.isIm, 1, hinfo.rows(), memType);
    math::Matrix* aux_VT = getter.useMatrix (hinfo.isRe, hinfo.isIm, hinfo.columns(), 1, memType);
    math::Matrix* aux_P = getter.useMatrix (hinfo, memType);
    math::Matrix* aux_VVT = getter.useMatrix (hinfo, memType);
    api.QRHT (Q, R, H, aux_V, aux_VT, aux_P, aux_VVT);
  }
}
}

/**
 * Inputs: ar.m_H
 * Outputs: ar.m_Q1, ar.m_R1, ar.m_H
 */
template<typename Arnoldi, typename Api>
void qrIteration (Arnoldi& ar, Api& api)
{
  _qr (ar.m_H, ar, api);
}

namespace iram_singleShiftedQRIteration
{

struct InOutArgs
{
  math::Matrix* Q;
  math::Matrix* R;
  math::Matrix* H;
};

struct InArgs
{
  const std::vector<EigenPair>& unwanted;
  const math::MatrixInfo hinfo;
  const std::string memType;
};

template<typename Api>
void proc (InOutArgs& io, const InArgs& iargs, oap::generic::MatricesContext& cm, Api& api, size_t idx, oap::QRType qrtype)
{
  auto getter = cm.getter ();
  math::Matrix* aux_HI = getter.useMatrix (iargs.hinfo, iargs.memType);

  api.setDiagonal (aux_HI, iargs.unwanted[idx].re(), iargs.unwanted[idx].im());
  api.substract (aux_HI, io.H, aux_HI);

  _qr (io.Q, io.R, aux_HI, iargs.hinfo, cm, iargs.memType, api, qrtype);
}
}

namespace iram_shiftedQRIterations
{

struct InOutArgs
{
  math::Matrix* Q;
  math::Matrix* H;
};

struct InArgs
{
  const std::vector<EigenPair>& unwanted;
  const math::MatrixInfo hinfo;
  const std::string memType;
};

template<typename Api>
void proc (InOutArgs& io, const InArgs& iargs, oap::generic::MatricesContext& cm, Api& api, oap::QRType qrtype)
{
  //debugAssert (!ar.m_unwanted.empty());
  auto getter = cm.getter ();

  math::Matrix* aux_QJ = getter.useMatrix (iargs.hinfo, iargs.memType);
  math::Matrix* aux_QT = getter.useMatrix (iargs.hinfo, iargs.memType);
  math::Matrix* aux_HO = getter.useMatrix (iargs.hinfo, iargs.memType);
  math::Matrix* aux_Q1 = getter.useMatrix (iargs.hinfo, iargs.memType);
  math::Matrix* aux_R = getter.useMatrix (iargs.hinfo, iargs.memType);

  api.setIdentity (aux_Q1);
  api.setIdentity (aux_QJ);

  oap::generic::iram_singleShiftedQRIteration::InOutArgs io1;
  io1.Q = aux_Q1;
  io1.R = aux_R;
  io1.H = io.H;

  oap::generic::iram_singleShiftedQRIteration::InArgs iargs1 = {iargs.unwanted, iargs.hinfo, iargs.memType};

  for (uint fa = 0; fa < iargs.unwanted.size(); ++fa)
  {
    oap::generic::iram_singleShiftedQRIteration::proc (io1, iargs1, cm, api, fa, qrtype);

    api.conjugateTranspose (aux_QT, aux_Q1);
    api.dotProduct (aux_HO, io.H, aux_Q1);
    api.dotProduct (io.H, aux_QT, aux_HO);
    api.dotProduct (io.Q, aux_QJ, aux_Q1);
    aux_swapPointers (&io.Q, &aux_QJ);
  }

  if (iargs.unwanted.size() % 2 != 0)
  {
    aux_swapPointers (&io.Q, &aux_QJ);
  }
}
}

template<typename Arnoldi, typename CalcApi, typename CopyKernelMatrixToKernelMatrix>
void calcTriangularH (Arnoldi& ar, uintt count, CalcApi& capi, CopyKernelMatrixToKernelMatrix&& copyKernelMatrixToKernelMatrix, oap::QRType qrtype)
{
  bool status = false;
  capi.setIdentity (ar.m_Q);
  math::Matrix* QJ = ar.m_QJ;
  math::Matrix* Q = ar.m_Q;

  status = capi.isUpperTriangular (ar.m_triangularH);

  for (uint idx = 0; idx < count && status == false; ++idx)
  {
    _qr (ar.m_triangularH, ar, capi, qrtype);

    capi.dotProduct (ar.m_triangularH, ar.m_R1, Q);
    capi.dotProduct (QJ, ar.m_Q1, Q);
    aux_swapPointers (&QJ, &Q);
    status = capi.isUpperTriangular (ar.m_triangularH);
  }

  aux_swapPointers (&QJ, &Q);

  copyKernelMatrixToKernelMatrix (ar.m_Q1, QJ);
}

namespace iram_calcTriangularH_Host
{

struct InOutArgs
{
  math::Matrix* H;
  math::Matrix* Q;
};

struct InArgs
{
  const math::MatrixInfo& thInfo;
  oap::generic::MatricesContext& context;
  const std::string& memType;
  uintt count;
  oap::QRType qrtype;
};

template<typename CalcApi, typename CopyKernelMatrixToKernelMatrix>
void proc (InOutArgs& io, const InArgs& iargs, CalcApi& capi, CopyKernelMatrixToKernelMatrix&& copyKernelMatrixToKernelMatrix)
{
  bool status = false;
  auto getter1 = iargs.context.getter ();

  math::Matrix* aux_Q = getter1.useMatrix (iargs.thInfo, iargs.memType);
  math::Matrix* aux_Q1 = getter1.useMatrix (iargs.thInfo, iargs.memType);

  math::Matrix* aux_R = getter1.useMatrix (iargs.thInfo, iargs.memType);

  capi.setIdentity (aux_Q);

  status = capi.isUpperTriangular (io.H);

  for (uint idx = 0; idx < iargs.count && status == false; ++idx)
  {
    _qr (io.Q, aux_R, io.H, iargs.thInfo, iargs.context, iargs.memType, capi, iargs.qrtype);

    capi.dotProduct (io.H, aux_R, io.Q);
    capi.dotProduct (aux_Q1, io.Q, aux_Q);
    aux_swapPointers (&aux_Q1, &aux_Q);
    status = capi.isUpperTriangular (io.H);
  }

  aux_swapPointers (&aux_Q1, &aux_Q);

  copyKernelMatrixToKernelMatrix (io.Q, aux_Q1);
}

}

namespace iram_calcTriangularH_Generic
{

struct InOutArgs
{
  math::Matrix* H;
  math::Matrix* Q;
};

struct InArgs
{
  math::MatrixInfo hinfo;
  oap::generic::MatricesContext& context;
  const std::string& memType;
};

template<typename CalcApi>
void proc (InOutArgs& io, const InArgs& iargs, CalcApi& capi)
{
  auto getter = iargs.context.getter();

  math::Matrix* aux_R = getter.useMatrix (iargs.hinfo, iargs.memType);
  math::Matrix* aux_R1 = getter.useMatrix (iargs.hinfo, iargs.memType);

  math::Matrix* aux_Q = getter.useMatrix (iargs.hinfo, iargs.memType);
  math::Matrix* aux_Q1 = getter.useMatrix (iargs.hinfo, iargs.memType);
  math::Matrix* aux_Q2 = getter.useMatrix (iargs.hinfo, iargs.memType);

  math::Matrix* aux_G = getter.useMatrix (iargs.hinfo, iargs.memType);
  math::Matrix* aux_GT = getter.useMatrix (iargs.hinfo, iargs.memType);

  capi.calcTriangularH (io.H, io.Q, aux_R, aux_Q, aux_Q1, aux_Q2, aux_R1, aux_G, aux_GT);
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
void allocStage3 (Arnoldi& ar, const math::MatrixInfo& matrixInfo, uint k, NewKernelMatrix&& newKernelMatrix, oap::QRType qrtype)
{
  traceFunction();
  ar.m_h = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  ar.m_s = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  ar.m_H = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_HO = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_triangularH = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_Q1 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_Q2 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_QT = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_R1 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_QJ = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_I = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_QT1 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_QT2 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_q = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);

  ar.m_Q = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  ar.m_R2 = newKernelMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
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
  deleteKernelMatrix (ar.m_HO);
  deleteKernelMatrix (ar.m_triangularH);
  deleteKernelMatrix (ar.m_Q1);
  deleteKernelMatrix (ar.m_Q2);
  deleteKernelMatrix (ar.m_QT);
  deleteKernelMatrix (ar.m_R1);
  deleteKernelMatrix (ar.m_QJ);
  deleteKernelMatrix (ar.m_I);
  deleteKernelMatrix (ar.m_QT1);
  deleteKernelMatrix (ar.m_QT2);
  deleteKernelMatrix (ar.m_q);

  deleteKernelMatrix (ar.m_Q);
  deleteKernelMatrix (ar.m_R2);
}

}}

#endif
