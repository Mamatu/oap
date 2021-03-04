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

#include "SquareMatrixApi.h"

#include "oapCudaMatrixUtils.h"
#include "oapDeviceMatrixUPtr.h"

#include "CuProceduresApi.h"

namespace oap
{

math::ComplexMatrix* SquareMatrixApi::getMatrix ()
{
  debugFunc ();

  math::MatrixInfo minfo = m_orig.getMatrixInfo ();

  if(!m_matrixInfo.isInitialized () || m_matrixInfo != minfo)
  {
    m_matrix = resetMatrix (m_matrix, minfo);
    m_matrix = m_orig.getDeviceMatrix (m_matrix);
    m_matrixInfo = minfo;
  }

  return m_matrix;
}

math::ComplexMatrix* SquareMatrixApi::getMatrixT ()
{
  debugFunc ();

  math::MatrixInfo minfo = m_orig.getMatrixInfo ();
  minfo.m_matrixDim = {minfo.m_matrixDim.rows, minfo.m_matrixDim.columns};

  if(!m_matrixTInfo.isInitialized () || m_matrixTInfo != minfo)
  {
    math::ComplexMatrix* matrix = getMatrix ();
    m_matrixT = resetMatrix (m_matrixT, minfo);

    m_api.transpose (m_matrixT, matrix);
    m_matrixTInfo = minfo;
  }

  return m_matrixT;
}

math::ComplexMatrix* SquareMatrixApi::getRowVector (uintt index)
{
  debugFunc ();

  math::MatrixInfo minfo = m_orig.getMatrixInfo ();
  minfo.m_matrixDim = {minfo.m_matrixDim.columns, 1};

  if (!m_rowVectorInfo.isInitialized () || m_rowVectorInfo != minfo)
  {
    m_rowVector = resetMatrix (m_rowVector, minfo);
    m_rowVector = m_orig.getDeviceSubMatrix (index, 1, m_rowVector);
    m_rowVectorInfo = minfo;
  }
  else
  {
    m_rowVector = m_orig.getDeviceSubMatrix (index, 1, m_rowVector);
  }
  return m_rowVector;
}

math::ComplexMatrix* SquareMatrixApi::getSubMatrix (uintt rindex, uintt rlength)
{
  debugFunc ();

  math::MatrixInfo minfo = m_orig.getMatrixInfo ();

  math::ComplexMatrix* hmatrix = m_orig.getHostSubMatrix (0, rindex, minfo.m_matrixDim.columns, rlength);

  minfo = oap::host::GetMatrixInfo (hmatrix);

  if (!m_subMatrixInfo.isInitialized () || m_subMatrixInfo != minfo)
  {
    m_subMatrix = resetMatrix (m_subMatrix, minfo);
    m_subMatrix = m_orig.getDeviceSubMatrix (rindex, rlength, m_subMatrix);
    m_subMatrixInfo = minfo;
  }
  else
  {
    m_subMatrix = m_orig.getDeviceSubMatrix (rindex, rlength, m_subMatrix);
  }
  return m_subMatrix;
}

void SquareMatrixApi::destroyMatrices ()
{
  destroyMatrix (&m_matrix);
  destroyMatrix (&m_matrixT);
  destroyMatrix (&m_rowVector);
  destroyMatrix (&m_subMatrix);

  m_matrixInfo.deinitialize();
  m_matrixTInfo.deinitialize();
  m_rowVectorInfo.deinitialize();
  m_subMatrixInfo.deinitialize();
}

SquareMatrixApi::SquareMatrixApi (CuProceduresApi& api, RecMatrixApi& orig) : m_api(api), m_orig(orig), m_matrix (nullptr), m_matrixT (nullptr), m_rowVector (nullptr), m_subMatrix (nullptr)
{}

SquareMatrixApi::~SquareMatrixApi ()
{
  destroyMatrices ();
}

math::MatrixInfo SquareMatrixApi::getMatrixInfo () const
{
  auto minfo = m_orig.getMatrixInfo ();
  minfo.m_matrixDim.columns = minfo.m_matrixDim.rows;
  return minfo;
}

math::ComplexMatrix* SquareMatrixApi::createDeviceMatrix ()
{
  auto minfo = getMatrixInfo ();
  math::ComplexMatrix* matrix = oap::cuda::NewDeviceMatrix (minfo);
  return getDeviceMatrix (matrix);
}

math::ComplexMatrix* SquareMatrixApi::getDeviceMatrix (math::ComplexMatrix* dmatrix)
{
  debugFunc ();
  auto minfo = m_orig.getMatrixInfo ();

  math::ComplexMatrix* matrix = getMatrix ();
  math::ComplexMatrix* matrixT = getMatrixT ();

  math::ComplexMatrix* output = oap::cuda::NewDeviceReMatrix (minfo.m_matrixDim.rows, minfo.m_matrixDim.rows);

  m_api.dotProduct (output, matrix, matrixT);

  return output;
}

math::ComplexMatrix* SquareMatrixApi::getDeviceSubMatrix (uintt rindex, uintt rlength, math::ComplexMatrix* dmatrix)
{
  debugFunc ();
  auto minfo = m_orig.getMatrixInfo ();

  math::ComplexMatrix* matrixT = getMatrixT ();

  math::ComplexMatrix* subMatrix = getSubMatrix (rindex, rlength);

  auto subinfo = oap::cuda::GetMatrixInfo (subMatrix);
  auto dinfo = oap::cuda::GetMatrixInfo (dmatrix);

  if (dinfo.m_matrixDim.rows != subinfo.m_matrixDim.rows)
  {
    dinfo.m_matrixDim.rows = subinfo.m_matrixDim.rows;

    oap::cuda::DeleteDeviceMatrix (dmatrix);
    dmatrix = oap::cuda::NewDeviceMatrix (dinfo);
  }

  m_api.dotProduct (dmatrix, subMatrix, matrixT);

  return dmatrix;
}

void SquareMatrixApi::destroyMatrix(math::ComplexMatrix** matrix)
{
  if (matrix != nullptr && *matrix != nullptr)
  {
    oap::cuda::DeleteDeviceMatrix (*matrix);
    *matrix = nullptr;
  }
}

math::ComplexMatrix* SquareMatrixApi::resetMatrix (math::ComplexMatrix* matrix, const math::MatrixInfo& minfo)
{
  oap::cuda::DeleteDeviceMatrix (matrix);
  return oap::cuda::NewDeviceMatrix (minfo);
}

}
