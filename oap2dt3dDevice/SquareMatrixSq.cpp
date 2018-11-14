/*
 * Copyright 2016 - 2018 Marcin Matula
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

#include "SquareMatrix.h"
#include "oapCudaMatrixUtils.h"

#include "oapDeviceMatrixUPtr.h"

namespace oap
{

math::Matrix* SquareMatrix::SqMatrix::getMatrix ()
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

math::Matrix* SquareMatrix::SqMatrix::getMatrixT ()
{
  debugFunc ();

  math::MatrixInfo minfo = m_orig.getMatrixInfo ();
  minfo.m_matrixDim = {minfo.m_matrixDim.rows, minfo.m_matrixDim.columns};

  if(!m_matrixTInfo.isInitialized () || m_matrixTInfo != minfo)
  {
    math::Matrix* matrix = getMatrix ();
    m_matrixT = resetMatrix (m_matrixT, minfo);

    m_api.transpose (m_matrixT, matrix);
    m_matrixTInfo = minfo;
  }

  return m_matrixT;
}

math::Matrix* SquareMatrix::SqMatrix::getRowVector (uintt index)
{
  debugFunc ();

  math::MatrixInfo minfo = m_orig.getMatrixInfo ();
  minfo.m_matrixDim = {minfo.m_matrixDim.columns, 1};

  checkArgs (index, 1, getMatrixInfo ());

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

math::Matrix* SquareMatrix::SqMatrix::getSubMatrix (uintt rindex, uintt rlength)
{
  debugFunc ();

  checkArgs (rindex, rlength, getMatrixInfo ());

  math::MatrixInfo minfo = m_orig.getMatrixInfo ();

  math::Matrix* hmatrix = m_orig.getHostSubMatrix (0, rindex, minfo.m_matrixDim.columns, rlength);

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

void SquareMatrix::SqMatrix::destroyMatrices ()
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

}

