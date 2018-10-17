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

#include "MatrixCreator.h"
#include "oapCudaMatrixUtils.h"

#include "oapDeviceMatrixUPtr.h"

namespace oap
{

MatrixCreator::MatrixCreator(DeviceDataLoader* ddl) : m_ddl(ddl)
{}

MatrixCreator::~MatrixCreator()
{
}

math::MatrixInfo MatrixCreator::getMatrixInfo() const
{
  debugFunc ();

  math::MatrixInfo minfo = m_ddl->getMatrixInfo ();

  if (minfo.m_matrixDim.columns != minfo.m_matrixDim.rows)
  {
    minfo.m_matrixDim.columns = minfo.m_matrixDim.rows;
  }

  return minfo;
}

math::Matrix* MatrixCreator::createDeviceMatrix()
{
  debugFunc ();

  const math::MatrixInfo minfo = m_ddl->getMatrixInfo ();
  if (minfo.m_matrixDim.columns == minfo.m_matrixDim.rows)
  {
    return m_ddl->createDeviceMatrix ();
  }

  return constructSquareMatrix (minfo);
}

math::Matrix* MatrixCreator::createDeviceSubMatrix(size_t rindex, size_t rlength)
{
  debugFunc ();

  const math::MatrixInfo minfo = m_ddl->getMatrixInfo ();

  math::Matrix* doutput = oap::cuda::NewDeviceReMatrix (minfo.m_matrixDim.rows, rlength);

  return getDeviceSubMatrix (rindex, rlength, doutput);
}

math::Matrix* MatrixCreator::getDeviceSubMatrix(size_t rindex, size_t rlength, math::Matrix* dmatrix)
{
  debugFunc ();

  const math::MatrixInfo minfo = m_ddl->getMatrixInfo ();

  if (minfo.m_matrixDim.columns == minfo.m_matrixDim.rows)
  {
    return m_ddl->getDeviceSubMatrix (0, rindex, minfo.m_matrixDim.columns, rlength, dmatrix);
  }

  math::Matrix* matrixT = getMatrixT (minfo);

  math::Matrix* subMatrix = getSubMatrix (rindex, rlength, minfo);

  m_api.dotProduct (dmatrix, subMatrix, matrixT);

  return dmatrix;
}

math::Matrix* MatrixCreator::createDeviceRowVector(size_t index)
{
  debugFunc ();

  const math::MatrixInfo minfo = m_ddl->getMatrixInfo ();

  math::Matrix* output = oap::cuda::NewDeviceReMatrix (minfo.m_matrixDim.rows, 1);
  return getDeviceRowVector (index, output);
}

math::Matrix* MatrixCreator::getDeviceRowVector(size_t index, math::Matrix* dmatrix)
{
  debugFunc ();

  const math::MatrixInfo minfo = m_ddl->getMatrixInfo ();

  if (minfo.m_matrixDim.columns == minfo.m_matrixDim.rows)
  {
    return m_ddl->getDeviceRowVector (index, dmatrix);
  }

  math::Matrix* matrixT = getMatrixT (minfo);

  math::Matrix* rowVector = getRowVector (index, minfo);

  m_api.dotProduct (dmatrix, rowVector, matrixT);

  return dmatrix;
}

math::Matrix* MatrixCreator::constructSquareMatrix (const math::MatrixInfo& minfo)
{
  debugFunc ();

  math::Matrix* matrix = getMatrix (minfo);
  math::Matrix* matrixT = getMatrixT (minfo);

  math::Matrix* output = oap::cuda::NewDeviceReMatrix (minfo.m_matrixDim.rows, minfo.m_matrixDim.rows);

  m_api.dotProduct (output, matrix, matrixT);

  return output;
}

math::Matrix* MatrixCreator::getMatrix (const math::MatrixInfo& minfo)
{
  debugFunc ();

  if(!m_matrixInfo.isInitialized () || m_matrixInfo != minfo)
  {
    m_matrix = m_ddl->createDeviceMatrix ();
    m_matrixInfo = minfo;
  }

  return m_matrix.get();
}

math::Matrix* MatrixCreator::getMatrixT (const math::MatrixInfo& minfo)
{
  debugFunc ();

  if(!m_matrixTInfo.isInitialized () || m_matrixTInfo != minfo)
  {
    math::Matrix* matrix = getMatrix (minfo);
    m_matrixT = oap::cuda::NewDeviceReMatrix (minfo.m_matrixDim.rows, minfo.m_matrixDim.columns);
    m_api.transpose (m_matrixT, matrix);
    m_matrixTInfo = minfo;
  }

  return m_matrixT.get();
}

math::Matrix* MatrixCreator::getRowVector (size_t index, const math::MatrixInfo& minfo)
{
  debugFunc ();

  if (!m_rowVectorInfo.isInitialized () || m_rowVectorInfo != minfo)
  {
    m_rowVector = m_ddl->createDeviceRowVector (index);
    m_rowVectorInfo = minfo;
    return m_rowVector.get ();
  }

  return m_ddl->getDeviceRowVector (index, m_rowVector);
}

math::Matrix* MatrixCreator::getSubMatrix (size_t rindex, size_t rlength, const math::MatrixInfo& minfo)
{
  debugFunc ();

  if (!m_subMatrixInfo.isInitialized () || m_subMatrixInfo != minfo)
  {
    m_subMatrix = m_ddl->createDeviceSubMatrix (0, rindex, minfo.m_matrixDim.columns, rlength);
    m_subMatrixInfo = minfo;
    return m_subMatrix.get ();
  }

  return m_ddl->getDeviceSubMatrix (0, rindex, minfo.m_matrixDim.columns, minfo.m_matrixDim.rows, m_subMatrix);
}

}
