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
{}

math::Matrix* MatrixCreator::createDeviceMatrix()
{
  const math::MatrixInfo minfo = m_ddl->getMatrixInfo ();
  if (minfo.m_matrixDim.columns == minfo.m_matrixDim.rows)
  {
    return m_ddl->createDeviceMatrix ();
  }

  return constructSquareMatrix (minfo);
}

math::Matrix* MatrixCreator::createDeviceRowVector(size_t index)
{
  const math::MatrixInfo minfo = m_ddl->getMatrixInfo ();

  math::Matrix* output = oap::cuda::NewDeviceReMatrix (minfo.m_matrixDim.rows, 1);
  return getDeviceRowVector (index, output);
}

math::Matrix* MatrixCreator::getDeviceRowVector(size_t index, math::Matrix* dmatrix)
{
  const math::MatrixInfo minfo = m_ddl->getMatrixInfo ();
  if (minfo.m_matrixDim.columns == minfo.m_matrixDim.rows)
  {
    return m_ddl->getDeviceRowVector (index, dmatrix);
  }

  oap::DeviceMatrixUPtr matrixT = getMatrixT (minfo);

  oap::DeviceMatrixUPtr rowVector = getRowVector (index, minfo);

  m_api.dotProduct (dmatrix, matrixT, rowVector);

  return dmatrix;
}

math::Matrix* MatrixCreator::constructSquareMatrix (const math::MatrixInfo& minfo)
{
  oap::DeviceMatrixUPtr matrix = m_ddl->createDeviceMatrix ();
  oap::DeviceMatrixUPtr matrixT = oap::cuda::NewDeviceReMatrix (minfo.m_matrixDim.rows, minfo.m_matrixDim.columns);
  m_api.transpose (matrixT, matrix);

  math::Matrix* output = oap::cuda::NewDeviceReMatrix (minfo.m_matrixDim.rows, minfo.m_matrixDim.rows);
  m_api.dotProduct (output, matrix, matrixT);
  return output;
}

math::Matrix* MatrixCreator::getMatrixT (const math::MatrixInfo& minfo)
{
  if(!m_matrixInfo.isInitialized () || m_matrixInfo != minfo)
  {
    oap::DeviceMatrixUPtr matrix = m_ddl->createDeviceMatrix ();
    m_matrixT = oap::cuda::NewDeviceReMatrix (minfo.m_matrixDim.rows, minfo.m_matrixDim.columns);
    m_api.transpose (m_matrixT, matrix);
    m_matrixInfo = minfo;
  }

  return m_matrixT.get();
}

math::Matrix* MatrixCreator::getRowVector (size_t index, const math::MatrixInfo& minfo)
{
  if (!m_matrixInfo.isInitialized () || m_matrixInfo != minfo)
  {
    m_rowVector = m_ddl->createDeviceRowVector (index);
  }

  return m_ddl->getDeviceRowVector (index, m_rowVector);
}

}
