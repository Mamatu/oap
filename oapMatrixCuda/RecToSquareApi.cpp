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

#include "RecToSquareApi.h"

#include "oapCudaMatrixUtils.h"
#include "oapDeviceMatrixUPtr.h"

#include "CuProceduresApi.h"

namespace oap
{

RecToSquareApi::RecToSquareApi (CuProceduresApi& api, const math::Matrix* recHostMatrix, bool deallocate) :
m_api(nullptr), m_rec (recHostMatrix, deallocate), m_sq (api, m_rec)
{}

RecToSquareApi::RecToSquareApi (const math::Matrix* recHostMatrix, bool deallocate) :
m_api(new CuProceduresApi()), m_rec (recHostMatrix, deallocate), m_sq (*m_api, m_rec)
{}

RecToSquareApi::~RecToSquareApi()
{
  delete m_api;
}

math::MatrixInfo RecToSquareApi::getMatrixInfo() const
{
  debugFunc ();

  math::MatrixInfo minfo = m_rec.getMatrixInfo ();

  if (minfo.m_matrixDim.columns != minfo.m_matrixDim.rows)
  {
    minfo.m_matrixDim.columns = minfo.m_matrixDim.rows;
  }

  return minfo;
}

math::Matrix* RecToSquareApi::createDeviceMatrix()
{
  debugFunc ();

  const math::MatrixInfo minfo = m_rec.getMatrixInfo ();

  if (minfo.m_matrixDim.columns == minfo.m_matrixDim.rows)
  {
    return m_rec.createDeviceMatrix ();
  }

  return m_sq.createDeviceMatrix ();
}

math::Matrix* RecToSquareApi::createDeviceSubMatrix(uintt rindex, uintt rlength)
{
  debugFunc ();

  math::MatrixInfo minfo = m_rec.getMatrixInfo ();
  minfo.m_matrixDim = {minfo.m_matrixDim.rows, rlength};

  math::Matrix* doutput = oap::cuda::NewDeviceMatrix (minfo);

  try
  {
    return getDeviceSubMatrix (rindex, rlength, doutput);
  }
  catch (...)
  {
    oap::cuda::DeleteDeviceMatrix (doutput);
    throw;
  }
}

math::Matrix* RecToSquareApi::getDeviceSubMatrix(uintt rindex, uintt rlength, math::Matrix* dmatrix)
{
  debugFunc ();

  if (dmatrix == nullptr)
  {
    return createDeviceSubMatrix (rindex, rlength); 
  }

  const math::MatrixInfo minfo = m_rec.getMatrixInfo ();

  if (minfo.m_matrixDim.columns == minfo.m_matrixDim.rows)
  {
    return m_rec.getDeviceSubMatrix (rindex, rlength, dmatrix);
  }

  return m_sq.getDeviceSubMatrix (rindex, rlength, dmatrix);
}

math::Matrix* RecToSquareApi::createDeviceRowVector(uintt index)
{
  debugFunc ();

  const math::MatrixInfo minfo = m_rec.getMatrixInfo ();

  math::Matrix* doutput = oap::cuda::NewDeviceReMatrix (minfo.m_matrixDim.rows, 1);

  try
  {
    return getDeviceRowVector (index, doutput);
  }
  catch (...)
  {
    oap::cuda::DeleteDeviceMatrix (doutput);
    throw;
  }
}

math::Matrix* RecToSquareApi::getDeviceRowVector(uintt index, math::Matrix* dmatrix)
{
  debugFunc ();

  if (dmatrix == nullptr)
  {
    return createDeviceRowVector (index); 
  }

  const math::MatrixInfo minfo = m_rec.getMatrixInfo ();

  if (minfo.m_matrixDim.columns == minfo.m_matrixDim.rows)
  {
    return m_rec.getDeviceSubMatrix (index, 1, dmatrix);
  }

  return m_sq.getDeviceSubMatrix (index, 1, dmatrix);
}

}
