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

#ifndef OAP_CU_MATRIX_EX_UTILS_H
#define OAP_CU_MATRIX_EX_UTILS_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"
#include "MatrixEx.h"
#include "CuMatrixIndexUtilsCommon.h"

__hostdeviceinline__ void cuAux_initMatrixEx (MatrixEx& ex, const math::ComplexMatrix* matrix)
{
  ex.column = 0;
  ex.row = 0;
  ex.columns = gColumns (matrix);
  ex.rows = gRows (matrix);
}

__hostdeviceinline__ void cuAux_initMatrixExByThreads (MatrixEx& ex, const math::ComplexMatrix* matrix)
{
  HOST_INIT ();
  THREAD_INDICES_INIT ();

  ex.column = threadIndexX;
  ex.row = threadIndexY;
  ex.columns = gColumns (matrix);
  ex.rows = gRows (matrix);
}

__hostdeviceinline__ void cuAux_initMatrixExs (MatrixEx exs[3], const math::ComplexMatrix* output, const math::ComplexMatrix* params0, const math::ComplexMatrix* params1)
{
  cuAux_initMatrixEx (exs[0], output);
  cuAux_initMatrixEx (exs[1], params0);
  cuAux_initMatrixEx (exs[2], params1);
}

__hostdeviceinline__ void cuAux_initMatrixExsByThreads (MatrixEx exs[3], const math::ComplexMatrix* output, const math::ComplexMatrix* params0, const math::ComplexMatrix* params1)
{
  cuAux_initMatrixExByThreads (exs[0], output);
  cuAux_initMatrixExByThreads (exs[1], params0);
  cuAux_initMatrixExByThreads (exs[2], params1);
}
#endif
