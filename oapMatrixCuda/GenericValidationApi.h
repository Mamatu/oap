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

#ifndef OAP_GENERIC_VALIDATION_API_H
#define OAP_GENERIC_VALIDATION_API_H

#include "GenericCoreApi.h"

#include "MatrixInfo.h"
#include "oapProcedures.h"

namespace oap
{
namespace generic
{

inline void check_isEqualDim (const math::MatrixInfo& minfo1, const math::MatrixInfo& minfo2)
{
  debugAssert (minfo1.columns() == minfo2.columns());
  debugAssert (minfo1.rows() == minfo2.rows());
  debugAssert (minfo1.isRe == minfo2.isRe);
  debugAssert (minfo1.isIm == minfo2.isIm);
}

template<typename BasicMatrixApi>
void check_isEqualDim (math::Matrix* m1, math::Matrix* m2, BasicMatrixApi& bmApi)
{
  if (m1 != m2)
  {
    auto minfo1 = bmApi.getMatrixInfo (m1);
    auto minfo2 = bmApi.getMatrixInfo (m2);
    check_isEqualDim (minfo1, minfo2);
  }
}

inline void check_dotProduct (const math::MatrixInfo& output, const math::MatrixInfo& minfo0, const math::MatrixInfo& minfo1)
{
  const uintt output_columns = output.columns ();
  const uintt output_rows = output.rows ();

  const uintt params0_columns = minfo0.columns ();
  const uintt params0_rows = minfo0.rows ();

  const uintt params1_columns = minfo1.columns ();
  const uintt params1_rows = minfo1.rows ();

#ifdef CU_PROCEDURES_API_PRINT
  oap::cuda::PrintMatrixInfo("params0 = ", params0);
  oap::cuda::PrintMatrixInfo("params1 = ", params1);
  oap::cuda::PrintMatrixInfo("ouput = ", output);
#endif

  debugAssertMsg(params0_columns == params1_rows, "params0_columns = %u params1_rows = %u", params0_columns, params1_rows);
  debugAssertMsg(output_columns == params1_columns, "output_columns = %u params1_columns = %u", output_columns, params1_columns);
  debugAssertMsg(output_rows == params0_rows, "output_rows = %u params0_rows = %u", output_rows, params0_rows);
}

template<typename BasicMatrixApi>
void check_dotProduct (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, BasicMatrixApi& bmApi)
{
  auto oinfo = bmApi.getMatrixInfo (output);
  auto minfo0 = bmApi.getMatrixInfo (params0);
  auto minfo1 = bmApi.getMatrixInfo (params1);

  check_dotProduct (oinfo, minfo0, minfo1);
}

template<typename Dim>
void check_Size (const math::MatrixInfo& minfo, math::Matrix* matrix, const Dim& dim)
{
  const uintt columns = minfo.columns ();
  const uintt rows = minfo.rows ();

  debugAssert (dim[0] <= columns);
  debugAssert (dim[1] <= rows);
}

inline void check_dotProduct (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, generic::Dim32 dim,
                              const math::MatrixInfo& oinfo, const math::MatrixInfo& minfo1, const math::MatrixInfo& minfo2)
{
  check_Size (oinfo, output, dim[0]);
  check_Size (minfo1, params0, dim[1]);
  check_Size (minfo2, params1, dim[2]);

  const uintt output_columns = dim[0][0];
  const uintt output_rows = dim[0][1];

  const uintt params0_columns = dim[1][0];
  const uintt params0_rows = dim[1][1];

  const uintt params1_columns = dim[2][0];
  const uintt params1_rows = dim[2][1];

#ifdef CU_PROCEDURES_API_PRINT
  oap::cuda::PrintMatrixInfo("params0 = ", params0);
  oap::cuda::PrintMatrixInfo("params1 = ", params1);
  oap::cuda::PrintMatrixInfo("ouput = ", output);
#endif

  debugAssertMsg (params0_columns == params1_rows, "params0_columns = %u params1_rows = %u", params0_columns, params1_rows);
  debugAssertMsg (output_columns == params1_columns, "output_columns = %u params1_columns = %u", output_columns, params1_columns);
  debugAssertMsg (output_rows == params0_rows, "output_rows = %u params0_rows = %u", output_rows, params0_rows);
}

inline void check_dotProductPeriodic (math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
                                      const math::MatrixInfo& oinfo, const math::MatrixInfo& minfo1, const math::MatrixInfo& minfo2)
{
  const uintt output_columns = oinfo.columns();
  const uintt output_rows = oinfo.rows();

  const uintt params0_columns = minfo1.columns();
  const uintt params0_rows = minfo1.rows();

  const uintt params1_columns = minfo2.columns();
  const uintt params1_rows = minfo2.rows();

#ifdef CU_PROCEDURES_API_PRINT
  oap::cuda::PrintMatrixInfo("params0 = ", params0);
  oap::cuda::PrintMatrixInfo("params1 = ", params1);
  oap::cuda::PrintMatrixInfo("ouput = ", output);
#endif

  debugAssertMsg (params1_rows % params0_columns == 0, "params0_columns = %u params1_rows = %u", params0_columns, params1_rows);
  debugAssertMsg (output_columns == params1_columns, "output_columns = %u params1_columns = %u", output_columns, params1_columns);
  debugAssertMsg (output_rows % params0_rows == 0, "output_rows = %u params0_rows = %u", output_rows, params0_rows);
}

inline void check_dotProductDimPeriodic (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, generic::Dim32 dim, uintt periodicRows,
                                      const math::MatrixInfo& oinfo, const math::MatrixInfo& minfo1, const math::MatrixInfo& minfo2)
{
  check_Size (oinfo, output, dim[0]);
  check_Size (minfo1, params0, dim[1]);
  check_Size (minfo2, params1, dim[2]);

  const uintt d_output_columns = dim[0][0];
  const uintt d_output_rows = dim[0][1];

  const uintt d_params0_columns = dim[1][0];
  const uintt d_params0_rows = dim[1][1];

  const uintt d_params1_columns = dim[2][0];
  const uintt d_params1_rows = dim[2][1];

  const uintt output_columns = oinfo.columns();
  const uintt output_rows = oinfo.rows();

  const uintt params0_columns = minfo1.columns();
  const uintt params0_rows = minfo1.rows();

  const uintt params1_columns = minfo2.columns();
  const uintt params1_rows = minfo2.rows();

#ifdef CU_PROCEDURES_API_PRINT
  oap::cuda::PrintMatrixInfo("params0 = ", params0);
  oap::cuda::PrintMatrixInfo("params1 = ", params1);
  oap::cuda::PrintMatrixInfo("ouput = ", output);
#endif

  debugAssertMsg (params1_rows % d_params0_columns == 0, "params0_columns = %u d_params1_rows = %u", params0_columns, d_params1_rows);
  debugAssertMsg (d_output_columns == d_params1_columns, "d_output_columns = %u d_params1_columns = %u", d_output_columns, d_params1_columns);
  debugAssertMsg (output_rows % periodicRows == 0, "output_rows = %u d_params0_rows = %u", output_rows, params0_rows);
  debugAssertMsg (output_rows / periodicRows == params1_rows / d_params0_columns, "output_rows = %u d_params0_rows = %u params1_rows = %u d_params0_columns = %u", output_rows, d_params0_rows, params1_rows, d_params0_columns);
}

template<typename GetMatrixInfo>
void check_tensorProduct (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt columns, uintt rows, BasicMatrixApi<GetMatrixInfo>& bmApi)
{
  const uintt output_columns = columns;
  const uintt output_rows = rows;

  auto minfo0 = bmApi.getMatrixInfo (params0);
  auto minfo1 = bmApi.getMatrixInfo (params1);

  const uintt params0_columns = minfo0.columns ();
  const uintt params0_rows = minfo0.rows ();

  const uintt params1_columns = minfo1.columns ();
  const uintt params1_rows = minfo1.rows ();

#ifdef CU_PROCEDURES_API_PRINT
  oap::cuda::PrintMatrixInfo("params0 = ", params0);
  oap::cuda::PrintMatrixInfo("params1 = ", params1);
  oap::cuda::PrintMatrixInfo("ouput = ", output);
#endif

  debugAssertMsg (output_rows == params0_rows * params1_rows, "output_rows = %u params0_rows = %u params1_rows %u", output_rows, params0_rows, params1_rows);
  debugAssertMsg (output_columns == params0_columns * params1_columns, "output_columns = %u params0_columns = %u params1_columns = %u", output_columns, params0_columns, params1_columns);
}

inline void check_tensorProduct (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, generic::Dim32 dim,
                          const math::MatrixInfo& oinfo, const math::MatrixInfo& minfo0, const math::MatrixInfo& minfo1)
{
  check_Size (oinfo, output, dim[0]);
  check_Size (minfo0, params0, dim[1]);
  check_Size (minfo1, params1, dim[2]);

  const uintt output_columns = dim[0][0];
  const uintt output_rows = dim[0][1];

  const uintt params0_columns = dim[1][0];
  const uintt params0_rows = dim[1][1];

  const uintt params1_columns = dim[2][0];
  const uintt params1_rows = dim[2][1];

#ifdef CU_PROCEDURES_API_PRINT
  oap::cuda::PrintMatrixInfo("params0 = ", params0);
  oap::cuda::PrintMatrixInfo("params1 = ", params1);
  oap::cuda::PrintMatrixInfo("ouput = ", output);
#endif

  debugAssertMsg (output_rows == params0_rows * params1_rows, "output_rows = %u params0_rows = %u params1_rows = %u", output_rows, params0_rows, params1_rows);
  debugAssertMsg (output_columns == params0_columns * params1_columns, "output_columns = %u params0_columns = %u params1_columns = %u", output_columns, params0_columns, params1_columns);
}

template<typename GetColumns, typename GetRows>
void check_hadamardProduct (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt columns, uintt rows, BasicMatrixDimApi<GetColumns, GetRows>& matrixDimApi)
{
  const uintt output_columns = columns;
  const uintt output_rows = rows;

  const uintt params0_columns = matrixDimApi.getColumns(params0);
  const uintt params0_rows = matrixDimApi.getRows(params0);

  const uintt params1_columns = matrixDimApi.getColumns(params1);
  const uintt params1_rows = matrixDimApi.getRows(params1);

#ifdef CU_PROCEDURES_API_PRINT
  oap::cuda::PrintMatrixInfo("params0 = ", params0);
  oap::cuda::PrintMatrixInfo("params1 = ", params1);
  oap::cuda::PrintMatrixInfo("ouput = ", output);
#endif

  debugAssertMsg (output_rows == params0_rows && output_rows == params1_rows, "output_rows = %u params0_rows = %u params1_rows = %u", output_rows, params0_rows, params1_rows);
  debugAssertMsg (output_columns == params0_columns && output_columns == params1_columns, "output_columns = %u params0_columns = %u params1_columns = %u", output_columns, params0_columns, params1_columns);
}

template<typename GetColumns, typename GetRows>
void check_hadamardProductVec (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt columns, uintt rows, BasicMatrixDimApi<GetColumns, GetRows>& matrixDimApi)
{
  const uintt output_columns = columns;
  const uintt output_rows = rows;

  const uintt params0_columns = matrixDimApi.getColumns(params0);
  const uintt params0_rows = matrixDimApi.getRows(params0);

  const uintt params1_columns = matrixDimApi.getColumns(params1);
  const uintt params1_rows = matrixDimApi.getRows(params1);

#ifdef CU_PROCEDURES_API_PRINT
  oap::cuda::PrintMatrixInfo("params0 = ", params0);
  oap::cuda::PrintMatrixInfo("params1 = ", params1);
  oap::cuda::PrintMatrixInfo("ouput = ", output);
#endif

  debugAssertMsg(output_rows == params0_rows && output_rows == params1_rows, "output_rows = %u params0_rows = %u params1_rows = %u", output_rows, params0_rows, params1_rows);
  debugAssertMsg(1 == params1_columns, "params1_columns = %u", params1_columns);
  debugAssertMsg(output_columns == params0_columns, "output_columns = %u params0_columns = %u", output_columns, params0_columns);
}

}
}

#endif
