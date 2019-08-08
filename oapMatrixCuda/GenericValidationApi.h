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

#ifndef OAP_GENERIC_VALIDATION_API_H
#define OAP_GENERIC_VALIDATION_API_H

#include "GenericCoreApi.h"

#include "MatrixInfo.h"

namespace oap
{
namespace generic
{

template<typename BasicMatrixApi>
void check_dotProduct (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt columns, uintt rows, BasicMatrixApi& bmApi)
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

  debugAssertMsg(params0_columns == params1_rows, "params0_columns = %u params1_rows = %u", params0_columns, params1_rows);
  debugAssertMsg(output_columns == params1_columns, "output_columns = %u params1_columns = %u", output_columns, params1_columns);
  debugAssertMsg(output_rows == params0_rows, "output_rows = %u params0_rows = %u", output_rows, params0_rows);
}

template<typename Dim>
void check_Size (const math::MatrixInfo& minfo, math::Matrix* matrix, const Dim& dim)
{
  const uintt columns = minfo.columns ();
  const uintt rows = minfo.rows ();

  debugAssert (dim[0] <= columns);
  debugAssert (dim[1] <= rows);
}

inline void check_dotProduct (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt dims[3][2],
                              const math::MatrixInfo& oinfo, const math::MatrixInfo& minfo1, const math::MatrixInfo& minfo2)
{
  check_Size (oinfo, output, dims[0]);
  check_Size (minfo1, params0, dims[1]);
  check_Size (minfo2, params1, dims[2]);

  const uintt output_columns = dims[0][0];//matrixDimApi.getColumns(params0);
  const uintt output_rows = dims[0][1]; //matrixDimApi.getRows(params0);

  const uintt params0_columns = dims[1][0];//matrixDimApi.getColumns(params0);
  const uintt params0_rows = dims[1][1]; //matrixDimApi.getRows(params0);

  const uintt params1_columns = dims[2][0];//matrixDimApi.getColumns(params1);
  const uintt params1_rows = dims[2][1];//matrixDimApi.getRows(params1);

#ifdef CU_PROCEDURES_API_PRINT
  oap::cuda::PrintMatrixInfo("params0 = ", params0);
  oap::cuda::PrintMatrixInfo("params1 = ", params1);
  oap::cuda::PrintMatrixInfo("ouput = ", output);
#endif

  debugAssertMsg (params0_columns == params1_rows, "params0_columns = %u params1_rows = %u", params0_columns, params1_rows);
  debugAssertMsg (output_columns == params1_columns, "output_columns = %u params1_columns = %u", output_columns, params1_columns);
  debugAssertMsg (output_rows == params0_rows, "output_rows = %u params0_rows = %u", output_rows, params0_rows);
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

inline void check_tensorProduct (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt dims[3][2],
                          const math::MatrixInfo& oinfo, const math::MatrixInfo& minfo0, const math::MatrixInfo& minfo1)
{
  check_Size (oinfo, output, dims[0]);
  check_Size (minfo0, params0, dims[1]);
  check_Size (minfo1, params1, dims[2]);

  const uintt output_columns = dims[0][0];
  const uintt output_rows = dims[0][1];

  const uintt params0_columns = dims[1][0];
  const uintt params0_rows = dims[1][1];

  const uintt params1_columns = dims[2][0];
  const uintt params1_rows = dims[2][1];

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
