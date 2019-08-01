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

#include "GenericProceduresApi.h"

namespace oap
{
namespace generic
{

template<typename GetColumns, typename GetRows>
void check_dotProduct (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt columns, uintt rows, BasicMatrixDimApi<GetColumns, GetRows>& matrixDimApi)
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

  debugAssertMsg(params0_columns == params1_rows, "params0_columns = %u params1_rows = %u", params0_columns, params1_rows);
  debugAssertMsg(output_columns == params1_columns, "output_columns = %u params1_columns = %u", output_columns, params1_columns);
  debugAssertMsg(output_rows == params0_rows, "output_rows = %u params0_rows = %u", output_rows, params0_rows);
}

template<typename GetColumns, typename GetRows>
void check_tensorProduct (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt columns, uintt rows, BasicMatrixDimApi<GetColumns, GetRows>& matrixDimApi)
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

  std::stringstream stream1, stream2;

  stream1 << __func__ << " output_rows = " << output_rows << ", params0_rows = " << params0_rows << ", params1_rows = " << params1_rows;
  throwExceptionMsg(output_rows == params0_rows * params1_rows, stream1);

  stream2 << __func__ << " output_columns = " << output_columns << ", params0_columns = " << params0_columns << ", params1_columns = " << params1_columns;
  throwExceptionMsg(output_columns == params0_columns * params1_columns, stream2);

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

  std::stringstream stream1, stream2;

  stream1 << __func__ << " output_rows = " << output_rows << ", params0_rows = " << params0_rows << ", params1_rows = " << params1_rows;
  throwExceptionMsg(output_rows == params0_rows && output_rows == params1_rows, stream1);

  stream2 << __func__ << " output_columns = " << output_columns << ", params0_columns = " << params0_columns << ", params1_columns = " << params1_columns;
  throwExceptionMsg(output_columns == params0_columns && output_columns == params1_columns, stream2);
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

  std::stringstream stream1, stream2, stream3;

  stream1 << __func__ << " output_rows = " << output_rows << ", params0_rows = " << params0_rows << ", params1_rows = " << params1_rows;
  throwExceptionMsg(output_rows == params0_rows && output_rows == params1_rows, stream1);

  stream2 << __func__ << " params1_columns = " << params1_columns;
  throwExceptionMsg(1 == params1_columns, stream1);

  stream3 << __func__ <<  "output_columns = " << output_columns << ", params0_columns = " << params0_columns;
  throwExceptionMsg(output_columns == params0_columns, stream2);
}

}
}

#endif
