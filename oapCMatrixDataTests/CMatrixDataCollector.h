#ifndef OAP_CMATRIXDATACOLLECTOR_H
#define OAP_CMATRIXDATACOLLECTOR_H

#include "CMatrixData1.h"
#include "CMatrixData2.h"
#include "CMatrixData3.h"

enum CMatrixTest
{
  Test_CMatrixData1 = 0,
  Test_CMatrixData2,
  Test_CMatrixData3,
};

int const data_columns[] =
{
  CMatrixData1::columns,
  CMatrixData2::columns,
  CMatrixData3::columns,
};

int const data_rows[] =
{
  CMatrixData1::rows,
  CMatrixData2::rows,
  CMatrixData3::rows,
};

double* const data_matrices[] =
{
  CMatrixData1::matrix,
  CMatrixData2::matrix,
  CMatrixData3::matrix,
};

double* const data_eigenvalues[] =
{
  CMatrixData1::eigenvalues,
  CMatrixData2::eigenvalues,
  CMatrixData3::eigenvalues,
};

#endif
