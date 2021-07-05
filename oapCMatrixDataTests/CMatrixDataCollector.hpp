#ifndef OAP_CMATRIXDATACOLLECTOR_H
#define OAP_CMATRIXDATACOLLECTOR_H

#include "CMatrixData1.hpp"
#include "CMatrixData2.hpp"
#include "CMatrixData3.hpp"
#include "CMatrixData4.hpp"
#include "CMatrixData5.hpp"

enum CMatrixTest
{
  Test_CMatrixData1 = 0,
  Test_CMatrixData2,
  Test_CMatrixData3,
  Test_CMatrixData4,
  Test_CMatrixData5,
};

int const data_columns[] =
{
  CMatrixData1::columns,
  CMatrixData2::columns,
  CMatrixData3::columns,
  CMatrixData4::columns,
  CMatrixData5::columns,
};

int const data_rows[] =
{
  CMatrixData1::rows,
  CMatrixData2::rows,
  CMatrixData3::rows,
  CMatrixData4::rows,
  CMatrixData5::rows,
};

double* const data_matrices[] =
{
  CMatrixData1::matrix,
  CMatrixData2::matrix,
  CMatrixData3::matrix,
  CMatrixData4::matrix,
  CMatrixData5::matrix,
};

double* const data_eigenvalues[] =
{
  CMatrixData1::eigenvalues,
  CMatrixData2::eigenvalues,
  CMatrixData3::eigenvalues,
  CMatrixData4::eigenvalues,
  CMatrixData5::eigenvalues,
};

#endif
