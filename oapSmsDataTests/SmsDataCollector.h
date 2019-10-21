#ifndef OAP_SMSDATACOLLECTOR_H
#define OAP_SMSDATACOLLECTOR_H

#include "SmsData1.h"
#include "SmsData2.h"

enum SmsTest
{
  Test_SmsData1 = 0,
  Test_SmsData2,
};

int const smsdata_columns[] =
{
  SmsData1::columns,
  SmsData2::columns,
};

int const smsdata_rows[] =
{
  SmsData1::rows,
  SmsData2::rows,
};

double* const smsdata_matrices[] =
{
  SmsData1::smsmatrix,
  SmsData2::smsmatrix,
};

double* const smsdata_eigenvalues[] =
{
  SmsData1::eigenvalues,
  SmsData2::eigenvalues,
};

double* const smsdata_eigenvectors[] =
{
  SmsData1::eigenvectors,
  SmsData2::eigenvectors,
};

#endif
