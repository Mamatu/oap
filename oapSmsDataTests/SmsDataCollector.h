#ifndef OAP_SMSDATACOLLECTOR_H
#define OAP_SMSDATACOLLECTOR_H

#include "SmsData1.h"
#include "SmsData2.h"
#include "SmsData3.h"
#include "SmsData4.h"

int const smsdata_columns[] =
{
  SmsData1::columns,
  SmsData2::columns,
  SmsData3::columns,
  SmsData4::columns
};

int const smsdata_rows[] =
{
  SmsData1::rows,
  SmsData2::rows,
  SmsData3::rows,
  SmsData4::rows
};

double* const smsdata_matrices[] =
{
  SmsData1::smsmatrix,
  SmsData2::smsmatrix,
  SmsData3::smsmatrix,
  SmsData4::smsmatrix
};

double* const smsdata_eigenvalues[] =
{
  SmsData1::eigenvalues,
  SmsData2::eigenvalues,
  SmsData3::eigenvalues,
  SmsData4::eigenvalues
};

double* const smsdata_eigenvectors[] =
{
  SmsData1::eigenvectors,
  SmsData2::eigenvectors,
  SmsData3::eigenvectors,
  SmsData4::eigenvectors
};

#endif
