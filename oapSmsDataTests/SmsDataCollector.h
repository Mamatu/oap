#ifndef OAP_SMSDATACOLLECTOR_H
#define OAP_SMSDATACOLLECTOR_H

#include "SmsData1.h"
#include "SmsData2.h"
#include "SmsData3.h"
#include "SmsData4.h"
#include "SmsData_64.h"
#include "SmsData1_Little.h"
#include "SmsData2_Little.h"
#include "SmsData3_Little.h"
#include "SmsData_Identity.h"

enum SmsTest
{
  Test_SmsData1 = 0,
  Test_SmsData2,
  Test_SmsData3,
  Test_SmsData4,
  Test_SmsData1_Little,
  Test_SmsData2_Little,
  Test_SmsData3_Little,
  Test_SmsData_Identity,
  Test_SmsData_64_1,
  Test_SmsData_64_2,
};

int const smsdata_columns[] =
{
  SmsData1::columns,
  SmsData2::columns,
  SmsData3::columns,
  SmsData4::columns,
  SmsData1_Little::columns,
  SmsData2_Little::columns,
  SmsData3_Little::columns,
  SmsData_Identity::columns,
  SmsData_64_1::columns,
  SmsData_64_2::columns,
};

int const smsdata_rows[] =
{
  SmsData1::rows,
  SmsData2::rows,
  SmsData3::rows,
  SmsData4::rows,
  SmsData1_Little::rows,
  SmsData2_Little::rows,
  SmsData3_Little::rows,
  SmsData_Identity::rows,
  SmsData_64_1::rows,
  SmsData_64_2::rows,
};

double* const smsdata_matrices[] =
{
  SmsData1::smsmatrix,
  SmsData2::smsmatrix,
  SmsData3::smsmatrix,
  SmsData4::smsmatrix,
  SmsData1_Little::smsmatrix,
  SmsData2_Little::smsmatrix,
  SmsData3_Little::smsmatrix,
  SmsData_Identity::smsmatrix,
  SmsData_64_1::smsmatrix,
  SmsData_64_2::smsmatrix,
};

double* const smsdata_eigenvalues[] =
{
  SmsData1::eigenvalues,
  SmsData2::eigenvalues,
  SmsData3::eigenvalues,
  SmsData4::eigenvalues,
  SmsData1_Little::eigenvalues,
  SmsData2_Little::eigenvalues,
  SmsData3_Little::eigenvalues,
  SmsData_Identity::eigenvalues,
  SmsData_64_1::eigenvalues,
  SmsData_64_2::eigenvalues,
};

double* const smsdata_eigenvectors[] =
{
  SmsData1::eigenvectors,
  SmsData2::eigenvectors,
  SmsData3::eigenvectors,
  SmsData4::eigenvectors,
  SmsData1_Little::eigenvectors,
  SmsData2_Little::eigenvectors,
  SmsData3_Little::eigenvectors,
  SmsData_Identity::eigenvectors,
  SmsData_64_1::eigenvectors,
  SmsData_64_2::eigenvectors,
};

#endif
