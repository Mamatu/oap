#ifndef SMSDATA_IDENTITY_H
#define SMSDATA_IDENTITY_H

namespace SmsData_Identity {

const unsigned int columns = 4;
const unsigned int rows = 4;

double smsmatrix[] =
{1, 0, 0, 0,
 0, 1, 0 ,0,
 0, 0, 1, 0,
 0, 0, 0, 1};

double eigenvalues[] =
{ 1,  1,  1,  1};

double eigenvectors[] =
{ 1,  0,  0,  0,
  0,  1,  0,  0,
  0,  0,  1,  0,
  0,  0,  0,  1};

}

#endif
