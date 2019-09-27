#ifndef SMSDATA3_LITTLE_H
#define SMSDATA3_LITTLE_H

namespace SmsData3_Little {

const unsigned int columns = 4;
const unsigned int rows = 4;

double smsmatrix[] =
{2, 0, 0, 0,
 0, 1, 0 ,0,
 0, 0, 1, 0,
 0, 0, 0, 1};

double eigenvalues[] =
{ 2,  1,  1,  1};

double eigenvectors[] =
{ 1,  0,  0,  0,
  0,  1,  0,  0,
  0,  0,  1,  0,
  0,  0,  0,  1};

}

#endif
