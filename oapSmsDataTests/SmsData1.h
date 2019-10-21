#ifndef OAP_SMSDATA1_H
#define OAP_SMSDATA1_H

namespace SmsData1
{

const unsigned int columns = 8;
const unsigned int rows = 8;

double smsmatrix[] =
{
  2, 0, 0, 0, 0, 0, 0, 0, 
  0, 1, 0, 0, 0, 0, 0, 0, 
  0, 0, 1, 0, 0, 0, 0, 0, 
  0, 0, 0, 1, 0, 0, 0, 0, 
  0, 0, 0, 0, 1, 0, 0, 0, 
  0, 0, 0, 0, 0, 1, 0, 0, 
  0, 0, 0, 0, 0, 0, 1, 0, 
  0, 0, 0, 0, 0, 0, 0, 1, 
};

double eigenvalues[] =
{ 2, 1, 1, 1, 1, 1, 1, 1};

double eigenvectors[] =
{
  1, 0, 0, 0, 0, 0, 0, 0, 
  0, 1, 0, 0, 0, 0, 0, 0, 
  0, 0, 1, 0, 0, 0, 0, 0, 
  0, 0, 0, 1, 0, 0, 0, 0, 
  0, 0, 0, 0, 1, 0, 0, 0, 
  0, 0, 0, 0, 0, 1, 0, 0, 
  0, 0, 0, 0, 0, 0, 1, 0, 
  0, 0, 0, 0, 0, 0, 0, 1, 
};

}

#endif
