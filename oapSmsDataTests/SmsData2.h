#ifndef OAP_SMSDATA2_H
#define OAP_SMSDATA2_H

namespace SmsData2
{

const unsigned int columns = 8;
const unsigned int rows = 8;

double smsmatrix[] =
{
  1, 0, 0, 0, 0, 0, 0, 0, 
  0, 2, 0, 0, 0, 0, 0, 0, 
  0, 0, 3, 0, 0, 0, 0, 0, 
  0, 0, 0, 4, 0, 0, 0, 0, 
  0, 0, 0, 0, 5, 0, 0, 0, 
  0, 0, 0, 0, 0, 6, 0, 0, 
  0, 0, 0, 0, 0, 0, 7, 0, 
  0, 0, 0, 0, 0, 0, 0, 8, 
};

double eigenvalues[] =
{ 1, 2, 3, 4, 5, 6, 7, 8};

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
