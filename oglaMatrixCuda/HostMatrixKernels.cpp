#include "HostMatrixKernels.h"
#include <math.h>
#include "DeviceMatrixKernels.h"
#include "DeviceMatrixModules.h"

void aux_switchPointer(math::Matrix** a, math::Matrix** b) {
  math::Matrix* temp = *b;
  *b = *a;
  *a = temp;
}

inline CUresult host_prepareGMatrix(math::Matrix* A, uintt column, uintt row,
                                    math::Matrix* G, cuda::Kernel& kernel) {
  CUresult result = DEVICEKernel_SetIdentity(G, kernel);
  if (result != CUDA_SUCCESS) {
    //return result;
  }

  /*floatt reg = 0;
  floatt img = 0;
  floatt ref = 0;
  floatt imf = 0;
  if (A->reValues) {
    reg = A->reValues[column + row * A->columns];
    ref = A->reValues[column + column * A->columns];
  }
  if (A->imValues) {
    img = A->imValues[column + row * A->columns];
    imf = A->imValues[column + column * A->columns];
  }
  floatt r = sqrt(ref * ref + reg * reg + img * img + imf * imf);
  floatt lf = sqrt(ref * ref + imf * imf);
  floatt sign = 1;
  floatt isign = 0;
  if (fabs(ref) >= MATH_VALUE_LIMIT || fabs(imf) >= MATH_VALUE_LIMIT) {
    sign = ref / lf;
    isign = imf / lf;
  }
  floatt s = (sign * reg + img * isign) / r;
  floatt is = (isign * reg - img * sign) / r;
  floatt c = lf / r;
  floatt ic = 0;*/

  floatt s = 0;
  floatt is = 0;
  floatt c = 0;
  floatt ic = 0;
  uintt Acolumns = CudaUtils::GetColumns(A);
  bool isre = CudaUtils::GetReValues(A) != NULL;
  bool isim = CudaUtils::GetImValues(A) != NULL;
  if (isre) {
    s = CudaUtils::GetReValue(A, column + row * Acolumns);
    c = CudaUtils::GetReValue(A, column + column * Acolumns);
  }
  if (isim) {
    is = CudaUtils::GetImValue(A, column + row * Acolumns);
    ic = CudaUtils::GetImValue(A, column + column * Acolumns);
  }
  floatt r = sqrt(c * c + s * s + is * is + ic * ic);
  c = c / r;
  ic = ic / r;
  s = s / r;
  is = is / r;
  if (isre) {
    CudaUtils::SetReValue(G, column + row * Acolumns, -s);
    CudaUtils::SetReValue(G, column + (column) * Acolumns, c);
    CudaUtils::SetReValue(G, (row) + (row) * Acolumns, c);
    CudaUtils::SetReValue(G, (row) + (column) * Acolumns, s);
  }
  if (isim) {
    CudaUtils::SetImValue(G, column + row * Acolumns, -is);
    CudaUtils::SetImValue(G, column + (column) * Acolumns, ic);
    CudaUtils::SetImValue(G, (row) + (row) * Acolumns, ic);
    CudaUtils::SetImValue(G, (row) + (column) * Acolumns, is);
  }
  return CUDA_SUCCESS;
}

CUresult HOSTKernel_QRGR(math::Matrix* Q, math::Matrix* R, math::Matrix* A,
                         math::Matrix* Q1, math::Matrix* R1, math::Matrix* G,
                         math::Matrix* GT, cuda::Kernel& kernel) {
  math::Matrix* rQ = Q;
  math::Matrix* rR = R;
  cuda::CopyDeviceMatrixToDeviceMatrix(R1, A);
  uintt count = 0;
  uintt Acolumns = CudaUtils::GetColumns(A);
  uintt Arows = CudaUtils::GetRows(A);
  for (uintt fa = 0; fa < Acolumns; ++fa) {
    for (uintt fb = Arows - 1; fb > fa; --fb) {
      floatt v = CudaUtils::GetReValue(A, fa + fb * Acolumns);
      if ((-0.0001 < v && v < 0.0001) == false) {
        host_prepareGMatrix(R1, fa, fb, G, kernel);
        DEVICEKernel_DotProduct(R, G, R1, kernel);
        if (count == 0) {
          DEVICEKernel_Transpose(Q, G, kernel);
        } else {
          DEVICEKernel_Transpose(GT, G, kernel);
          DEVICEKernel_DotProduct(Q, Q1, GT, kernel);
        }
        ++count;
        aux_switchPointer(&R1, &R);
        aux_switchPointer(&Q1, &Q);
      }
    }
  }
  if (count & 1 == 1) {
    cuda::CopyDeviceMatrixToDeviceMatrix(rQ, Q1);
    cuda::CopyDeviceMatrixToDeviceMatrix(rR, R1);
  }
  return CUDA_SUCCESS;
}
