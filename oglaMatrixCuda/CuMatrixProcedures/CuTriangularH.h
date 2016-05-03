#ifndef CUTRIANGULARH
#define CUTRIANGULARH

#include "CuCore.h"
#include "Matrix.h"

__hostdevice__ void CUDA_CalculateTriangularH(
    math::Matrix* H, math::Matrix* Q, math::Matrix* R, math::Matrix* temp,
    math::Matrix* temp1, math::Matrix* temp2, math::Matrix* temp3,
    math::Matrix* temp4, math::Matrix* temp5) {
  HOST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  bool status = false;
  CUDA_SetIdentityMatrix(temp, tx, ty);
  status = CUDA_isUpperTriangular(H);
  uintt fb = 0;
  for (; status == false && fb < 600; ++fb) {
    CUDA_QRGR(Q, R, H, temp2, temp3, temp4, temp5);
    CUDA_dotProduct(H, R, Q, tx, ty);
    CUDA_dotProduct(temp1, Q, temp, tx, ty);
    CUDA_switchPointer(&temp1, &temp);
    status = CUDA_isUpperTriangular(H);
  }
  // TODO: optymalization
  if (fb & 1 == 0) {
    CUDA_copyMatrix(Q, temp);
  } else {
    CUDA_copyMatrix(Q, temp1);
  }
  // cuda_debug_function();
}

#endif  // CUTRIANGULARH
