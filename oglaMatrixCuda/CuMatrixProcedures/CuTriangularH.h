#ifndef CUTRIANGULARH
#define CUTRIANGULARH

#include "CuCore.h"
#include "Matrix.h"
#include "CuMatrixProcedures/CuDotProductOptProcedures.h"
#include "CuMatrixProcedures/CuQRProcedures.h"
#include "CuMatrixProcedures/CuIsUpperTriangularProcedures.h"

__hostdevice__ void CUDA_HMtoUTM(
    math::Matrix* H, math::Matrix* Q, math::Matrix* R, math::Matrix* Qoutput,
    math::Matrix* temp1, math::Matrix* temp2, math::Matrix* temp3,
    math::Matrix* temp4, math::Matrix* temp5) {
  HOST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  bool status = false;
  CUDA_SetIdentityMatrix(Qoutput, tx, ty);
  status = CUDA_isUpperTriangular(H);
  uintt fa = 0;
  while (status == false) {
    CUDA_QRGR(Q, R, H, temp2, temp3, temp4, temp5);
    CUDA_dotProduct(H, R, Q, tx, ty);
    CUDA_dotProduct(temp1, Q, Qoutput, tx, ty);
    CUDA_switchPointer(&temp1, &Qoutput);
    status = CUDA_isUpperTriangular(H);
    // threads_sync();
    ++fa;
  }
  // TODO: optymalization
  if (fa % 2 == 0) {
    CUDA_copyMatrix(Q, Qoutput);
  } else {
    CUDA_copyMatrix(Q, temp1);
  }
  // cuda_debug_function();
}

#endif  // CUTRIANGULARH
