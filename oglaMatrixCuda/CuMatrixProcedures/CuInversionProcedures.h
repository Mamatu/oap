#ifndef CUINVERSIONPROCEDURES
#define CUINVERSIONPROCEDURES

#include "CuIdentityProcedures.h"
#include "CuAdditionProcedures.h"
#include "CuSubstractionProcedures.h"
#include "CuDotProductProcedures.h"
#include "CuSwitchPointer.h"

__hostdevice__ void CUDA_invertMatrix(math::Matrix* AI, math::Matrix* A,
                                      math::Matrix* aux1, math::Matrix* aux2,
                                      math::Matrix* aux3) {
  HOST_INIT();
  CUDA_SetIdentityMatrix(aux2, threadIdx.x, threadIdx.y);
  CUDA_SetIdentityMatrix(aux3, threadIdx.x, threadIdx.y);
  CUDA_substractMatrices(aux3, aux2, A, threadIdx.x, threadIdx.y);

  for (uintt fa = 0; fa < 15; ++fa) {
    CUDA_SetIdentityMatrix(aux2, threadIdx.x, threadIdx.y);
    for (uintt fb = 0; fb < fa; ++fb) {
      CUDA_dotProduct(aux1, aux2, aux3, threadIdx.x, threadIdx.y);
      CUDA_switchPointer(&aux1, &aux2);
    }
    if (fa % 2 != 0) {
      CUDA_switchPointer(&aux1, &aux2);
    }
    CUDA_addMatrix(AI, AI, aux1);
  }
}

#endif  // CUINVERSEMATRIX
