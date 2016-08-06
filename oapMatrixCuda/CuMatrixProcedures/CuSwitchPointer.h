#ifndef CUSWITCHPOINTER
#define CUSWITCHPOINTER

#include "CuCore.h"

__hostdeviceinline__ void CUDA_switchPointer(math::Matrix** a, math::Matrix** b) {
  HOST_INIT();
  math::Matrix* temp = *b;
  *b = *a;
  *a = temp;
  threads_sync();
}

#endif // CUSWITCHPOINTER
