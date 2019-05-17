#ifndef OAP_MATRIX_CUDA_COMMON_H
#define OAP_MATRIX_CUDA_COMMON_H

#include "oapDeviceMatrixPtr.h"
#include "oapDeviceMatrixUPtr.h"
#include "oapCudaMatrixUtils.h"
#include "oapHostMatrixUtils.h"

inline std::ostream& operator<<(std::ostream& out, const math::Matrix* matrix)
{
  std::string str;
	if (oap::cuda::IsDeviceMatrix (matrix))
  {
    oap::cuda::ToString (str, matrix);
  }
  else
  {
    oap::host::ToString (str, matrix);
  }
  out << str;
  return out;
}

#endif
