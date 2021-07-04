#ifndef OAP_MATRIX_CUDA_COMMON_H
#define OAP_MATRIX_CUDA_COMMON_H

#include "oapDeviceComplexMatrixPtr.hpp"
#include "oapDeviceComplexMatrixUPtr.hpp"
#include "oapCudaMatrixUtils.hpp"
#include "oapHostComplexMatrixApi.hpp"

inline std::ostream& operator<<(std::ostream& out, const math::ComplexMatrix* matrix)
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
