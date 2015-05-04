#ifndef CUDA_ARNOLDIMETHODIMPL_H
#define CUDA_ARNOLDIMETHODIMPL_H

#include <vector>
#include "IArnoldiMethod.h"
#include "Math.h"
#include "ArnoldiProcedures.h"
#include "DeviceMatrixModules.h"

/**
 * Implementation of implicity reverse arnoldi method.
 *
 */
namespace math {

class ArnoldiMethodGpu : public math::IArnoldiMethod {
  CuHArnoldiDefault cuHArnoldi;
  uintt m_k;
  floatt m_rho;
  uintt m_wantedCount;
  math::Matrix outputs;
  math::Matrix* m_dmatrix;
  ArnUtils::MatrixInfo m_matrixInfo;
  void SetupMatrixInfo();

 public:
  ArnoldiMethodGpu();
  ArnoldiMethodGpu(MatrixModule* matrixModule);
  virtual ~ArnoldiMethodGpu();
  void setHSize(uintt k);
  void setRho(floatt rho);
  void execute();
};
}

#endif /* ARNOLDIMETHOD_H */
