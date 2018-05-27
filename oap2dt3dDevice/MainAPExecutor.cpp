#include "MainAPExecutor.h"

#include "IEigenCalculator.h"
#include "DeviceDataLoader.h"

#include "oapCudaMatrixUtils.h"
#include "oapHostMatrixUtils.h"

#include "oapHostMatrixPtr.h"
#include "oapDeviceMatrixPtr.h"

namespace oap
{

MainAPExecutor::MainAPExecutor() :
  MainAPExecutor (new CuHArnoldiCallback(), true)
{}

MainAPExecutor::~MainAPExecutor()
{
  destroy();
}

void MainAPExecutor::destroy()
{
  delete m_eigenCalc;
  m_eigenCalc = nullptr;
  if (m_bDeallocateArnoldi)
  {
    delete m_cuhArnoldi;
    m_cuhArnoldi = nullptr;
  }
}

struct UserData
{
  oap::DeviceMatrixPtr value;
  DeviceDataLoader* dataLoader;
};

void MainAPExecutor::multiplyCallback(math::Matrix* m_w, math::Matrix* m_v, oap::CuProceduresApi& cuProceduresApi, void* userData, CuHArnoldi::MultiplicationType mt)
{
  if (mt == CuHArnoldi::TYPE_WV)
  {
    UserData* udObj = static_cast<UserData*>(userData);
    auto dataLoader = udObj->dataLoader;
    auto dvalue = udObj->value;
    math::MatrixInfo matrixInfo = dataLoader->getMatrixInfo();
    for (uintt index = 0; index < matrixInfo.m_matrixDim.columns; ++index)
    {
      math::Matrix* vec = dataLoader->createDeviceRowVector(index);
      cuProceduresApi.dotProduct(dvalue, vec, m_v);
      oap::cuda::SetMatrix(m_w, dvalue, 0, index);
      oap::cuda::DeleteDeviceMatrix(vec);
    }
  }
}

std::shared_ptr<Outcome> MainAPExecutor::run(ArnUtils::Type type)
{
  oap::DeviceMatrixPtr dvalue = oap::cuda::NewDeviceReMatrix(1, 1); 

  UserData userData = {dvalue, m_eigenCalc->getDataLoader()};

  m_cuhArnoldi->setCallback (MainAPExecutor::multiplyCallback, &userData);

  std::vector<floatt> revalues;
  std::vector<floatt> errors;
  const size_t wanted = m_eigenCalc->getWantedEigensCount();

  revalues.resize(wanted);
  errors.resize(wanted);

  math::MatrixInfo matrixInfo = m_eigenCalc->getMatrixInfo();

  oap::HostMatricesPtr evectors = oap::HostMatricesPtr(wanted);

  for (uint idx = 0; idx < wanted; ++idx)
  {
    evectors[idx] = oap::host::NewReMatrix(1, matrixInfo.m_matrixDim.rows);
  }

  m_eigenCalc->setEigenvaluesOutput(revalues.data());

  m_eigenCalc->setEigenvectorsOutput(evectors, type);

  m_eigenCalc->calculate();

  for (uint idx = 0; idx < m_eigenCalc->getWantedEigensCount(); ++idx)
  {
    errors[idx] = m_cuhArnoldi->testOutcome(idx);
  }

  return std::make_shared<Outcome>(revalues, errors, evectors);
}

IEigenCalculator* MainAPExecutor::operator->() const
{
  return m_eigenCalc;
}

MainAPExecutor::MainAPExecutor(CuHArnoldiCallback* cuhArnoldi, bool deallocateArnoldi) :
  m_eigenCalc (new MainAPExecutor::EigenCalculator (cuhArnoldi)), m_cuhArnoldi (cuhArnoldi), m_bDeallocateArnoldi (deallocateArnoldi)
{}

}
