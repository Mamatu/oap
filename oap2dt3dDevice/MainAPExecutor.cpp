#include "MainAPExecutor.h"

#include "PngFile.h"
#include "EigenCalculator.h"
#include "DeviceDataLoader.h"

#include "oapCudaMatrixUtils.h"
#include "oapHostMatrixUtils.h"

#include "oapHostMatrixPtr.h"
#include "oapDeviceMatrixPtr.h"

namespace oap
{

struct UserData
{
  oap::DeviceMatrixPtr value;
  MainAPExecutor* mainAPExec;
};

MainAPExecutor::MainAPExecutor() :
  m_ddloader(nullptr)
{
  oap::cuda::Context::Instance().create();
}

MainAPExecutor::~MainAPExecutor()
{
  oap::cuda::Context::Instance().destroy();
}

void MainAPExecutor::setEigensType(ArnUtils::Type eigensType)
{
  m_eigensType = eigensType;
}

void MainAPExecutor::setInfo (const oap::DataLoader::Info& info)
{
  m_info = info;
}

void MainAPExecutor::setWantedCount(int wantedEigensCount)
{
  m_wantedEigensCount = wantedEigensCount;
}

void MainAPExecutor::setMaxIterationCounter(int maxIterationCounter)
{
  m_maxIterationCounter = maxIterationCounter;
}

std::shared_ptr<MainAPExecutor::Outcome> MainAPExecutor::run()
{
  checkValidity();
  CuHArnoldiCallback cuharnoldi;
  
  m_ddloader.reset (oap::DeviceDataLoader::createDataLoader<oap::PngFile, oap::DeviceDataLoader>(m_info));

  oap::DeviceMatrixPtr dvalue = oap::cuda::NewDeviceReMatrix(1, 1); 

  UserData userData = {dvalue, this};

  cuharnoldi.setCallback(MainAPExecutor::multiplyFunc, &userData);

  std::vector<floatt> revalues;
  std::vector<floatt> errors;

  revalues.resize(m_wantedEigensCount);
  errors.resize(m_wantedEigensCount);

  math::MatrixInfo matrixInfo = m_ddloader->getMatrixInfo();

  m_evectors = oap::HostMatricesPtr(m_wantedEigensCount);

  for (uint idx = 0; idx < m_wantedEigensCount; ++idx)
  {
    m_evectors[idx] = oap::host::NewReMatrix(1, matrixInfo.m_matrixDim.rows);
  }

  oap::EigenCalculator eigenCalculator(&cuharnoldi);

  eigenCalculator.setDataLoader(m_ddloader.get());
  eigenCalculator.setEigensCount(m_wantedEigensCount);

  eigenCalculator.setEigenvaluesOutput(revalues.data());

  eigenCalculator.setEigenvectorsOutput(m_evectors, m_eigensType);

  eigenCalculator.calculate();

  for (uint idx = 0; idx < m_wantedEigensCount; ++idx)
  {
    errors[idx] = cuharnoldi.testOutcome(idx);
  }

  return std::make_shared<Outcome>(revalues, errors, m_evectors);
}

void MainAPExecutor::multiplyFunc(math::Matrix* m_w, math::Matrix* m_v, oap::CuProceduresApi& cuProceduresApi, void* userData, CuHArnoldi::MultiplicationType mt)
{
  if (mt == CuHArnoldi::TYPE_WV)
  {
    UserData* udObj = static_cast<UserData*>(userData);

    auto dataLoader = udObj->mainAPExec->m_ddloader;
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

void MainAPExecutor::checkValidity()
{
  if (!m_info.isValid())
  {
    throw std::runtime_error("Info is invalid.");
  }
}

}
