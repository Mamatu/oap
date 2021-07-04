#include "MainAPExecutor.hpp"

#include "IEigenCalculator.hpp"
#include "DeviceImagesLoader.hpp"

#include "oapCudaMatrixUtils.hpp"
#include "oapHostComplexMatrixApi.hpp"

#include "oapHostComplexMatrixPtr.hpp"
#include "oapDeviceComplexMatrixPtr.hpp"

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
  oap::DeviceComplexMatrixPtr value;
  DeviceImagesLoader* dataLoader;
  oap::RecToSquareApi* rtsApi;
};

void MainAPExecutor::multiplyMatrixCallback (math::ComplexMatrix* m_w, math::ComplexMatrix* m_v, oap::CuProceduresApi& cuProceduresApi, void* userData, oap::VecMultiplicationType mt)
{
  debugFunc ();

  if (mt == oap::VecMultiplicationType::TYPE_WV)
  {
    UserData* udObj = static_cast<UserData*>(userData);
    auto dataLoader = udObj->dataLoader;
    auto dvalue = udObj->value;
    auto rts = udObj->rtsApi;

    math::ComplexMatrix* dmatrix = rts->createDeviceMatrix();
    cuProceduresApi.dotProduct(m_w, dmatrix, m_v);
    oap::cuda::DeleteDeviceMatrix(dmatrix);
  }
}

void MainAPExecutor::multiplySubMatrixCallback (math::ComplexMatrix* m_w, math::ComplexMatrix* m_v, oap::CuProceduresApi& cuProceduresApi, void* userData, oap::VecMultiplicationType mt)
{
  debugFunc ();

  if (mt == oap::VecMultiplicationType::TYPE_WV)
  {
    UserData* udObj = static_cast<UserData*>(userData);
    auto dataLoader = udObj->dataLoader;
    auto rts = udObj->rtsApi;

    math::MatrixInfo subInfo = rts->getMatrixInfo();
    math::MatrixInfo matrixInfo = rts->getMatrixInfo();

    while  (!sizeCondition (subInfo))
    {
      subInfo.m_matrixDim.rows = subInfo.m_matrixDim.rows - 1; // Todo: binary search
    }

    math::ComplexMatrix* submatrix = nullptr;

    uintt index = 0;
    while (index < matrixInfo.m_matrixDim.rows)
    {
      submatrix = rts->getDeviceSubMatrix (index, subInfo.m_matrixDim.rows, submatrix);
      uintt rows = oap::cuda::GetRows (submatrix);

      debugInfo ("rows = %u", rows);

      if (udObj->value == nullptr || rows != oap::cuda::GetRows (udObj->value))
      {
        udObj->value = oap::cuda::NewDeviceReMatrix (1, rows);
      }

      cuProceduresApi.dotProduct(udObj->value, submatrix, m_v);
      oap::cuda::SetMatrix(m_w, udObj->value, 0, index);

      index += rows;
    }

    oap::cuda::DeleteDeviceMatrix (submatrix);
  }
}

void MainAPExecutor::multiplyVecsCallback (math::ComplexMatrix* m_w, math::ComplexMatrix* m_v, oap::CuProceduresApi& cuProceduresApi, void* userData, oap::VecMultiplicationType mt)
{
  debugFunc ();

  if (mt == oap::VecMultiplicationType::TYPE_WV)
  {
    UserData* udObj = static_cast<UserData*>(userData);
    auto dataLoader = udObj->dataLoader;
    udObj->value = oap::cuda::NewDeviceReMatrix (1, 1);;
    auto rts = udObj->rtsApi;

    math::MatrixInfo matrixInfo = rts->getMatrixInfo();

    math::ComplexMatrix* vec = rts->createDeviceRowVector(0);
    cuProceduresApi.dotProduct(udObj->value, vec, m_v);
    oap::cuda::SetMatrix(m_w, udObj->value, 0, 0);

    for (uintt index = 1; index < matrixInfo.m_matrixDim.rows; ++index)
    {
      vec = rts->getDeviceRowVector(index, vec);
      cuProceduresApi.dotProduct(udObj->value, vec, m_v);
      oap::cuda::SetMatrix(m_w, udObj->value, 0, index);
    }
    oap::cuda::DeleteDeviceMatrix(vec);
  }
}

std::shared_ptr<Outcome> MainAPExecutor::run(ArnUtils::Type type)
{
  oap::RecToSquareApi rtsApi (m_eigenCalc->getImagesLoader()->createMatrix (), true);

  auto dataLoader = m_eigenCalc->getImagesLoader ();
  UserData userData = {nullptr, m_eigenCalc->getImagesLoader(), &rtsApi};

  auto minfo = rtsApi.getMatrixInfo ();
  if (sizeCondition (minfo))
  {
    m_cuhArnoldi->setCallback (MainAPExecutor::multiplyMatrixCallback, &userData);
  }
  else
  {
    m_cuhArnoldi->setCallback (MainAPExecutor::multiplySubMatrixCallback, &userData);
  }

  std::vector<floatt> revalues;
  std::vector<floatt> errors;
  const size_t wanted = m_eigenCalc->getWantedEigensCount();

  revalues.resize(wanted);
  errors.resize(wanted);

  math::MatrixInfo matrixInfo = m_eigenCalc->getMatrixInfo();

  oap::HostComplexMatricesPtr evectors (wanted);

  for (uint idx = 0; idx < wanted; ++idx)
  {
    evectors[idx] = oap::chost::NewReMatrix(1, matrixInfo.m_matrixDim.rows);
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
