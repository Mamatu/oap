#ifndef MAIN_AP_EXECUTOR_H
#define MAIN_AP_EXECUTOR_H

#include "ArnoldiProceduresImpl.h"
#include "ArnoldiUtils.h"
#include "oapDeviceMatrixPtr.h"
#include "DeviceDataLoader.h"

#include "Outcome.h"

#include <functional>
#include <vector>

namespace oap
{

class MainAPExecutor
{
  public:

    MainAPExecutor();
    ~MainAPExecutor();

    void setEigensType(ArnUtils::Type eigensType);

    void setInfo (const oap::DataLoader::Info& info);

    void setWantedCount(int wantedEigensCount);
    void setMaxIterationCounter(int maxIterationCounter);

    std::shared_ptr<oap::Outcome> run();

  private:
    ArnUtils::Type m_eigensType;
    oap::DataLoader::Info m_info;
    CuHArnoldiCallback::MultiplyFunc m_callback;

    int m_wantedEigensCount;
    int m_maxIterationCounter;

    oap::MatricesSharedPtr m_evectors;
    std::shared_ptr<oap::DeviceDataLoader> m_ddloader;

    oap::CuProceduresApi* m_cuProcedures;

    static void multiplyFunc(math::Matrix* m_w, math::Matrix* m_v, oap::CuProceduresApi& cuProceduresApi,
                             void* userData, CuHArnoldi::MultiplicationType mt);

    void checkValidity();
};
}

#endif
