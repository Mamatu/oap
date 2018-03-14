#ifndef MAIN_AP_EXECUTOR_H
#define MAIN_AP_EXECUTOR_H

#include "ArnoldiProceduresImpl.h"
#include "ArnoldiUtils.h"
#include "oapDeviceMatrixPtr.h"
#include "DeviceDataLoader.h"

#include <functional>
#include <vector>

namespace oap
{

class MainAPExecutor
{
  public:
    class Outcome
    {
      public:
        Outcome(const std::vector<floatt>& revalues, const std::vector<floatt> errors, const oap::MatricesSharedPtr& evectors) :
          m_revalues(revalues),
          m_errors(errors),
          m_evectors(evectors)
        {}

        ~Outcome()
        {}

        floatt getValue(size_t index) const
        {
          return m_revalues[index];
        }

        floatt getError(size_t index) const
        {
          return m_errors[index];
        }

        const math::Matrix* getVector(size_t index)
        {
          return m_evectors[index];
        }

      private:
        std::vector<floatt> m_revalues;
        std::vector<floatt> m_errors;
        oap::MatricesSharedPtr m_evectors;
        std::function<void()> m_deleter;
    };

    MainAPExecutor();
    ~MainAPExecutor();

    void setEigensType(ArnUtils::Type eigensType);

    void setInfo (const oap::DataLoader::Info& info);

    void setWantedCount(int wantedEigensCount);
    void setMaxIterationCounter(int maxIterationCounter);

    std::shared_ptr<Outcome> run();

  private:
    ArnUtils::Type m_eigensType;
    oap::DataLoader::Info m_info;
    CuHArnoldiCallback::MultiplyFunc m_callback;

    int m_wantedEigensCount;
    int m_maxIterationCounter;

    oap::MatricesSharedPtr m_evectors;
    std::shared_ptr<oap::DeviceDataLoader> m_ddloader;

    CuProceduresApi* m_cuProcedures;

    static void multiplyFunc(math::Matrix* m_w, math::Matrix* m_v, CuProceduresApi& cuProceduresApi,
                             void* userData, CuHArnoldi::MultiplicationType mt);

    void checkValidity();
};
}

#endif
