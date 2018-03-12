#ifndef MAIN_FOURIER_EXECUTOR_H
#define MAIN_FOURIER_EXECUTOR_H

#include "MainAPExecutor.h"

namespace oap
{

class MainFourierExecutor
{
  public:
    MainFourierExecutor ();   

    void setOutcome(const std::shared_ptr<MainAPExecutor::Outcome>& outcome);
    void run();

  private:
    std::shared_ptr<MainAPExecutor::Outcome> m_outcome;
};

}

#endif

