#ifndef MAIN_FOURIER_EXECUTOR_H
#define MAIN_FOURIER_EXECUTOR_H

#include "Outcome.hpp"

namespace oap
{

class MainFourierExecutor
{
  public:
    MainFourierExecutor ();   

    void setOutcome(const std::shared_ptr<oap::Outcome>& outcome);
    void run();

  private:
    std::shared_ptr<oap::Outcome> m_outcome;
};

}

#endif

