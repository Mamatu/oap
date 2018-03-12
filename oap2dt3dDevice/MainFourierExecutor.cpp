#include "MainFourierExecutor.h"

namespace oap
{
  MainFourierExecutor::MainFourierExecutor ()
  {}

  void MainFourierExecutor::setOutcome(const std::shared_ptr<MainAPExecutor::Outcome>& outcome)
  {
    m_outcome = outcome;
  }

  void MainFourierExecutor::run()
  {}
}
