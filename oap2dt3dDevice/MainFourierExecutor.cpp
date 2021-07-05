#include "MainFourierExecutor.hpp"

namespace oap
{
  MainFourierExecutor::MainFourierExecutor ()
  {}

  void MainFourierExecutor::setOutcome(const std::shared_ptr<oap::Outcome>& outcome)
  {
    m_outcome = outcome;
  }

  void MainFourierExecutor::run()
  {}
}
