#include "Controllers.h"

SquareErrorLimitController::SquareErrorLimitController (floatt limit, size_t dataSetSize): m_limit(limit), m_dataSetSize(dataSetSize), m_sqes(0), m_sc(true)
{}

SquareErrorLimitController::~SquareErrorLimitController()
{}

bool SquareErrorLimitController::shouldCalculateError (size_t step)
{
  m_step = step;
  return true;
}

void SquareErrorLimitController::setSquareError (floatt sqe)
{
  m_sqes += sqe;
  if (m_step % m_dataSetSize == 0)
  {
    m_sqes = m_sqes / static_cast<floatt>(m_dataSetSize);
    m_sc = m_sqes > m_limit;
    debug("square error = %f limit = %f step = %lu", m_sqes, m_limit, m_step);
    m_sqes = 0;
  }
}

bool SquareErrorLimitController::shouldContinue()
{
  return m_sc;
}

DerivativeController::DerivativeController (size_t activationLimit): m_activationLimit(activationLimit)
{}

DerivativeController::~DerivativeController()
{}

bool DerivativeController::shouldCalculateError (size_t step)
{
  return true;
}

void DerivativeController::setSquareError (floatt sqe)
{
}

bool DerivativeController::shouldContinue()
{
  return true;
}

