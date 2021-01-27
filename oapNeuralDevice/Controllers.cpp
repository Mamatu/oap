#include "Controllers.h"

SE_ID_Controller::SE_ID_Controller (floatt limit, uintt dataSetSize, const std::function<void(floatt, uintt, floatt)>& callback) :
  m_limit(limit), m_dataSetSize(dataSetSize), m_sqes(0), m_sc(true), m_callback(callback)
{}

SE_ID_Controller::~SE_ID_Controller()
{}

bool SE_ID_Controller::shouldCalculateError (uintt step)
{
  m_step = step;
  return true;
}

void SE_ID_Controller::setError (floatt sqe, oap::ErrorType etype)
{
  m_sqes += sqe;
  if (m_step % m_dataSetSize == 0)
  {
    m_sqes = m_sqes / static_cast<floatt>(m_dataSetSize);
    m_sc = m_sqes > m_limit;
    debug("square error = %f limit = %f step = %lu", m_sqes, m_limit, m_step);

    if (m_callback)
    {
      m_callback (m_sqes, m_step, m_limit);
    }

    m_sqes = 0;
  }
}

bool SE_ID_Controller::shouldContinue()
{
  return m_sc;
}

SE_CD_Controller::SE_CD_Controller (floatt limit, uintt dataSetSize, const std::function<void(floatt, uintt, floatt)>& callback) :
  m_limit(limit), m_dataSetSize(dataSetSize), m_sqe(0), m_sc(true), m_callback(callback)
{}

SE_CD_Controller::~SE_CD_Controller()
{}

bool SE_CD_Controller::shouldCalculateError (uintt step)
{
  m_step = step;
  return true;
}

void SE_CD_Controller::setError (floatt sqe, oap::ErrorType etype)
{
  m_sqe += sqe;
  m_sqes.push (sqe);
  if (m_step >= m_dataSetSize)
  {
    floatt e = m_sqes.front ();
    m_sqes.pop ();
    m_sqe -= e;

    floatt csqe = m_sqe / static_cast<floatt>(m_dataSetSize);
    m_sc = csqe > m_limit;
    debug("square error = %f limit = %f step = %lu", csqe, m_limit, m_step);

    if (m_callback)
    {
      m_callback (csqe, m_step, m_limit);
    }
  }
}

bool SE_CD_Controller::shouldContinue()
{
  return m_sc;
}

DerivativeController::DerivativeController (uintt activationLimit): m_activationLimit(activationLimit)
{}

DerivativeController::~DerivativeController()
{}

bool DerivativeController::shouldCalculateError (uintt step)
{
  return true;
}

void DerivativeController::setError (floatt sqe, oap::ErrorType etype)
{}

bool DerivativeController::shouldContinue()
{
  return true;
}

StepController::StepController (uintt sstep) : m_sstep (sstep)
{}

StepController::~StepController () {}

bool StepController::shouldCalculateError (uintt step)
{
  m_step = step;
  return false;
}

void StepController::setError (floatt sqe, oap::ErrorType etype) {}

bool StepController::shouldContinue()
{
  return (m_step < m_sstep);
}
