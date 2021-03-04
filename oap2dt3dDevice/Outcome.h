#ifndef OAP_OUTCOME_H
#define OAP_OUTCOME_H

#include "Math.h"
#include "Matrix.h"
#include "oapMatrixSPtr.h"

#include <vector>

namespace oap
{

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

    const math::ComplexMatrix* getVector(size_t index)
    {
      return m_evectors[index];
    }

  private:
    std::vector<floatt> m_revalues;
    std::vector<floatt> m_errors;
    oap::MatricesSharedPtr m_evectors;
    std::function<void()> m_deleter;
};

}

#endif
