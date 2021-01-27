#ifndef OAP_MATRIX_RANDOM_GENERATOR_H
#define OAP_MATRIX_RANDOM_GENERATOR_H

#include <functional>
#include "oapRandomGenerator.h"

namespace oap
{
namespace utils
{

class MatrixRandomGenerator final
{
  public:
    using Filter = std::function<floatt(uintt, uintt, floatt)>;

    MatrixRandomGenerator (floatt min, floatt max);
    MatrixRandomGenerator (floatt min, floatt max, uintt seed);
    MatrixRandomGenerator (RandomGenerator* rg);

    ~MatrixRandomGenerator ();

    void setFilter(Filter&& filter);
    void setFilter(const Filter& filter);

    floatt operator()(uintt c, uintt r);

  private:
    oap::utils::RandomGenerator* m_rg;
    bool m_deallocateRG;
    Filter m_filter;
};
}
}
#endif
