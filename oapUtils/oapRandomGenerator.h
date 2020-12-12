#ifndef OAP_RANDOM_GENERATOR_H
#define OAP_RANDOM_GENERATOR_H

#include <functional>
#include <random>
#include "Math.h"

namespace oap
{
namespace utils
{

class RandomGenerator final
{
  public:
    RandomGenerator (floatt min, floatt max);
    RandomGenerator (floatt min, floatt max, uintt seed);

    RandomGenerator (const RandomGenerator& rg);

    floatt operator()();

  private:
    floatt m_min, m_max;
    std::random_device m_rd;
    std::default_random_engine m_dre;
    std::uniform_real_distribution<floatt> m_dis;
};
}
}
#endif
