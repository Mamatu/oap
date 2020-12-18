#include "oapRandomGenerator.h"

namespace oap
{
namespace utils
{

RandomGenerator::RandomGenerator (floatt min, floatt max) :
  m_dre (m_rd())
{
  setRange (min, max);
}

RandomGenerator::RandomGenerator (floatt min, floatt max, uintt seed) : RandomGenerator(min, max)
{
  m_dre.seed (seed);
}

std::pair<floatt, floatt> RandomGenerator::setRange (floatt min, floatt max)
{
  std::pair<floatt, floatt> previousRange = m_range;

  m_range = std::make_pair (min, max);
  m_dis = std::uniform_real_distribution<floatt>(min, max);

  return previousRange;
}

floatt RandomGenerator::operator()()
{
  floatt v = m_dis(m_dre);
  return v;
}

floatt RandomGenerator::operator()(floatt min, floatt max)
{
  auto prange = setRange (min, max);
  floatt v = this->operator()();

  setRange (prange.first, prange.second);
  return v;
}

}
}
