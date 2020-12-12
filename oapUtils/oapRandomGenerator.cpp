#include "oapRandomGenerator.h"

namespace oap
{
namespace utils
{

RandomGenerator::RandomGenerator (floatt min, floatt max) :
  m_min(min), m_max(max), m_dre (m_rd()), m_dis (m_min, m_max)
{}

RandomGenerator::RandomGenerator (floatt min, floatt max, uintt seed) : RandomGenerator(min, max)
{
  m_dre.seed (seed);
}

RandomGenerator::RandomGenerator (const RandomGenerator& rg) : m_min (rg.m_min), m_max (rg.m_max), m_dre (rg.m_dre), m_dis (rg.m_dis)
{}

floatt RandomGenerator::operator()()
{
  floatt v = m_dis(m_dre);
  return v;
}
}
}
