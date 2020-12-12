#include "oapMatrixRandomGenerator.h"

namespace oap
{
namespace utils
{

MatrixRandomGenerator::MatrixRandomGenerator (floatt min, floatt max) :
  m_rg(new oap::utils::RandomGenerator(min, max)), m_deallocateRG(true)
{}

MatrixRandomGenerator::MatrixRandomGenerator (floatt min, floatt max, uintt seed) :
  m_rg(new oap::utils::RandomGenerator(min, max, seed)), m_deallocateRG(true)
{}

MatrixRandomGenerator::MatrixRandomGenerator (RandomGenerator* rg) : m_rg (rg), m_deallocateRG (false)
{}

MatrixRandomGenerator::~MatrixRandomGenerator ()
{
  if (m_deallocateRG)
  {
    delete m_rg;
  }
}

void MatrixRandomGenerator::setFilter(Filter&& filter)
{
  m_filter = std::move (filter);
}

void MatrixRandomGenerator::setFilter(const Filter& filter)
{
  m_filter = filter;
}

floatt MatrixRandomGenerator::operator()(uintt c, uintt r)
{
  floatt v = m_rg->operator()();

  if (m_filter)
  {
    return m_filter (v, c, r);
  }

  return v;
}

}
}
