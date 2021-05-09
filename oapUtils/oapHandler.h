#ifndef OAP_HANDLER_H
#define OAP_HANDLER_H

#include "Math.h"

namespace oap
{
template<uintt hash>
class Handler
{
  private:
    uintt m_v = true;
    bool m_isvalid = false;
    uintt m_hash = hash;
  public:
    Handler (uintt v, bool isvalid = true);
    Handler (int v, bool isvalid = true);
    Handler (unsigned long v, bool isvalid = true);
    Handler (bool isvalid);

    Handler (const Handler& orig);

    void operator=(uintt v);

    operator uintt() const
    {
      return m_v;
    }
};

template<uintt hash>
Handler<hash>::Handler (uintt v, bool isvalid) : m_v(v), m_isvalid(isvalid)
{}

template<uintt hash>
Handler<hash>::Handler (int v, bool isvalid) : m_v(static_cast<uintt>(v)), m_isvalid(isvalid)
{
  if (v < 0) {
    m_isvalid = false;
  }
}

template<uintt hash>
Handler<hash>::Handler (unsigned long v, bool isvalid) :  m_v(static_cast<uintt>(v)), m_isvalid(isvalid)
{}

template<uintt hash>
Handler<hash>::Handler (bool isvalid) : m_v(0), m_isvalid(isvalid)
{}

template<uintt hash>
Handler<hash>::Handler (const Handler<hash>& orig) : m_v(orig.m_v), m_isvalid(orig.m_isvalid)
{}

template<uintt hash>
void Handler<hash>::operator=(uintt v) {
  this->m_v = v;
}
}



#endif
