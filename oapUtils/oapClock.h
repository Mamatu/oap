/*
 * Copyright 2016 - 2019 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef OAP_CLOCK_H
#define OAP_CLOCK_H

#include <chrono>
#include "Logger.h"

namespace oap
{

namespace clock
{

using TimePoints = std::chrono::time_point<std::chrono::steady_clock>;
using TimeUnit = std::chrono::milliseconds;

struct CodeInfo
{
  const char* function = nullptr;
  const char* file = nullptr;
  int line = -1;
};

inline CodeInfo createCodeInfo (const char* function = nullptr, const char* file = nullptr, int line = -1)
{
  CodeInfo ci;
  ci.function = function;
  ci.file = file;
  ci.line = line;
  return ci;
}

inline bool isValid (const CodeInfo& ci)
{
  return ci.function != nullptr && ci.file != nullptr && ci.line > 0;
}

inline void print (TimeUnit duration, const CodeInfo& ci)
{
  if (isValid (ci))
  {
    logInfo ("%s %s %d Duration = %lu ms", ci.function, ci.file, ci.line, duration.count());
  }
  else
  {
    logInfo ("Duration = %lu ms", duration.count());
  }
}

inline TimePoints start ()
{
  return std::chrono::steady_clock::now();
}

inline TimeUnit end (TimePoints start_time)
{
  return std::chrono::duration_cast<TimeUnit> (std::chrono::steady_clock::now() - start_time);
}

inline void end_print (TimePoints start_time, const char* file, const char* function, int line)
{
  auto duration = end (start_time);
  CodeInfo ci = createCodeInfo (file, function, line);
  print (duration, ci);
}

class Clock final
{
    TimePoints m_startTime;
    CodeInfo m_ci;

  public:
    inline Clock () : m_startTime (oap::clock::start())
    {}

    inline Clock (CodeInfo ci) : m_startTime (oap::clock::start())
    {
      m_ci = ci;
    }

    inline ~Clock ()
    {
      TimeUnit duration = oap::clock::end (m_startTime);
      print (duration, m_ci);
    }
};

}
}
#ifdef OAP_PERFORMANCE_CLOCK_ENABLE

#define OAP_CLOCK_INIT() oap::clock::CodeInfo oap_clock_ci = oap::clock::createCodeInfo(__FUNCTION__,__FILE__,__LINE__); oap::clock::Clock oap_clock_clock (oap_clock_ci);

#define OAP_CLOCK_START(val) auto val = oap::clock::start ();

#define OAP_CLOCK_END(start) oap::clock::end_print (start, __func__, __FILE__, __LINE__);

#else

#define OAP_CLOCK_INIT()

#define OAP_CLOCK_START(val)

#define OAP_CLOCK_END(start)

#endif

#endif
