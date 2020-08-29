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

#include "oapClock.h"

namespace oap
{

namespace clock
{

class MillisecondsSuffix
{
  private:
    MillisecondsSuffix() {}

    std::string getStr(void*) const { return "ms"; }

    friend std::string getExtraStr (unsigned long int);
};

class MicrosecondsSuffix
{
  private:
    MicrosecondsSuffix() {}

    std::string getStr(unsigned long int dt) const
    {
      using uli = unsigned long int;
      const uli microseconds = dt;
      uli milliseconds = microseconds / 1000;
      std::string str = "us (";
      str += std::to_string (milliseconds);
      str += " ms)";
      return str;
    }

    friend std::string getExtraStr (unsigned long int);
};

CodeInfo createCodeInfo (const char* function, const char* file, int line)
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

inline std::string getExtraStr (unsigned long int dt)
{
  constexpr bool isMilli = std::is_same<typename TimeUnit::period, std::milli>::value;
  constexpr bool isMicro = std::is_same<typename TimeUnit::period, std::micro>::value;

  static_assert ((isMilli && !isMicro) || (!isMilli && isMicro), "Supported durations are: milliseconds and microseconds");

  std::conditional<isMilli, MillisecondsSuffix, MicrosecondsSuffix>::type suffix;
  return suffix.getStr (dt);
}

void print (TimeUnit duration, const CodeInfo& ci)
{
  unsigned long int dt = duration.count ();
  if (isValid (ci))
  {
    logInfo ("%s %s %d Duration = %lu %s", ci.function, ci.file, ci.line, duration.count(), getExtraStr (dt).c_str());
  }
  else
  {
    logInfo ("Duration = %lu %s", duration.count(), getExtraStr (dt).c_str());
  }
}

TimePoints start ()
{
  return std_clock::now();
}

TimeUnit end (TimePoints start_time)
{
  return std::chrono::duration_cast<TimeUnit> (std_clock::now() - start_time);
}

void end_print (TimePoints start_time, const char* file, const char* function, int line)
{
  auto duration = end (start_time);
  CodeInfo ci = createCodeInfo (file, function, line);
  print (duration, ci);
}

Clock::Clock () : m_startTime (oap::clock::start())
{}

Clock::Clock (CodeInfo ci) : m_startTime (oap::clock::start())
{
  m_ci = ci;
}

Clock::~Clock ()
{
  TimeUnit duration = oap::clock::end (m_startTime);
  print (duration, m_ci);
}

}
}
