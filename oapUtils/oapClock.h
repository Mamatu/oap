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
#include <type_traits>

#include "Logger.h"

namespace oap
{

namespace clock
{

//using std_clock = std::chrono::steady_clock;
//using TimePoints = std::chrono::time_point<std_clock>;
//using TimeUnit = std::chrono::milliseconds;
using std_clock = std::chrono::high_resolution_clock;
using TimePoints = std::chrono::time_point<std_clock>;
using TimeUnit = std::chrono::microseconds;

struct CodeInfo
{
  const char* function = nullptr;
  const char* file = nullptr;
  int line = -1;
};

CodeInfo createCodeInfo (const char* function = nullptr, const char* file = nullptr, int line = -1);

void print (TimeUnit duration, const CodeInfo& ci);

TimePoints start ();

TimeUnit end (TimePoints start_time);

void end_print (TimePoints start_time, const char* file, const char* function, int line);

class Clock final
{
    TimePoints m_startTime;
    CodeInfo m_ci;

  public:
    Clock ();
    Clock (CodeInfo ci);

    ~Clock ();
};

}
}
#ifdef OAP_PERFORMANCE_CLOCK_ENABLE

#define OAP_CLOCK_INIT() oap::clock::CodeInfo oap_clock_ci = oap::clock::createCodeInfo(__func__,__FILE__,__LINE__); oap::clock::Clock oap_clock_clock (oap_clock_ci);

#define OAP_CLOCK_START(handler) auto handler = oap::clock::start ();

#define OAP_CLOCK_END(handler) oap::clock::end_print (handler, __func__, __FILE__, __LINE__);

#else

#define OAP_CLOCK_INIT()

#define OAP_CLOCK_START(val)

#define OAP_CLOCK_END(start)

#endif

#endif
