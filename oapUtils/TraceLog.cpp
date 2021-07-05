/*
 * Copyright 2016 - 2021 Marcin Matula
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

#include "TraceLog.hpp"

#include <stdio.h>
#include <cstring>
#include <string>

namespace trace {

  std::string g_buffer;
  bool g_init = false;

  void InitTraceBuffer(size_t size) {
    g_buffer.clear();
    g_buffer.reserve(size);
    g_init = true;
  }

  void Trace(const char* string) {
    if (g_init) { g_buffer += string; }
  }

  void Trace(const std::string& str) {
    if (g_init) { g_buffer += str; }
  }

  void Trace(const char* format, ...) {
    if (g_init == false) { return; }
    
    const size_t size = 256;
    char cstring[size];
    memset(cstring, 0, size * sizeof(char));
    
    va_list args;
    va_start (args, format);
    vsnprintf(cstring, size, format, args);
    va_end(args);
    
    std::string stdstring = std::string(cstring);
    g_buffer += stdstring;
  }
  
  void GetOutputString(std::string& output) {
    if (g_init) {
      output = g_buffer;
      g_buffer.clear();
      g_init = false;
    }
  }
}

