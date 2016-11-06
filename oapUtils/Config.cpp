/*
 * Copyright 2016 Marcin Matula
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

#include "Config.h"

namespace utils {
#define TO_STRING(s) STRING(s)
#define STRING(s) #s

std::string Config::oapPath;

const std::string& Config::getOapPath() {
  if (oapPath.empty()) {
    oapPath = TO_STRING(OAP_PATH);
  }
  return oapPath;
}

std::string Config::getPathInOap(const char* relativePath) {
  return getOapPath() + relativePath;
}
}
