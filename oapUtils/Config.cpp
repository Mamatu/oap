/*
 * Copyright 2016, 2017 Marcin Matula
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
std::string Config::tmpPath;

const std::string& Config::getOapPath() {
  createPath(TO_STRING(OAP_PATH), oapPath);
  return oapPath;
}

const std::string& Config::getTmpPath() {
  createPath(TO_STRING(TMP_PATH), tmpPath);
  return tmpPath;
}

std::string Config::getPathInOap(const std::string& relativePath) {
  return getPathInRoot(getOapPath(), relativePath);
}

std::string Config::getPathInTmp(const std::string& relativePath) {
  return getPathInRoot(getTmpPath(), relativePath);
}

void Config::createPath(const std::string& root, std::string& storePath) {
  if (storePath.empty()) {
    storePath = root;
    if (storePath[storePath.size() - 1] != '/') {
      storePath = storePath + "/";
    }
  }
}

std::string Config::getPathInRoot(const std::string& root, const std::string& relativePath) {
  std::string output =  root + relativePath;
  if (output[output.size() - 1] != '/') {
      output += '/';
  }
  return output;
}
}
