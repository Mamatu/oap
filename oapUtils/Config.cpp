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

#include "Config.hpp"
#include <cstdlib>

namespace oap {
namespace utils {
#define TO_STRING(s) STRING(s)
#define STRING(s) #s

std::string Config::oapPath;
std::string Config::tmpPath;

void endSlash (std::string& str)
{
  if (str[str.size() - 1] != '/')
  {
    str = str + "/";
  }
}

const std::string& Config::getOapPath()
{
  createPath(TO_STRING(OAP_PATH), oapPath);
  return oapPath;
}

const std::string& Config::getTmpPath()
{
  createPath(TO_STRING(TMP_PATH), tmpPath);
  return tmpPath;
}

std::string Config::getPathInOap(const std::string& relativePath)
{
  return getPathInRoot(getOapPath(), relativePath);
}

std::string Config::getPathInTmp(const std::string& relativePath)
{
  return getPathInRoot(getTmpPath(), relativePath);
}

std::string Config::getFileInOap(const std::string& relativePath)
{
  return getPathInRoot(getOapPath(), relativePath, false);
}

std::string Config::getFileInTmp(const std::string& relativePath)
{
  return getPathInRoot(getTmpPath(), relativePath, false);
}

void Config::createPath(const std::string& root, std::string& storePath)
{
  if (storePath.empty())
  {
    storePath = root;
    endSlash (storePath);
  }
}

std::string Config::getPathInRoot(const std::string& root, const std::string& relativePath, bool slashOnEnd)
{
  std::string output = root;
  endSlash (output);

  output += relativePath;

  if (slashOnEnd)
  {
    endSlash (output);
  }

  return output;
}

std::string Config::getVariable (const std::string& variable)
{
  const char* val = std::getenv(variable.c_str());
  if (val == nullptr)
  {
    return "";
  }
  return std::string(val);
}
}
}
