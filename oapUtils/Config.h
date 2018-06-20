/*
 * Copyright 2016 - 2018 Marcin Matula
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

#ifndef CONFIG_H
#define CONFIG_H

#include <string>

namespace utils {
class Config {
 public:
  static const std::string& getOapPath();

  static const std::string& getTmpPath();

  static std::string getPathInOap(const std::string& relativePath);

  static std::string getPathInTmp(const std::string& relativePath);

 private:
  static std::string oapPath;
  static std::string tmpPath;

  static void createPath(const std::string& root, std::string& storePath);
  static std::string getPathInRoot(const std::string& root, const std::string& relativePath);
};
}

#endif  // CONFIG_H
