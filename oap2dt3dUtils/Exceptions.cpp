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

#include "Exceptions.h"
#include <sstream>

namespace oap {
namespace exceptions {

template <class T>
std::string to_string(T i) {
  std::stringstream ss;
  std::string s;
  ss << i;
  s = ss.str();

  return s;
}

OutOfRange::OutOfRange(unsigned int _value, unsigned int _maxValue)
    : value(_value), maxValue(_maxValue) {}

std::string OutOfRange::getMessage() const {
  std::string msg;
  msg += "Value is out of range.";
  return msg;
}

FileNotExist::FileNotExist(const std::string& path) : m_path(path) {}

std::string FileNotExist::getMessage() const {
  std::string msg;
  msg = m_path + " doesnt exist.";
  return msg;
}

NotCorrectFormat::NotCorrectFormat(const std::string& path,
                                   const std::string& sufix)
    : m_path(path), m_sufix(sufix) {}

std::string NotCorrectFormat::getMessage() const {
  std::string msg;
  msg = m_path + " is not correct format. Correct format is ";
  msg += m_sufix;
  return msg;
}

NotIdenticalLengths::NotIdenticalLengths(size_t refLength, size_t length)
    : m_refLength(refLength), m_length(length) {}

std::string NotIdenticalLengths::getMessage() const {
  std::string lstr = to_string(m_length);
  std::string reflstr = to_string(m_refLength);
  std::string msg = "Not identical length: refLength = " + reflstr +
                    ", length = " + lstr + ".";
}

std::string NotInitialzed::getMessage() const {
  return "EigenCalculator has not initialzed yet.";
}
}
}
