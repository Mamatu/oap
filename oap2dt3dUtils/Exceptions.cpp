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

const char* OutOfRange::what() const throw() {
  return "Value is out of range.";
}

FileNotExist::FileNotExist(const std::string& path) : m_path(path) {
  m_msg = m_path + " doesnt exist.";
}

const char* FileNotExist::what() const throw() {
  return m_msg.c_str();
}

NotCorrectFormat::NotCorrectFormat(const std::string& path,
                                   const std::string& sufix)
    : m_path(path), m_sufix(sufix) 
{
  m_msg = m_path + " is not correct format. Correct format is ";
  m_msg += m_sufix;
}

const char* NotCorrectFormat::what() const throw() {
  return m_msg.c_str();
}

NotIdenticalLengths::NotIdenticalLengths(size_t refLength, size_t length)
    : m_refLength(refLength), m_length(length) 
{
  std::string lstr = to_string(m_length);
  std::string reflstr = to_string(m_refLength);
  std::string m_msg = "Not identical length: refLength = " + reflstr +
                    ", length = " + lstr + ".";
}

const char* NotIdenticalLengths::what() const throw() {
  return m_msg.c_str();
}

const char* NotInitialzed::what() const throw() {
  return "EigenCalculator has not initialzed yet.";
}
}
}
