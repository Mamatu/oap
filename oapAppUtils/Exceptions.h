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

#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <string>
#include <exception>

namespace oap {
namespace exceptions {

class OutOfRange : public std::exception {
 public:
  OutOfRange(unsigned int _value, unsigned int _maxValue);

  virtual const char* what() const throw();

 private:
  unsigned int value;
  unsigned int maxValue;
  std::string m_msg;
};

class FileNotExist : public std::exception {
 public:
  FileNotExist(const std::string& path);

  virtual const char* what() const throw();

 private:
  std::string m_path;
  std::string m_msg;
};

class NotCorrectFormat : public std::exception {
 public:
  NotCorrectFormat(const std::string& path, const std::string& sufix);

  virtual const char* what() const throw();

 private:
  std::string m_path;
  std::string m_sufix;
  std::string m_msg;
};

class NotIdenticalLengths : public std::exception {
 public:
  NotIdenticalLengths(size_t refLength, size_t length);

  virtual const char* what() const throw();

 private:
  size_t m_refLength;
  size_t m_length;
  std::string m_msg;
};

class NotInitialzed : public std::exception {
 public:
  virtual const char* what() const throw();
};

class TmpOapNotExist : public std::exception {
 public:
  virtual const char* what() const throw();
};
}
}

#endif  // EXCEPTIONS_H
