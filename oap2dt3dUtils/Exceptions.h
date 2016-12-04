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

#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <string>

namespace oap {
namespace exceptions {

class Exception {
 public:
  virtual std::string getMessage() const = 0;
};

class OutOfRange : public Exception {
 public:
  OutOfRange(unsigned int _value, unsigned int _maxValue);

  virtual std::string getMessage() const;

 private:
  unsigned int value;
  unsigned int maxValue;
};

class FileNotExist : public Exception {
 public:
  FileNotExist(const std::string& path);

  virtual std::string getMessage() const;

 private:
  std::string m_path;
};

class FileIsNotPng : public Exception {
 public:
  FileIsNotPng(const std::string& path);

  virtual std::string getMessage() const;

 private:
  std::string m_path;
};

class NotIdenticalLengths : public Exception {
 public:
  NotIdenticalLengths(size_t refLength, size_t length);

  virtual std::string getMessage() const;

 private:
  size_t m_refLength;
  size_t m_length;
};

class NotInitialzed : public Exception {
 public:
  virtual std::string getMessage() const;
};
}
}

#endif  // EXCEPTIONS_H
