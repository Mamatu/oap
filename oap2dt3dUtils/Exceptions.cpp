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

FileIsNotPng::FileIsNotPng(const std::string& path) : m_path(path) {}

std::string FileIsNotPng::getMessage() const {
  std::string msg;
  msg = m_path + " is not png.";
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
