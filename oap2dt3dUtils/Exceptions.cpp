#include "Exceptions.h"

namespace oap {
namespace exceptions {

OutOfRange::OutOfRange(unsigned int _x, unsigned int _y, unsigned int _width,
                       unsigned int _height)
    : x(_x), y(_y), width(_width), height(_height) {}

std::string OutOfRange::getMessage() const {
  std::string msg;
  if (x >= width) {
    msg += "x is out of range ";
  }
  if (y >= height) {
    msg += "y is out of range ";
  }
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
}
}
