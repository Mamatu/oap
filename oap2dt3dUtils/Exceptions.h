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
  OutOfRange(unsigned int _x, unsigned int _y, unsigned int _width,
             unsigned int _height);

  unsigned int x;
  unsigned int y;
  unsigned int width;
  unsigned int height;

  virtual std::string getMessage() const;
};

class FileNotExist : public Exception {
 public:
  FileNotExist(const std::string& path);

  std::string m_path;

  virtual std::string getMessage() const;
};

class FileIsNotPng : public Exception {
 public:
  FileIsNotPng(const std::string& path);

  std::string m_path;

  virtual std::string getMessage() const;
};
}
}

#endif  // EXCEPTIONS_H
