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

  virtual std::string getMessage() const;

 private:
  unsigned int x;
  unsigned int y;
  unsigned int width;
  unsigned int height;
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
}
}

#endif  // EXCEPTIONS_H
