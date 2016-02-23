#ifndef INFOCREATOR_H
#define INFOCREATOR_H

#include "MatrixAPI.h"
#include "InfoType.h"

class InfoCreator {
 private:
  InfoType m_infoType;
  math::Matrix* m_expected;
  math::Matrix* m_output;

 public:
  InfoCreator();
  virtual ~InfoCreator();

  void setInfoType(const InfoType& infoType);
  InfoType getInfoType() const;

  void setExpected(math::Matrix* expected);
  void setOutput(math::Matrix* output);

  void getInfo(std::string& output) const;

 protected:
  class ComplexInternal : public Complex {
   public:
    bool m_isValid;
    ComplexInternal(bool isValid);
  };

  enum MatrixType {
    EXPECTED = 1 << 0,
    OUTPUT = 1 << 1,
    HOST = 1 << 2,
    DEVICE = 1 << 3
  };

  math::Matrix* getExpectedMatrix() const;
  math::Matrix* getOutputMatrix() const;

  virtual void setInfoTypeCallback(const InfoType& infoType);
  virtual void setExpectedCallback(math::Matrix* expected);
  virtual void setOutputCallback(math::Matrix* output);

  virtual math::Matrix* getMatrix(MatrixType matrixType) const = 0;

  virtual void getString(std::string& output, math::Matrix* matrix,
                         MatrixType matrixType) const = 0;

  virtual Complex getMean(math::Matrix* matrix,
                          MatrixType matrixType) const = 0;

  virtual math::Matrix* createDiffMatrix(math::Matrix* matrix1,
                                         math::Matrix* matrix2,
                                         MatrixType matrixType) const = 0;

  virtual void destroyDiffMatrix(math::Matrix* diffMatrix) const = 0;

  virtual uintt getIndexOfLargestValue(math::Matrix* matrix) const = 0;
  virtual uintt getIndexOfSmallestValue(math::Matrix* matrix) const = 0;

 private:
  void printMatrix(std::string& output, const std::string& message,
                   math::Matrix* matrix) const;

  bool printMean(std::string& output, const std::string& message,
                 InfoCreator::MatrixType matrixType) const;

  void printMean(std::string& output, const std::string& message,
                 math::Matrix* matrix) const;
};

class InfoCreatorHost : public InfoCreator {
 public:
 protected:
  virtual math::Matrix* getMatrix(MatrixType matrixType) const;

  virtual void getString(std::string& output, math::Matrix* matrix,
                         MatrixType matrixType) const;

  virtual Complex getMean(math::Matrix* matrix, MatrixType matrixType) const;

  virtual math::Matrix* createDiffMatrix(math::Matrix* matrix1,
                                         math::Matrix* matrix2,
                                         MatrixType matrixType) const;

  virtual void destroyDiffMatrix(math::Matrix* diffMatrix) const;

  virtual uintt getIndexOfLargestValue(math::Matrix* matrix) const;
  virtual uintt getIndexOfSmallestValue(math::Matrix* matrix) const;
};

/*
class InfoCreatorHost : public InfoCreator {
 private:
  inline uintt getIndexOfLargestDiff(math::Matrix* diff) const;
  inline uintt getIndexOfSmallestDiff(math::Matrix* diff) const;

 protected:
  void printMatrix(std::string& output, const std::string& message,
                   math::Matrix* matrix) const;

  void printMean(std::string& output, const std::string& message,
                 math::Matrix* matrix) const;

  floatt calculateMean(const matrixUtils::MatrixRange& matrixRange,
                       floatt (*GetValue)(const math::Matrix*, uintt,
                                          uintt)) const;
  uintt getIndexOfLargestDiff() const;
  uintt getIndexOfSmallestDiff() const;

  virtual Complex getMean(math::Matrix* matrix, MatrixType matrixType) const;

 public:
  InfoCreatorHost();

  InfoCreatorHost(math::Matrix* expected, math::Matrix* output,
                  const InfoType& infoType);

  virtual ~InfoCreatorHost();

  bool isEquals() const;
};*/

#endif  // INFOCREATOR_H
