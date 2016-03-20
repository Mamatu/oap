#ifndef INFOCREATOR_H
#define INFOCREATOR_H

#include "MatrixAPI.h"
#include "InfoType.h"

class InfoCreator {
 private:
  math::Matrix* m_expected;
  math::Matrix* m_output;

 public:
  InfoCreator();
  virtual ~InfoCreator();

  void setExpected(math::Matrix* expected);
  void setOutput(math::Matrix* output);

  void getInfo(std::string& output, const InfoType& infoType) const;

 protected:
  class ComplexInternal : public Complex {
   public:
    bool m_isValid;
    ComplexInternal(bool isValid);
  };

  virtual math::Matrix* getExpectedMatrix() const;
  virtual math::Matrix* getOutputMatrix() const;

  virtual void setInfoTypeCallback(const InfoType& infoType);
  virtual void setExpectedCallback(math::Matrix* expected);
  virtual void setOutputCallback(math::Matrix* output);

  virtual void getString(std::string& output, math::Matrix* matrix) const = 0;

  virtual void getMean(floatt& re, floatt& im, math::Matrix* matrix) const = 0;

  virtual bool compare(math::Matrix* matrix1, math::Matrix* matrix2) const = 0;

  virtual math::Matrix* createDiffMatrix(math::Matrix* matrix1,
                                         math::Matrix* matrix2) const = 0;

  virtual void destroyDiffMatrix(math::Matrix* diffMatrix) const = 0;

  virtual uintt getIndexOfLargestValue(math::Matrix* matrix) const = 0;
  virtual uintt getIndexOfSmallestValue(math::Matrix* matrix) const = 0;

 public:
  void printMatrix(std::string& output, const std::string& message,
                   math::Matrix* matrix) const;

  bool printMean(std::string& output) const;

  void printMean(std::string& output, const std::string& message,
                 math::Matrix* matrix) const;

  bool isEqual() const;
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
