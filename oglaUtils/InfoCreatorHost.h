#ifndef INFOCREATORHOST
#define INFOCREATORHOST

#include "InfoCreator.h"
#include "HostProcedures.h"

class InfoCreatorHost : public InfoCreator {
 private:
 protected:
  virtual void setInfoTypeCallback(const InfoType& infoType);
  virtual void setExpectedCallback(math::Matrix* expected);
  virtual void setOutputCallback(math::Matrix* output);

  virtual void getString(std::string& output, math::Matrix* matrix) const;

  virtual void getMean(floatt& re, floatt& im, math::Matrix* matrix) const;

  virtual bool compare(math::Matrix* matrix1, math::Matrix* matrix2) const;

  virtual math::Matrix* createDiffMatrix(math::Matrix* matrix1,
                                         math::Matrix* matrix2) const;

  virtual void destroyDiffMatrix(math::Matrix* diffMatrix) const;

  virtual uintt getIndexOfLargestValue(math::Matrix* matrix) const;
  virtual uintt getIndexOfSmallestValue(math::Matrix* matrix) const;
};

#endif  // INFOCREATORHOST
