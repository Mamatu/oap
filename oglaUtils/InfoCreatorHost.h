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

  virtual std::pair<floatt, uintt> getLargestReValue(
      math::Matrix* matrix) const;
  virtual std::pair<floatt, uintt> getLargestImValue(
      math::Matrix* matrix) const;

  virtual std::pair<floatt, uintt> getSmallestReValue(
      math::Matrix* matrix) const;
  virtual std::pair<floatt, uintt> getSmallestImValue(
      math::Matrix* matrix) const;

  virtual bool isRe(math::Matrix* matrix) const;

  virtual bool isIm(math::Matrix* matrix) const;
};

#endif  // INFOCREATORHOST
