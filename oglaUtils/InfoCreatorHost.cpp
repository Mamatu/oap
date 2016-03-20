#include "InfoCreatorHost.h"
#include "MatrixUtils.h"
#include "HostMatrixModules.h"

void InfoCreatorHost::setInfoTypeCallback(const InfoType& infoType) {}

void InfoCreatorHost::setExpectedCallback(math::Matrix* expected) {}

void InfoCreatorHost::setOutputCallback(math::Matrix* output) {}

void InfoCreatorHost::getString(std::string& output,
                                math::Matrix* matrix) const {
  matrixUtils::PrintMatrix(output, matrix);
}

void InfoCreatorHost::getMean(floatt& re, floatt& im,
                              math::Matrix* matrix) const {}

bool InfoCreatorHost::compare(math::Matrix* matrix1,
                              math::Matrix* matrix2) const {
  HostProcedures hostProcedures;
  return hostProcedures.isEqual(matrix1, matrix2);
}

math::Matrix* InfoCreatorHost::createDiffMatrix(math::Matrix* matrix1,
                                                math::Matrix* matrix2) const {
  HostProcedures hostProcedures;
  if (hostProcedures.isEqual(matrix1, matrix2) == true) {
    return NULL;
  }
  math::Matrix* output = host::NewMatrix(matrix1);
  hostProcedures.substract(output, matrix1, matrix2);
  return output;
}

void InfoCreatorHost::destroyDiffMatrix(math::Matrix* diffMatrix) const {
  host::DeleteMatrix(diffMatrix);
}

uintt InfoCreatorHost::getIndexOfLargestValue(math::Matrix* matrix) const {}

uintt InfoCreatorHost::getIndexOfSmallestValue(math::Matrix* matrix) const {}
