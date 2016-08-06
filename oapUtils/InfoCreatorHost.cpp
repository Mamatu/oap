#include "InfoCreatorHost.h"
#include "MatrixUtils.h"
#include "HostMatrixModules.h"
#include "Utils.h"

void InfoCreatorHost::setInfoTypeCallback(const InfoType& infoType) {}

void InfoCreatorHost::setExpectedCallback(math::Matrix* expected) {}

void InfoCreatorHost::setOutputCallback(math::Matrix* output) {}

void InfoCreatorHost::getString(std::string& output,
                                math::Matrix* matrix) const {
  matrixUtils::PrintMatrix(output, matrix);
}

void InfoCreatorHost::getMean(floatt& re, floatt& im,
                              math::Matrix* matrix) const {
  re = utils::getMean(matrix->reValues, matrix->columns * matrix->rows);
  im = utils::getMean(matrix->imValues, matrix->columns * matrix->rows);
}

bool InfoCreatorHost::compare(math::Matrix* matrix1,
                              math::Matrix* matrix2) const {
  return utils::IsEqual(*matrix1, *matrix2);
}

math::Matrix* InfoCreatorHost::createDiffMatrix(math::Matrix* matrix1,
                                                math::Matrix* matrix2) const {
  math::Matrix* output = NULL;
  utils::IsEqual(*matrix1, *matrix2, &output);
  return output;
}

void InfoCreatorHost::destroyDiffMatrix(math::Matrix* diffMatrix) const {
  host::DeleteMatrix(diffMatrix);
}

bool InfoCreatorHost::isRe(math::Matrix* matrix) const {
  return matrix->reValues != NULL;
}

bool InfoCreatorHost::isIm(math::Matrix* matrix) const {
  return matrix->imValues != NULL;
}

std::pair<floatt, uintt> InfoCreatorHost::getLargestReValue(
    math::Matrix* matrix) const {
  return utils::getLargest(matrix->reValues, matrix->columns * matrix->rows);
}

std::pair<floatt, uintt> InfoCreatorHost::getLargestImValue(
    math::Matrix* matrix) const {
  return utils::getLargest(matrix->imValues, matrix->columns * matrix->rows);
}

std::pair<floatt, uintt> InfoCreatorHost::getSmallestReValue(
    math::Matrix* matrix) const {
  return utils::getSmallest(matrix->reValues, matrix->columns * matrix->rows);
}

std::pair<floatt, uintt> InfoCreatorHost::getSmallestImValue(
    math::Matrix* matrix) const {
  return utils::getSmallest(matrix->imValues, matrix->columns * matrix->rows);
}
