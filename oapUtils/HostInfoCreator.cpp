/*
 * Copyright 2016 - 2018 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "HostInfoCreator.h"
#include "MatrixUtils.h"
#include "oapHostMatrixUtils.h"
#include "Utils.h"

void HostInfoCreator::setInfoTypeCallback(const InfoType& infoType) {}

void HostInfoCreator::setExpectedCallback(math::Matrix* expected) {}

void HostInfoCreator::setOutputCallback(math::Matrix* output) {}

void HostInfoCreator::getString(std::string& output,
                                math::Matrix* matrix) const {
  matrixUtils::PrintMatrix(output, matrix);
}

void HostInfoCreator::getMean(floatt& re, floatt& im,
                              math::Matrix* matrix) const {
  if (matrix->reValues) {
    re = utils::getMean(matrix->reValues, matrix->columns * matrix->rows, -1.);
  }
  if (matrix->imValues) {
    im = utils::getMean(matrix->imValues, matrix->columns * matrix->rows, -1.);
  }
}

bool HostInfoCreator::compare(math::Matrix* matrix1, math::Matrix* matrix2,
                              math::Matrix** diffMatrix) const {
  return utils::IsEqual(*matrix1, *matrix2, diffMatrix);
}

bool HostInfoCreator::compareValues(math::Matrix* matrix1,
                                    math::Matrix* matrix2,
                                    math::Matrix** diffMatrix) const {
  return utils::HasValues(*matrix1, *matrix2, diffMatrix);
}

void HostInfoCreator::destroyMatrix(math::Matrix* diffMatrix) const {
  oap::host::DeleteMatrix(diffMatrix);
}

bool HostInfoCreator::isRe(math::Matrix* matrix) const {
  return matrix->reValues != NULL;
}

bool HostInfoCreator::isIm(math::Matrix* matrix) const {
  return matrix->imValues != NULL;
}

std::pair<floatt, uintt> HostInfoCreator::getLargestReValue(
    math::Matrix* matrix) const {
  return utils::getLargest(matrix->reValues, matrix->columns * matrix->rows);
}

std::pair<floatt, uintt> HostInfoCreator::getLargestImValue(
    math::Matrix* matrix) const {
  return utils::getLargest(matrix->imValues, matrix->columns * matrix->rows);
}

std::pair<floatt, uintt> HostInfoCreator::getSmallestReValue(
    math::Matrix* matrix) const {
  return utils::getSmallest(matrix->reValues, matrix->columns * matrix->rows);
}

std::pair<floatt, uintt> HostInfoCreator::getSmallestImValue(
    math::Matrix* matrix) const {
  return utils::getSmallest(matrix->imValues, matrix->columns * matrix->rows);
}
