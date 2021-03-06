/*
 * Copyright 2016 - 2021 Marcin Matula
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

#include "MatrixPrinter.h"

#include "oapHostMatrixUtils.h"

#include "Utils.h"

void HostInfoCreator::setInfoTypeCallback(const InfoType& infoType) {}

void HostInfoCreator::setExpectedCallback(math::ComplexMatrix* expected) {}

void HostInfoCreator::setOutputCallback(math::ComplexMatrix* output) {}

void HostInfoCreator::getString(std::string& output, math::ComplexMatrix* matrix) const
{
  matrixUtils::PrintMatrix(output, matrix, matrixUtils::PrintArgs());
}

void HostInfoCreator::getMean(floatt& re, floatt& im,
                              math::ComplexMatrix* matrix) const {
  if (gReValues (matrix)) {
    re = utils::getMean(gReValues (matrix), gColumns (matrix) * gRows (matrix), -1.);
  }
  if (gImValues (matrix)) {
    im = utils::getMean(gImValues (matrix), gColumns (matrix) * gRows (matrix), -1.);
  }
}

bool HostInfoCreator::compare(math::ComplexMatrix* matrix1, math::ComplexMatrix* matrix2,
                              floatt tolerance, math::ComplexMatrix** diffMatrix) const {
  return utils::IsEqual(*matrix1, *matrix2, tolerance, diffMatrix);
}

bool HostInfoCreator::compareValues(math::ComplexMatrix* matrix1,
                                    math::ComplexMatrix* matrix2,
                                    floatt tolerance,
                                    math::ComplexMatrix** diffMatrix) const {
  return utils::HasValues(*matrix1, *matrix2, tolerance, diffMatrix);
}

void HostInfoCreator::destroyMatrix(math::ComplexMatrix* diffMatrix) const {
  oap::host::DeleteMatrix(diffMatrix);
}

bool HostInfoCreator::isRe(math::ComplexMatrix* matrix) const {
  return gReValues (matrix) != NULL;
}

bool HostInfoCreator::isIm(math::ComplexMatrix* matrix) const {
  return gImValues (matrix) != NULL;
}

std::pair<floatt, uintt> HostInfoCreator::getLargestReValue(
    math::ComplexMatrix* matrix) const {
  return utils::getLargest(gReValues (matrix), gColumns (matrix) * gRows (matrix));
}

std::pair<floatt, uintt> HostInfoCreator::getLargestImValue(
    math::ComplexMatrix* matrix) const {
  return utils::getLargest(gImValues (matrix), gColumns (matrix) * gRows (matrix));
}

std::pair<floatt, uintt> HostInfoCreator::getSmallestReValue(
    math::ComplexMatrix* matrix) const {
  return utils::getSmallest(gReValues (matrix), gColumns (matrix) * gRows (matrix));
}

std::pair<floatt, uintt> HostInfoCreator::getSmallestImValue(
    math::ComplexMatrix* matrix) const {
  return utils::getSmallest(gImValues (matrix), gColumns (matrix) * gRows (matrix));
}
