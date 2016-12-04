/*
 * Copyright 2016 Marcin Matula
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

#include "EigenCalculator.h"
#include "Exceptions.h"
#include "HostMatrixModules.h"
#include "DeviceMatrixModules.h"
#include "ArnoldiProceduresImpl.h"

namespace oap {

EigenCalculator::EigenCalculator() : m_eigensCount(2) {}

EigenCalculator::~EigenCalculator() {}

math::Matrix* EigenCalculator::createMatrix(
    const PngDataLoaders& pngDataLoaders) {
  const size_t refLength = pngDataLoaders[0]->getLength();
  floatt* floatsvec = new floatt[refLength];

  math::Matrix* hostMatrix =
      host::NewReMatrix(pngDataLoaders.size(), refLength);

  for (size_t fa = 0; fa < pngDataLoaders.size(); ++fa) {
    PngDataLoader* it = pngDataLoaders[fa];
    const size_t length = it->getLength();
    if (refLength != length) {
      delete[] floatsvec;
      host::DeleteMatrix(hostMatrix);
      throw oap::exceptions::NotIdenticalLengths(refLength, length);
    }
    it->getFloattVector(floatsvec);
    host::SetReVector(hostMatrix, fa, floatsvec, refLength);
  }

  delete[] floatsvec;

  return hostMatrix;
}

ArnUtils::MatrixInfo EigenCalculator::createMatrixInfo() const {
  checkIfInitialized();

  const uintt width = m_pngDataLoaders.size();
  const uintt height = m_pngDataLoaders[0]->getLength();

  return ArnUtils::MatrixInfo(true, false, width, height);
}

void EigenCalculator::addPngDataLoader(PngDataLoader* pngDataLoader) {
  m_pngDataLoaders.push_back(pngDataLoader);
}

void EigenCalculator::calculate() {
  checkIfInitialized();

  ArnUtils::MatrixInfo matrixInfo = createMatrixInfo();

  m_cuHArnoldi = new CuHArnoldiDefault();
  m_cuHArnoldi->setSortType(ArnUtils::SortLargestReValues);

  m_cuHArnoldi->execute(32, m_eigensCount, matrixInfo);
}

floatt EigenCalculator::getEigenvalue(uintt index) const {
  checkIfInitialized();
  checkOutOfRange(index, m_eigensCount);

  return 0;  // m_cuHArnoldi->getEigenvalue(index);
}

math::Matrix* EigenCalculator::getEigenvector(uintt index) const {
  checkIfInitialized();
  checkOutOfRange(index, m_eigensCount);

  return 0;  // m_cuHArnoldi->getEigenvector(index);
}

math::Matrix* EigenCalculator::createMatrix() const {
  checkIfInitialized();

  return EigenCalculator::createMatrix(m_pngDataLoaders);
}

math::Matrix* EigenCalculator::createDeviceMatrix() const {
  checkIfInitialized();

  math::Matrix* host = createMatrix();
  math::Matrix* device = device::NewDeviceMatrixCopy(host);
  host::DeleteMatrix(host);
  return device;
}

void EigenCalculator::checkIfInitialized() const {
  if (!isInitialized()) {
    throw oap::exceptions::NotInitialzed();
  }
}

bool EigenCalculator::isInitialized() const {
  return m_pngDataLoaders.size() > 0;
}

void EigenCalculator::checkOutOfRange(size_t v, size_t max) const {
  if (v >= max) {
    throw oap::exceptions::OutOfRange(v, max);
  }
}
}
