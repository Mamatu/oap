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

EigenCalculator::EigenCalculator()
    : m_eigensCount(2), m_revalues(NULL), m_eigenvectors(NULL) {}

EigenCalculator::~EigenCalculator() {
  destroyEigenvalues();
  destroyEigenvectors();
}

math::Matrix* EigenCalculator::createMatrix(const DataLoaders& pngDataLoaders) {
  const size_t refLength = pngDataLoaders[0]->getLength();
  floatt* floatsvec = new floatt[refLength];

  math::Matrix* hostMatrix =
      host::NewReMatrix(pngDataLoaders.size(), refLength);

  for (size_t fa = 0; fa < pngDataLoaders.size(); ++fa) {
    DataLoader* it = pngDataLoaders[fa];
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

  const uintt width = m_dataLoaders.size();
  const uintt height = m_dataLoaders[0]->getLength();

  return ArnUtils::MatrixInfo(true, false, width, height);
}

void EigenCalculator::addPngDataLoader(DataLoader* pngDataLoader) {
  m_dataLoaders.push_back(pngDataLoader);
}

void EigenCalculator::calculate() {
  checkIfInitialized();

  ArnUtils::MatrixInfo matrixInfo = createMatrixInfo();

  m_cuHArnoldi = new CuHArnoldiDefault();
  m_cuHArnoldi->setSortType(ArnUtils::SortLargestReValues);

  initializeEigenvalues();
  initializeEigenvectors();

  const unsigned int hdim = 32;

  m_cuHArnoldi->setOutputsEigenvalues(m_revalues, NULL);
  m_cuHArnoldi->setOutputsEigenvectors(m_eigenvectors);

  m_cuHArnoldi->execute(hdim, m_eigensCount, matrixInfo);
}

void EigenCalculator::getEigenvalues(floatt* revalues) const {
  checkIfInitialized();

  memcpy(revalues, m_revalues, sizeof(floatt) * m_eigensCount);
}

void EigenCalculator::getEigenvectors(math::Matrix* eigenvectors) const {
  checkIfInitialized();

  host::CopyMatrix(eigenvectors, m_eigenvectors);
}

math::Matrix* EigenCalculator::createMatrix() const {
  checkIfInitialized();

  return EigenCalculator::createMatrix(m_dataLoaders);
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

bool EigenCalculator::isInitialized() const { return m_dataLoaders.size() > 0; }

void EigenCalculator::checkOutOfRange(size_t v, size_t max) const {
  if (v >= max) {
    throw oap::exceptions::OutOfRange(v, max);
  }
}

void EigenCalculator::initializeEigenvalues() {
  destroyEigenvalues();
  m_revalues = new floatt[m_eigensCount];
}

void EigenCalculator::initializeEigenvectors() {
  destroyEigenvectors();
  ArnUtils::MatrixInfo matrixInfo = createMatrixInfo();
  m_eigenvectors =
      host::NewReMatrix(m_eigensCount, matrixInfo.m_matrixDim.rows);
}

void EigenCalculator::destroyEigenvalues() {
  delete[] m_revalues;
  m_revalues = NULL;
}

void EigenCalculator::destroyEigenvectors() {
  host::DeleteMatrix(m_eigenvectors);
  m_eigenvectors = NULL;
}
}
