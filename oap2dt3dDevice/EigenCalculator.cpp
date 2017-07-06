/*
 * Copyright 2016, 2017 Marcin Matula
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
#include "HostMatrixUtils.h"
#include "ArnoldiProceduresImpl.h"

namespace oap {

EigenCalculator::EigenCalculator(CuHArnoldi* cuHArnoldi)
    : m_eigensCount(0),
      m_eigenvectorsType(ArnUtils::UNDEFINED),
      m_dataLoader(NULL),
      m_cuHArnoldi(cuHArnoldi),
      m_revalues(NULL),
      m_eigenvectors(NULL) {}

EigenCalculator::~EigenCalculator() {}

void EigenCalculator::setDataLoader(DataLoader* dataLoader) {
  m_dataLoader = dataLoader;
}

void EigenCalculator::calculate() {
  checkIfInitialized();

  math::MatrixInfo matrixInfo = m_dataLoader->getMatrixInfo();

  m_cuHArnoldi->setSortType(ArnUtils::SortLargestReValues);
  m_cuHArnoldi->setOutputType(m_eigenvectorsType);

  const unsigned int hdim = 32;

  m_cuHArnoldi->setOutputsEigenvalues(m_revalues, NULL);
  m_cuHArnoldi->setOutputsEigenvectors(m_eigenvectors);

  m_cuHArnoldi->execute(hdim, m_eigensCount, matrixInfo);
}

void EigenCalculator::setEigensCount(size_t eigensCount) {
  m_eigensCount = eigensCount;
}

void EigenCalculator::setEigenvaluesOutput(floatt* revalues) {
  m_revalues = revalues;
}

void EigenCalculator::setEigenvectorsOutput(math::Matrix** eigenvectors) {
  m_eigenvectors = eigenvectors;
}

void EigenCalculator::setEigenvectorsType(ArnUtils::Type eigenvectorsType) {
  m_eigenvectorsType = eigenvectorsType;
}

math::MatrixInfo EigenCalculator::getMatrixInfo() const {
  checkIfDataLoaderInitialized();
  return m_dataLoader->getMatrixInfo();
}

void EigenCalculator::checkIfInitialized() const {
  if (!isInitialized()) {
    throw oap::exceptions::NotInitialzed();
  }
}

void EigenCalculator::checkIfOutputInitialized() const {
  if (!isOutputInitialized()) {
    throw oap::exceptions::NotInitialzed();
  }
}

void EigenCalculator::checkIfDataLoaderInitialized() const {
  if (!isDataLoaderInitialized()) {
    throw oap::exceptions::NotInitialzed();
  }
}

bool EigenCalculator::isInitialized() const {
  return m_eigensCount > 0 && m_eigenvectorsType != ArnUtils::UNDEFINED &&
         isOutputInitialized() && isDataLoaderInitialized();
}

bool EigenCalculator::isOutputInitialized() const {
  return m_eigenvectors != NULL && m_revalues != NULL;
}

bool EigenCalculator::isDataLoaderInitialized() const {
  return m_dataLoader != NULL;
}

void EigenCalculator::checkOutOfRange(size_t v, size_t max) const {
  if (v >= max) {
    throw oap::exceptions::OutOfRange(v, max);
  }
}
}
