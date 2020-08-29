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

#include "IEigenCalculator.h"
#include "Exceptions.h"
#include "oapHostMatrixUtils.h"
#include "ArnoldiProceduresImpl.h"

#include "PngFile.h"
#include "DeviceImagesLoader.h"

namespace oap {

IEigenCalculator::IEigenCalculator(CuHArnoldiCallback* cuHArnoldi)
    : m_eigensCount(0),
      m_eigenvectorsType(ArnUtils::UNDEFINED),
      m_dataLoader(nullptr),
      m_bDestroyImagesLoader(false),
      m_cuHArnoldi(cuHArnoldi),
      m_revalues(nullptr),
      m_eigenvectors(nullptr) {}

IEigenCalculator::~IEigenCalculator()
{}

void IEigenCalculator::loadData (const ImagesLoader::Info& dataInfo)
{
  destroyImagesLoader();
  m_dataLoader = ImagesLoader::createImagesLoader<oap::PngFile, oap::DeviceImagesLoader> (dataInfo);
  m_bDestroyImagesLoader = true;
}

void IEigenCalculator::setImagesLoader (DeviceImagesLoader* dataLoader)
{
  destroyImagesLoader();
  m_dataLoader = dataLoader;
  m_bDestroyImagesLoader = false;
}

void IEigenCalculator::calculate() {
  checkIfInitialized();

  math::MatrixInfo matrixInfo = m_dataLoader->getMatrixInfo();

  m_cuHArnoldi->setSortType(ArnUtils::SortLargestReValues);
  m_cuHArnoldi->setOutputType(m_eigenvectorsType);
  m_cuHArnoldi->setCheckType(ArnUtils::CHECK_INTERNAL);
  m_cuHArnoldi->setCalcTraingularHType(ArnUtils::CALC_IN_HOST);

  m_cuHArnoldi->setOutputsEigenvalues(m_revalues, NULL);
  m_cuHArnoldi->setOutputsEigenvectors(m_eigenvectors);

  m_cuHArnoldi->execute(m_eigensCount, m_wantedEigensCount, matrixInfo);
}

void IEigenCalculator::setEigensCount(size_t eigensCount, size_t wantedEigensCount) {
  m_eigensCount = eigensCount;
  m_wantedEigensCount = wantedEigensCount;
}

void IEigenCalculator::setEigenvaluesOutput(floatt* revalues) {
  m_revalues = revalues;
}

oap::DeviceImagesLoader* IEigenCalculator::getImagesLoader() const
{
  return m_dataLoader;
}

void IEigenCalculator::setEigenvectorsOutput(math::Matrix** eigenvectors, ArnUtils::Type eigenvectorsType) {
  m_eigenvectors = eigenvectors;
  m_eigenvectorsType = eigenvectorsType;
}

ArnUtils::Type IEigenCalculator::getEigenvectorsType() const {
  return m_eigenvectorsType;
}

math::MatrixInfo IEigenCalculator::getMatrixInfo() const {
  checkIfImagesLoaderInitialized();
  return m_dataLoader->getMatrixInfo();
}

size_t IEigenCalculator::getEigensCount() const
{
  return m_eigensCount;
}

size_t IEigenCalculator::getWantedEigensCount() const
{
  return m_wantedEigensCount;
}

void IEigenCalculator::checkIfInitialized() const {
  if (!isInitialized()) {
    throw oap::exceptions::NotInitialzed();
  }
}

void IEigenCalculator::checkIfOutputInitialized() const {
  if (!isOutputInitialized()) {
    throw oap::exceptions::NotInitialzed();
  }
}

void IEigenCalculator::checkIfImagesLoaderInitialized() const {
  if (!isImagesLoaderInitialized()) {
    throw oap::exceptions::NotInitialzed();
  }
}

bool IEigenCalculator::isInitialized() const {
  return m_eigensCount > 0 && m_eigenvectorsType != ArnUtils::UNDEFINED &&
         isOutputInitialized() && isImagesLoaderInitialized();
}

bool IEigenCalculator::isOutputInitialized() const {
  return m_eigenvectors != NULL && m_revalues != NULL;
}

bool IEigenCalculator::isImagesLoaderInitialized() const {
  return m_dataLoader != NULL;
}

void IEigenCalculator::checkOutOfRange(size_t v, size_t max) const {
  if (v >= max) {
    throw oap::exceptions::OutOfRange(v, max);
  }
}

void IEigenCalculator::destroyImagesLoader()
{
  if (m_bDestroyImagesLoader)
  {
    delete m_dataLoader;
  }
}
}
