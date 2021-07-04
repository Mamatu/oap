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

#ifndef IEIGENCALCULATOR_H
#define IEIGENCALCULATOR_H

#include <vector>

#include "DeviceImagesLoader.hpp"
#include "ArnoldiProceduresImpl.hpp"

namespace oap {

using ImagesLoaders = std::vector<ImagesLoader*>;

class IEigenCalculator {
 public:
  IEigenCalculator(CuHArnoldiCallback* cuHArnoldi);
  virtual ~IEigenCalculator() = 0;

  void loadData (const ImagesLoader::Info& dataInfo);
  void setImagesLoader (DeviceImagesLoader* dataLoader);

  void calculate();

  void setEigensCount(size_t eigensCount, size_t wantedEigensCount);

  ArnUtils::Type getEigenvectorsType() const;

  math::MatrixInfo getMatrixInfo() const;

  size_t getEigensCount() const;
  size_t getWantedEigensCount() const;

 protected:
  void setEigenvaluesOutput(floatt*);

  void setEigenvectorsOutput(math::ComplexMatrix**, ArnUtils::Type);

  oap::DeviceImagesLoader* getImagesLoader() const;

 private:
  void checkIfInitialized() const;
  void checkIfOutputInitialized() const;
  void checkIfImagesLoaderInitialized() const;

  bool isInitialized() const;
  bool isOutputInitialized() const;
  bool isImagesLoaderInitialized() const;

  void checkOutOfRange(size_t v, size_t max) const;

  void destroyEigenvalues();
  void destroyEigenvectors();

  void createArnoldiModule();
  void destroyArnoldiModule();

  void destroyImagesLoader();

  size_t m_eigensCount;
  size_t m_wantedEigensCount;

  ArnUtils::Type m_eigenvectorsType;
  DeviceImagesLoader* m_dataLoader;
  bool m_bDestroyImagesLoader;
  CuHArnoldiCallback* m_cuHArnoldi;

  floatt* m_revalues;
  math::ComplexMatrix** m_eigenvectors;
};
}

#endif  // DATAOPERATOR_H
