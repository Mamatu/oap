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

#ifndef IEIGENCALCULATOR_H
#define IEIGENCALCULATOR_H

#include <vector>

#include "DeviceDataLoader.h"
#include "ArnoldiProceduresImpl.h"

namespace oap {

using DataLoaders = std::vector<DataLoader*>;

class IEigenCalculator {
 public:
  IEigenCalculator(CuHArnoldiCallback* cuHArnoldi);
  virtual ~IEigenCalculator() = 0;

  void loadData (const DataLoader::Info& dataInfo);
  void setDataLoader (DeviceDataLoader* dataLoader);

  void calculate();

  void setEigensCount(size_t eigensCount, size_t wantedEigensCount);

  ArnUtils::Type getEigenvectorsType() const;

  math::MatrixInfo getMatrixInfo() const;

  size_t getEigensCount() const;
  size_t getWantedEigensCount() const;

 protected:
  void setEigenvaluesOutput(floatt*);

  void setEigenvectorsOutput(math::Matrix**, ArnUtils::Type);

  oap::DeviceDataLoader* getDataLoader() const;

 private:
  void checkIfInitialized() const;
  void checkIfOutputInitialized() const;
  void checkIfDataLoaderInitialized() const;

  bool isInitialized() const;
  bool isOutputInitialized() const;
  bool isDataLoaderInitialized() const;

  void checkOutOfRange(size_t v, size_t max) const;

  void destroyEigenvalues();
  void destroyEigenvectors();

  void createArnoldiModule();
  void destroyArnoldiModule();

  void destroyDataLoader();

  size_t m_eigensCount;
  size_t m_wantedEigensCount;

  ArnUtils::Type m_eigenvectorsType;
  DeviceDataLoader* m_dataLoader;
  bool m_bDestroyDataLoader;
  CuHArnoldiCallback* m_cuHArnoldi;

  floatt* m_revalues;
  math::Matrix** m_eigenvectors;
};
}

#endif  // DATAOPERATOR_H
