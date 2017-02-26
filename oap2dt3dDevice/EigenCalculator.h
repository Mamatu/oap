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

#ifndef EIGENCALCULATOR_H
#define EIGENCALCULATOR_H

#include <vector>

#include "DataLoader.h"
#include "ArnoldiProceduresImpl.h"

namespace oap {

typedef std::vector<DataLoader*> DataLoaders;

class EigenCalculator {
 public:
  EigenCalculator(CuHArnoldi* cuHArnoldi);
  virtual ~EigenCalculator();

  void setDataLoader(DataLoader* dataLoader);

  void calculate();

  void setEigensCount(size_t eigensCount);

  void getEigenvalues(floatt*) const;

  void getEigenvectors(math::Matrix**) const;

  ArnUtils::MatrixInfo getMatrixInfo() const;
 private:
  void checkIfInitialized() const;
  bool isInitialized() const;

  void checkOutOfRange(size_t v, size_t max) const;

  void initializeEigenvalues();
  void initializeEigenvectors();

  void destroyEigenvalues();
  void destroyEigenvectors();

  void createArnoldiModule();
  void destroyArnoldiModule();

  size_t m_eigensCount;
  DataLoader* m_dataLoader;
  CuHArnoldi* m_cuHArnoldi;

  floatt* m_revalues;
  math::Matrix** m_eigenvectors;
};
}

#endif  // DATAOPERATOR_H
