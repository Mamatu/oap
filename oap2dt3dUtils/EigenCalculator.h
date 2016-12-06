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

#ifndef DATAOPERATOR_H
#define DATAOPERATOR_H

#include <vector>

#include "DataLoader.h"
#include "Matrix.h"
#include "ArnoldiProceduresImpl.h"

namespace oap {

typedef std::vector<DataLoader*> DataLoaders;

class EigenCalculator {
 public:
  EigenCalculator();
  virtual ~EigenCalculator();

  static math::Matrix* createMatrix(const DataLoaders& pngDataLoaders);

  void addPngDataLoader(DataLoader* pngDataLoader);

  void calculate();

  floatt getEigenvalue(uintt index) const;

  math::Matrix* getEigenvector(uintt index) const;

  /**
   * @brief Creates matrix from sets of pngDataLoader
   * @return matrix in host space
   */
  math::Matrix* createMatrix() const;

  /**
   * @brief Creates device matrix from set of pngDataLoader
   * @return matrix in device space
   */
  math::Matrix* createDeviceMatrix() const;

  /**
   * @brief Creates Matrxinfo from set of pngDataLoader
   * @return
   */
  ArnUtils::MatrixInfo createMatrixInfo() const;

 private:
  void checkIfInitialized() const;
  bool isInitialized() const;

  void checkOutOfRange(size_t v, size_t max) const;

  size_t m_eigensCount;
  DataLoaders m_dataLoaders;
  CuHArnoldi* m_cuHArnoldi;
};
}

#endif  // DATAOPERATOR_H
