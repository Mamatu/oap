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

#ifndef OAP_HOST_MATRIX_MODULES_H
#define OAP_HOST_MATRIX_MODULES_H

#include "MatrixModules.h"
#include "HostMatrixUtils.h"
#include "Matrix.h"
#include "MatrixInfo.h"
#include "MatrixEx.h"
#include "ThreadUtils.h"

#include <stdio.h>
#include <vector>

class HostMatrixUtils : public MatrixUtils {
 public:
  HostMatrixUtils();
  ~HostMatrixUtils();
  void getReValues(floatt* dst, math::Matrix* matrix, uintt index,
                   uintt length);
  void getImValues(floatt* dst, math::Matrix* matrix, uintt index,
                   uintt length);
  void setReValues(math::Matrix* matrix, floatt* src, uintt index,
                   uintt length);
  void setImValues(math::Matrix* matrix, floatt* src, uintt index,
                   uintt length);
  void setDiagonalReMatrix(math::Matrix* matrix, floatt a);
  uintt getColumns(const math::Matrix* matrix) const;
  uintt getRows(const math::Matrix* matrix) const;
  bool isMatrix(const math::Matrix* matrix) const;
  bool isReMatrix(const math::Matrix* matrix) const;
  bool isImMatrix(const math::Matrix* matrix) const;
};

class HostMatrixCopier : public MatrixCopier {
  HostMatrixUtils hmu;

 public:
  HostMatrixCopier();
  virtual ~HostMatrixCopier();
  void copyMatrixToMatrix(math::Matrix* dst, const math::Matrix* src);
  void copyReMatrixToReMatrix(math::Matrix* dst, const math::Matrix* src);
  void copyImMatrixToImMatrix(math::Matrix* dst, const math::Matrix* src);
  /**
   * Copy floatts where length is number of floatts (not bytes!).
   * @param dst
   * @param src
   * @param length number of numbers to copy
   */
  void copy(floatt* dst, const floatt* src, uintt length);

  void setReVector(math::Matrix* matrix, uintt column, floatt* vector,
                   uintt length);
  void setTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector,
                            uintt length);
  void setImVector(math::Matrix* matrix, uintt column, floatt* vector,
                   uintt length);
  void setTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector,
                            uintt length);

  void getReVector(floatt* vector, uintt length, math::Matrix* matrix,
                   uintt column);
  void getTransposeReVector(floatt* vector, uintt length, math::Matrix* matrix,
                            uintt row);
  void getImVector(floatt* vector, uintt length, math::Matrix* matrix,
                   uintt column);
  void getTransposeImVector(floatt* vector, uintt length, math::Matrix* matrix,
                            uintt row);
  void setVector(math::Matrix* matrix, uintt column, math::Matrix* vector,
                 uintt rows);
  void getVector(math::Matrix* vector, uintt rows, math::Matrix* matrix,
                 uintt column);
};

class HostMatrixAllocator : public MatrixAllocator {
  HostMatrixUtils hmu;
  HostMatrixCopier hmc;
  typedef std::vector<math::Matrix*> HostMatrices;
  static HostMatrices hostMatrices;
  static utils::sync::Mutex mutex;
  static math::Matrix* createHostMatrix(math::Matrix* matrix, uintt columns,
                                        uintt rows, floatt* values,
                                        floatt** valuesPtr);
  static void initMatrix(math::Matrix* matrix);
  static math::Matrix* createHostReMatrix(uintt columns, uintt rows,
                                          floatt* values);
  static math::Matrix* createHostImMatrix(uintt columns, uintt rows,
                                          floatt* values);

 public:
  HostMatrixAllocator();
  ~HostMatrixAllocator();
  math::Matrix* newReMatrix(uintt columns, uintt rows, floatt value = 0);
  math::Matrix* newImMatrix(uintt columns, uintt rows, floatt value = 0);
  math::Matrix* newMatrix(uintt columns, uintt rows, floatt value = 0);
  bool isMatrix(math::Matrix* matrix);
  math::Matrix* newMatrixFromAsciiFile(const char* path);
  math::Matrix* newMatrixFromBinaryFile(const char* path);
  void deleteMatrix(math::Matrix* matrix);
};

class HostMatrixPrinter : public MatrixPrinter {
 public:
  HostMatrixPrinter();
  ~HostMatrixPrinter();
  void getMatrixStr(std::string& str, const math::Matrix* matrix);
  void getReMatrixStr(std::string& str, const math::Matrix* matrix);
  void getImMatrixStr(std::string& str, const math::Matrix* matrix);
};

class HostMatrixModules : public MatrixModule {
  HostMatrixAllocator* m_hma;
  HostMatrixCopier* m_hmc;
  HostMatrixUtils* m_hmu;
  HostMatrixPrinter* m_hmp;
  static HostMatrixModules* hostMatrixModule;

 protected:
  HostMatrixModules();
  virtual ~HostMatrixModules();

 public:
  static HostMatrixModules* GetInstance();
  HostMatrixAllocator* getMatrixAllocator();
  HostMatrixCopier* getMatrixCopier();
  HostMatrixUtils* getMatrixUtils();
  HostMatrixPrinter* getMatrixPrinter();
};

#endif /* MATRIXALLOCATOR_H */
