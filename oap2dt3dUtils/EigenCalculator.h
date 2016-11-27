#ifndef DATAOPERATOR_H
#define DATAOPERATOR_H

#include <vector>

#include "PngDataLoader.h"
#include "Matrix.h"
#include "ArnoldiProcedures.h"

namespace oap {

typedef std::vector<PngDataLoader*> PngDataLoaders;

class EigenCalculator {
 public:
  EigenCalculator();
  virtual ~EigenCalculator();

  static math::Matrix* createMatrix(const PngDataLoaders& pngDataLoaders);

  void addPngDataLoader(PngDataLoader* pngDataLoader);

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
  PngDataLoaders m_pngDataLoaders;
  CuHArnoldi* m_cuHArnoldi;
};
}

#endif  // DATAOPERATOR_H
