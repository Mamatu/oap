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

  math::Matrix* getLargestEigenValues() const;

  math::Matrix* getLargestEigenVectors() const;

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

 private:
  math::Matrix* m_eigenvalues;
  math::Matrix* m_eigenvectors;
  PngDataLoaders m_internalPngDataLoaders;
  CuHArnoldi* m_cuHArnoldi;
};
}

#endif  // DATAOPERATOR_H
