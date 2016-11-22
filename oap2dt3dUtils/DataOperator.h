#ifndef DATAOPERATOR_H
#define DATAOPERATOR_H

#include <vector>

#include "PngDataLoader.h"
#include "Matrix.h"

namespace oap {

typedef std::vector<PngDataLoader*> PngDataLoaders;

class DataOperator {
 public:
  DataOperator();
  virtual ~DataOperator();

  static math::Matrix* createMatrix(const PngDataLoaders& pngDataLoaders);

  void addPngDataLoader(PngDataLoader* pngDataLoader);

  math::Matrix* createMatrix() const;

  math::Matrix* createDeviceMatrix() const;

 private:
  PngDataLoaders m_internalPngDataLoaders;
};
}

#endif  // DATAOPERATOR_H
