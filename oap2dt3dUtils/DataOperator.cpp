#include "DataOperator.h"
#include "Exceptions.h"
#include "HostMatrixModules.h"
#include "DeviceMatrixModules.h"

namespace oap {

DataOperator::DataOperator() {}

DataOperator::~DataOperator() {}

math::Matrix* DataOperator::createMatrix(const PngDataLoaders& pngDataLoaders) {
  const size_t length = pngDataLoaders[0]->getLength();
  floatt* pixels = new floatt[length];

  math::Matrix* hostMatrix = host::NewReMatrix(pngDataLoaders.size(), length);

  for (size_t fa = 0; fa < pngDataLoaders.size(); ++fa) {
    PngDataLoader* it = pngDataLoaders[fa];
    const size_t length1 = it->getLength();
    if (length != length1) {
      delete[] pixels;
      host::DeleteMatrix(hostMatrix);
      throw oap::exceptions::NotIdenticalLengths(length, length1);
    }
    it->getFloattVector(pixels);
    host::SetReVector(hostMatrix, fa, pixels, length);
  }

  delete[] pixels;

  return hostMatrix;
}

void DataOperator::addPngDataLoader(PngDataLoader* pngDataLoader) {
  m_internalPngDataLoaders.push_back(pngDataLoader);
}

math::Matrix* DataOperator::createMatrix() const {
  return DataOperator::createMatrix(m_internalPngDataLoaders);
}

math::Matrix* DataOperator::createDeviceMatrix() const {
  math::Matrix* host = createMatrix();
  math::Matrix* device = device::NewDeviceMatrixCopy(host);
  host::DeleteMatrix(host);
  return device;
}
}
