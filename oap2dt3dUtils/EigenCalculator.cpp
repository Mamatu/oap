#include "EigenCalculator.h"
#include "Exceptions.h"
#include "HostMatrixModules.h"
#include "DeviceMatrixModules.h"

namespace oap {

EigenCalculator::EigenCalculator()
    : m_eigenvalues(NULL), m_eigenvectors(NULL) {}

EigenCalculator::~EigenCalculator() {}

math::Matrix* EigenCalculator::createMatrix(
    const PngDataLoaders& pngDataLoaders) {
  const size_t refLength = pngDataLoaders[0]->getLength();
  floatt* floatsvec = new floatt[refLength];

  math::Matrix* hostMatrix =
      host::NewReMatrix(pngDataLoaders.size(), refLength);

  for (size_t fa = 0; fa < pngDataLoaders.size(); ++fa) {
    PngDataLoader* it = pngDataLoaders[fa];
    const size_t length = it->getLength();
    if (refLength != length) {
      delete[] floatsvec;
      host::DeleteMatrix(hostMatrix);
      throw oap::exceptions::NotIdenticalLengths(refLength, length);
    }
    it->getFloattVector(floatsvec);
    host::SetReVector(hostMatrix, fa, floatsvec, refLength);
  }

  delete[] floatsvec;

  return hostMatrix;
}

void EigenCalculator::addPngDataLoader(PngDataLoader* pngDataLoader) {
  m_internalPngDataLoaders.push_back(pngDataLoader);
}

void EigenCalculator::calculate() {}

math::Matrix* EigenCalculator::getLargestEigenValues() const {
  return m_eigenvalues;
}

math::Matrix* EigenCalculator::getLargestEigenVectors() const {
  return m_eigenvectors;
}

math::Matrix* EigenCalculator::createMatrix() const {
  return EigenCalculator::createMatrix(m_internalPngDataLoaders);
}

math::Matrix* EigenCalculator::createDeviceMatrix() const {
  math::Matrix* host = createMatrix();
  math::Matrix* device = device::NewDeviceMatrixCopy(host);
  host::DeleteMatrix(host);
  return device;
}
}
