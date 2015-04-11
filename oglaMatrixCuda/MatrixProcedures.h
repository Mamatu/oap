#ifndef OGLA_CUMATRIXPROCEDURES_H
#define OGLA_CUMATRIXPROCEDURES_H

#include "Matrix.h"
#include "MatrixEx.h"
#include "CudaUtils.h"
#include "KernelExecutor.h"

class CuMatrix {
 public:
  CuMatrix();
  virtual ~CuMatrix();

  void dotProduct(math::Matrix* ouput, math::Matrix* params0,
                  math::Matrix* params1);

  void dotProductEx(math::Matrix* ouput, math::Matrix* params0,
                    math::Matrix* params1, MatrixEx* matrixEx);

  void dotProductOpt(math::Matrix* ouput, math::Matrix* params0,
                     math::Matrix* params1);

  void dotProductExOpt(math::Matrix* ouput, math::Matrix* params0,
                       math::Matrix* params1, MatrixEx* matrixEx);

  void transposeMatrixEx(math::Matrix* output, math::Matrix* params0,
                         MatrixEx* matrixEx);

  void transposeMatrix(math::Matrix* output, math::Matrix* params0);

  void substract(math::Matrix* output, math::Matrix* params0,
                 math::Matrix* params1);

  void addMatrix(math::Matrix* output, math::Matrix* params0,
                 math::Matrix* params1);

  void setVector(math::Matrix* output, uintt column, math::Matrix* params0,
                 uintt length);

  void getVector(math::Matrix* vector, uintt length, math::Matrix* matrix,
                 uintt column);

  void magnitude(floatt& output, math::Matrix* params0);

  void magnitudeOpt(floatt& output, math::Matrix* params0);

  void magnitudeOptVer2(floatt& output, math::Matrix* params0);

  void magnitude2(floatt& output, math::Matrix* params0);

  void magnitude2Opt(floatt& output, math::Matrix* params0);

  void magnitude2OptVer2(floatt& output, math::Matrix* params0);

  void multiplyConstantMatrix(math::Matrix* v, math::Matrix* f, floatt re);

  void multiplyConstantMatrix(math::Matrix* v, math::Matrix* f, floatt re,
                              floatt im);

  void setDiagonal(math::Matrix* matrix, floatt re, floatt im);

  void setIdentity(math::Matrix* matrix);

  void setZeroMatrix(math::Matrix* matrix);

  bool compare(math::Matrix* matrix1, math::Matrix* matrix2);

  bool compareVer2(math::Matrix* matrix1, math::Matrix* matrix2);

  uintt getCompareOperationSum() const;

  void QR(math::Matrix* Q, math::Matrix* R, math::Matrix* H, math::Matrix* R1,
          math::Matrix* Q1, math::Matrix* G, math::Matrix* GT);

  CUresult getStatus() const;

 private:
  bool compareProcedure(const char* cuKernelName, math::Matrix* matrix1,
                        math::Matrix* matrix2, uintt w, uintt h, uintt wthreads,
                        uintt hthreads);

  floatt magnitude2Procedure(const char* cuKernelName, math::Matrix* matrix1,
                             uintt wthreads, uintt hthreads);

  CUresult m_cuResult;
  floatt* m_magniuteOutput;

  enum Type { HOST, CUDA };

  template <typename T>
  class Buffer {
   public:
    T* m_buffer;
    uintt m_size;

    Buffer(CuMatrix::Type type);
    ~Buffer();

    void realloc(uintt size);

   private:
    Type m_type;
    void free(T* buffer);
    T* alloc(uintt size);
  };

  Buffer<int> m_dcompareOutputBuffer;
  Buffer<int> m_dcompareBuffer;
  Buffer<int> m_hcompareOutputBuffer;
  Buffer<floatt> m_magnitudeBuffer;
  Buffer<floatt> m_dmagnitudeOutputBuffer;
  Buffer<floatt> m_dmagnitudeBuffer;
  Buffer<floatt> m_hmagnitudeOutputBuffer;

  template <typename T1>
  friend class Buffer;

 private:
  void init();
  CUresult execute(const char* functionName, uintt w, uintt h, void** params,
                   uintt sharedMemory);

  bool m_isIntialized;
  cuda::Kernel m_kernel;
  const char* m_pathes[3];
  void* m_image;
  uintt m_maxThreadsPerBlock;
  uintt m_compareOperationOutput;
  CuMatrix(const CuMatrix&);
};

template <typename T>
CuMatrix::Buffer<T>::Buffer(CuMatrix::Type type)
    : m_buffer(NULL), m_size(0), m_type(type) {
  // not implemented
}

template <typename T>
CuMatrix::Buffer<T>::~Buffer() {
  if (m_buffer != NULL && m_type == CUDA) {
    free(m_buffer);
  }
}

template <typename T>
void CuMatrix::Buffer<T>::realloc(uintt size) {
  if (size > m_size) {
    if (m_buffer != NULL) {
      free(m_buffer);
    }
    m_buffer = alloc(size);
    m_size = size;
  }
}

template <typename T>
void CuMatrix::Buffer<T>::free(T* buffer) {
  if (m_type == CuMatrix::CUDA) {
    CudaUtils::FreeDeviceMem(m_buffer);
  } else if (m_type == CuMatrix::HOST) {
    delete[] buffer;
  }
}

template <typename T>
T* CuMatrix::Buffer<T>::alloc(uintt size) {
  switch (m_type) {
    case CuMatrix::CUDA:
      return static_cast<T*>(CudaUtils::AllocDeviceMem(size));
    case CuMatrix::HOST:
      return new T[size];
  };
}

#endif /* MATRIXPROCEDURES_H */
