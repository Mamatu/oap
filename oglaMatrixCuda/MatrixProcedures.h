#ifndef OGLA_CUMATRIXPROCEDURES_H
#define	OGLA_CUMATRIXPROCEDURES_H

#include "Matrix.h"
#include "MatrixEx.h"
#include "CudaUtils.h"

class CuMatrix {
public:
    CuMatrix();
    virtual ~CuMatrix();

    void dotProduct(math::Matrix* ouput,
        math::Matrix* params0, math::Matrix* params1);

    void dotProductEx(math::Matrix* ouput,
        math::Matrix* params0, math::Matrix* params1,
        MatrixEx* matrixEx);

    void transposeMatrixEx(math::Matrix* output,
        math::Matrix* params0, MatrixEx* matrixEx);

    void transposeMatrix(math::Matrix* output,
        math::Matrix* params0);

    void substract(math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1);

    void addMatrix(math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1);

    void setVector(math::Matrix* output, uintt column,
        math::Matrix* params0, uintt length);

    void getVector(math::Matrix* vector, uintt length,
        math::Matrix* matrix, uintt column);

    void magnitude(floatt& output, math::Matrix* params0);

    void multiplyConstantMatrix(math::Matrix* v,
        math::Matrix* f, floatt re);

    void multiplyConstantMatrix(math::Matrix* v,
        math::Matrix* f, floatt re, floatt im);

    void setDiagonal(math::Matrix* matrix, floatt re, floatt im);

    void setIdentity(math::Matrix* matrix);

    void setZeroMatrix(math::Matrix* matrix);

    bool compare(math::Matrix* matrix1, math::Matrix* matrix2);

    void QR(math::Matrix* Q,
        math::Matrix* R, math::Matrix* H,
        math::Matrix* R1, math::Matrix* Q1,
        math::Matrix* G, math::Matrix * GT);
    
    CUresult getStatus() const;
    
private:
    CUresult m_cuResult;
    floatt* m_magniuteOutput;
    uintt* m_dcompareOutput;

    template<typename T> class CuBuffer {
    public:
        T* m_buffer;
        uintt m_size;

        CuBuffer();
        virtual ~CuBuffer();

        void realloc(uintt size);
    };

    CuBuffer<int> m_compareBuffer;
    CuBuffer<floatt> m_magnitudeBuffer;
};

template<typename T> CuMatrix::CuBuffer<T>::CuBuffer() :
m_buffer(NULL),
m_size(0) {
    // not implemented
}

template<typename T> CuMatrix::CuBuffer<T>::~CuBuffer() {
    if (m_buffer != NULL) {
        CudaUtils::FreeDeviceMem(m_buffer);
    }
}

template<typename T> void CuMatrix::CuBuffer<T>::realloc(uintt size) {
    if (size > m_size) {
        if (m_buffer != NULL) {
            CudaUtils::FreeDeviceMem(m_buffer);
        }
        m_buffer = static_cast<T*> (CudaUtils::AllocDeviceMem(size));
        m_size = size;
    }
}



#endif	/* MATRIXPROCEDURES_H */

