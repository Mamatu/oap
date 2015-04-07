/* 
 * File:   CuProcedures.h
 * Author: mmatula
 *
 * Created on August 17, 2014, 1:20 AM
 */

#ifndef OGLA_CU_ARNOLDIPROCEDURES_H
#define	OGLA_CU_ARNOLDIPROCEDURES_H

#include <vector>
#include "MatrixEx.h"
#include "Matrix.h"
#include "KernelExecutor.h"
#include "MatrixProcedures.h"

namespace ArnUtils {

bool SortLargestValues(const Complex& i, const Complex& j);

bool SortSmallestValues(const Complex& i, const Complex& j);

typedef bool (*SortType)(const Complex& i, const Complex& j);

}

class CuHArnoldi {
    void initVvector();

    bool continueProcedure();

    void calculateTriangularH();

    void calculateTriangularHEigens(uintt unwantedCount);

    bool executeArnoldiFactorization(bool init, intt initj,
        MatrixEx** dMatrixEx, floatt m_rho);
public:
    CuHArnoldi();
    virtual ~CuHArnoldi();

    void execute(math::Matrix* outputs, math::Matrix* hostA,
        uintt k, uintt wantedCount, floatt rho = 1. / 3.14);

    virtual void multiply(math::Matrix* w, math::Matrix* v);

    void setSortType(ArnUtils::SortType sortType);
    
protected:
    CuMatrix m_cuMatrix;
    math::Matrix* w;
    math::Matrix* f;
    math::Matrix* f1;
    math::Matrix* vh;
    math::Matrix* h;
    math::Matrix* s;
    math::Matrix* vs;
    math::Matrix* V;
    math::Matrix* transposeV;
    math::Matrix* V1;
    math::Matrix* V2;
    math::Matrix* H;
    math::Matrix* HC;
    math::Matrix* H1;
    math::Matrix* H2;
    math::Matrix* A1;
    math::Matrix* A2;
    math::Matrix* I;
    math::Matrix* v;
    math::Matrix* QT;
    math::Matrix* Q1;
    math::Matrix* Q2;
    math::Matrix* R1;
    math::Matrix* R2;
    math::Matrix* HO;
    math::Matrix* HO1;
    math::Matrix* Q;
    math::Matrix* QJ;
    math::Matrix* q;
    math::Matrix* q1;
    math::Matrix* q2;
    math::Matrix* GT;
    math::Matrix* G;
    math::Matrix* EV;
    math::Matrix* EV1;
    math::Matrix* EQ1;
    math::Matrix* EQ2;
    math::Matrix* EQ3;
private:
    math::Matrix* A;
    bool m_wasAllocated;
    uintt m_Acolumns;
    uintt m_Arows;
    uintt m_k;
    std::vector<Complex> wanted;
    std::vector<Complex> unwanted;
    std::vector<uintt> wantedIndecies;
    std::vector<Complex> notSorted;
    ArnUtils::SortType m_sortType;

    void* m_image;
    cuda::Kernel m_kernel;

    uintt m_transposeVcolumns;
    uintt m_hrows;
    uintt m_scolumns;
    uintt m_vscolumns;
    uintt m_vsrows;
    uintt m_vrows;
    uintt m_qrows;
    uintt m_Hcolumns;
    uintt m_Hrows;
    uintt m_H1columns;
    uintt m_Qrows;
    uintt m_Qcolumns;

    void alloc(math::Matrix* hostA, uintt m_k);
    void deallocIfNeeded(math::Matrix* hostA, uintt k);
    void dealloc1();
    void dealloc2();
    void dealloc3();
    void dealloc4();
    void copy(math::Matrix* hostA);
};

class CuHArnoldiCallback : public CuHArnoldi {
public:
    typedef void (*MultiplyFunc) (math::Matrix* w, math::Matrix* v, void* userData);

    void multiply(math::Matrix* w, math::Matrix* v);

    void setCallback(MultiplyFunc multiplyFunc, void* userData);

private:
    MultiplyFunc m_multiplyFunc;
    void* m_userData;
};

#endif	/* CUPROCEDURES_H */

