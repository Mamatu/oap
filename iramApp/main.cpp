#include <cstdlib>
#include <math.h>
#include "HostMatrixModules.h"
#include "MathOperationsCpu.h"

#define PRINT_INFO() fprintf(stderr,"%s %s : %d", __FUNCTION__,__FILE__,__LINE__);

#define PRINT_INFO_1(b) fprintf(stderr,"%s %s : %d value == %f \n", __FUNCTION__,__FILE__,__LINE__,b);

using namespace std;

floatt dotProduct(math::Matrix* matrix1, int column1, math::Matrix* matrix2, int column2) {
    floatt output = 0;
    for (intt fa = 0; fa < matrix1->rows; fa++) {
        output += matrix1->reValues[column1 + matrix1->columns * fa] * matrix2->reValues[column2 + matrix2->columns * fa];
    }
    return output;
}

floatt getLength2(math::Matrix* matrix, int column) {
    floatt output = 0;
    for (intt fa = 0; fa < matrix->rows; fa++) {
        output += matrix->reValues[column + matrix->columns * fa] * matrix->reValues[column + matrix->columns * fa];
    }
    return output;
}

floatt getLength(math::Matrix* matrix, int column) {
    floatt output = getLength2(matrix, column);
    return sqrt(output);
}

void cast(math::Matrix* proj, int column, math::Matrix* a, int column1, math::Matrix* e, int column2) {
    floatt value = dotProduct(a, column1, e, column2);
    floatt len = getLength(e, column2);
    host::PrintReMatrix(e);
    for (intt fa = 0; fa < proj->rows; fa++) {
        proj->reValues[column + proj->columns * fa] = e->reValues[column2 + e->columns * fa] * value / len;
    }
}

void normalize(math::Matrix* output, int column1, math::Matrix* matrix, int column2) {
    floatt len = getLength(matrix, column2);
    for (intt fa = 0; fa < matrix->rows; fa++) {
        output->reValues[column1 + output->columns * fa] = matrix->reValues[column2 + matrix->columns * fa] / len;
    }
}

void normalize(math::Matrix* output, int column1) {
    normalize(output, column1, output, column1);
}

void qrDecomposition(math::Matrix* A, math::Matrix* q1, math::Matrix* r1) {
    math::Matrix* u = host::NewMatrixCopy(A);
    math::Matrix* tu = host::NewMatrixCopy(A);
    math::Matrix* e = host::NewMatrixCopy(A);
    math::Matrix* proj = host::NewMatrixCopy(A);
    intt length = A->columns;
    normalize(e, 0, u, 0);
    for (intt fa = 1; fa < u->columns; fa++) {
        for (intt fb = 0; fb < fa - 1; fb++) {
            cast(proj, fb, A, fa, e, fb);
        }
        for (intt fb = 0; fb < fa; fb++) {
            cast(proj, fb, A, fa, e, fb);
            for (intt fc = 0; fc < u->rows; fc++) {
                u->reValues[fa + u->columns * fc] = u->reValues[fa + u->columns * fc] - proj->reValues[fb + proj->columns * fc];
            }
        }
        normalize(e, fa, u, fa);
    }
    math::Matrix* q = host::NewMatrixCopy(u);
    math::Matrix* tq = host::NewMatrixCopy(u);
    math::Matrix* r = host::NewMatrixCopy(u);
    host::PrintReMatrix("u2=", u);
    for (intt fa = 0; fa < u->columns; fa++) {
        normalize(q, fa, u, fa);
    }
    math::cpu::TransposeOperation to;
    to.setMatrix(q);
    to.setOutputMatrix(tq);
    to.start();
    math::cpu::DotProductOperation dpo;
    dpo.setMatrix1(tq);
    dpo.setMatrix2(A);
    dpo.setOutputMatrix(r);
    dpo.start();
    host::PrintReMatrix("A=", A);
    host::PrintReMatrix("q=", q);
    host::PrintReMatrix("r=", r);
    host::Copy(q1, q);
    host::Copy(r1, r);
    host::DeleteMatrix(u);
    host::DeleteMatrix(q);
    host::DeleteMatrix(r);
    host::DeleteMatrix(tu);
    host::DeleteMatrix(e);
    host::DeleteMatrix(proj);
}

floatt getLength2(math::Matrix * matrix) {
    intt length = matrix->columns * matrix->rows;
    floatt value = 0;
    for (intt fa = 0; fa < length; fa++) {
        value += matrix->reValues[fa] * matrix->reValues[fa];
    }
    return value;
}

floatt getLength(math::Matrix * matrix) {
    floatt output = getLength(matrix);
    return sqrt(output);
}

class IraMethod {
public:
    math::cpu::MathOperations mo;
    math::Matrix* w;
    math::Matrix* f;
    math::Matrix* vh;
    math::Matrix* h;
    math::Matrix* s;
    math::Matrix* vs;
    math::Matrix* V;
    math::Matrix* V1;
    math::Matrix* transposeV;
    math::Matrix* H;
    math::Matrix* H1;
    math::Matrix* I;
    math::Matrix* A;
    math::Matrix* v;
    math::Matrix* Q1T;
    math::Matrix* Q1;
    math::Matrix* R1;
    math::Matrix* HO;
    math::Matrix* HO1;
    math::Matrix* Q;
    math::Matrix* QJ;
    intt k;
    floatt rho;

    IraMethod(math::Matrix* A, intt k, floatt rho) {
        w = host::NewReMatrix(1, A->rows, 0);
        v = host::NewReMatrix(1, A->rows, 0);
        f = host::NewReMatrix(1, A->rows, 0);
        vh = host::NewReMatrix(1, A->rows, 0);
        h = host::NewReMatrix(1, k, 0);
        s = host::NewReMatrix(1, k, 0);
        vs = host::NewReMatrix(1, A->rows, 0);
        V = host::NewReMatrix(k, A->rows, 0);
        V1 = host::NewReMatrix(k, A->rows, 0);
        v->reValues[0] = 1;
        V->reValues[0] = 1;
        transposeV = host::NewReMatrix(A->rows, k, 0);
        H = host::NewReMatrix(k, k, 0);
        Q = host::NewReMatrix(k, k, 0);
        Q1 = host::NewReMatrix(k, k, 0);
        Q1T = host::NewReMatrix(k, k, 0);
        R1 = host::NewReMatrix(k, k, 0);
        QJ = host::NewReMatrix(k, k, 0);
        HO = host::NewReMatrix(k, k, 0);
        HO1 = host::NewReMatrix(k, k, 0);
        H1 = host::NewReMatrix(k, k, 0);
        I = host::NewReMatrix(k, k, 0);
        this->A = A;
        this->k = k;
        this->rho = rho;
    }

    ~IraMethod() {
        host::DeleteMatrix(w);
        host::DeleteMatrix(f);
        host::DeleteMatrix(vh);
        host::DeleteMatrix(h);
        host::DeleteMatrix(s);
        host::DeleteMatrix(vs);
        host::DeleteMatrix(V);
        host::DeleteMatrix(H);
        host::DeleteMatrix(H1);
        host::DeleteMatrix(I);
        host::DeleteMatrix(v);
        host::DeleteMatrix(transposeV);
    }

    void start() {
        this->executeArnoldiFactorization();
        std::vector<floatt> unwanted;
        std::vector<floatt> wanted;
        this->selection(H, unwanted, wanted);
        host::SetIdentity(Q);
        host::SetIdentity(QJ);
        debugFunc();
        for (intt fa = 0; fa < unwanted.size(); fa++) {
            debugFunc();
            math::Matrix* temp = Q;
            Q = QJ;
            QJ = temp;
            host::SetIdentity(I);
            mo.multiply(I, I, &unwanted[fa]);
            mo.substract(I, H, I);
            math::Matrix* q = NULL;
            math::Matrix* r = NULL;
            qrDecomposition(I, Q1, R1);
            mo.transpose(Q1T, Q1);
            mo.dotProduct(HO, H, Q1);
            mo.dotProduct(HO1, Q1T, HO);
            mo.dotProduct(V1, V, Q1);
            mo.dotProduct(Q, QJ, Q1);
            debugFunc();
        }
        debugFunc();
        H = HO1;
        // floatt qk = Q->reValues[Q->columns * (Q->rows - 1) + wanted.size()];
        // floatt Bk = H->reValues[Q->columns * (wanted.size() + 1) + wanted.size()];
    }

protected:
    void executeArnoldiFactorization();
    void selection(math::Matrix* H, std::vector<floatt>& unwanted, std::vector<floatt>& wanted);
};

void IraMethod::executeArnoldiFactorization() {
    HostMatrixPrinter hmp;
    mo.dotProduct(w, A, v);
    host::SetReVector(V, 0, v->reValues, v->rows);
    mo.setSubRows(0, 1);
    mo.transpose(transposeV, V);
    mo.setSubColumns(0, 1);
    mo.dotProduct(h, transposeV, w);
    mo.dotProduct(vh, V, h);
    mo.substract(f, w, vh);
    floatt mf = 0;
    floatt mh = 0;
    floatt B = 0;
    debug("-------------------------------------------\n");
    for (intt j = 0; j < k - 1; j++) {
        mo.magnitude(&B, f);
        floatt rB = 1. / B;
        mo.multiply(v, f, &rB);
        //host::PrintReMatrix("v=", v);
        host::SetReVector(V, j + 1, v->reValues, v->rows);
        H->reValues[j + H->columns * j] = B;
        mo.dotProduct(w, A, v);
        mo.setSubRows(0, j + 1);
        mo.transpose(transposeV, V);
        mo.dotProduct(h, transposeV, w);
        mo.dotProduct(vh, V, h);
        mo.magnitude(&mf, f);
        mo.magnitude(&mh, h);
        if (mf < rho * mh) {
            mo.dotProduct(s, transposeV, f);
            mo.setSubColumns(0, s->rows);
            mo.dotProduct(vs, V, s);
            mo.substract(f, f, vs);
            mo.add(h, h, s);
        }
        host::SetReVector(H, j, h->reValues, j);
        debug("-------------------------------------------\n");
    }
}

void IraMethod::selection(math::Matrix* H, std::vector<floatt>& unwanted, std::vector<floatt>& wanted) {
    debugFuncBegin();
    floatt max = 0;
    bool isfirst = true;
    for (intt fa = 0; fa < H->rows; fa++) {
        floatt value = H->reValues[fa * H->columns + fa];
        if (max < value) {
            if (isfirst == false) {
                unwanted.push_back(max);
            } else {
                isfirst = false;
            }
            max = value;
        } else {
            unwanted.push_back(value);
        }
    }
    wanted.push_back(max);
    debugFuncEnd();
}

void test() {
    HostMatrixAllocator hmm;
    math::cpu::MathOperations mo;
    HostMatrixUtils hmu;
    HostMatrixPrinter hmp;
    floatt arrayA[] = {12, -51, 4,
        6, 167, -68,
        -4, 24, -41};
    math::Matrix* a = host::NewReMatrixCopy(3, 3, arrayA);

    floatt arrayB[] = {1, 0, 0,
        0, 1, 0,
        0, 0, 1};

    floatt arrayC[] = {2, 2, 2, 2, 2, 2, 2, 2, 2};
    math::Matrix* b = host::NewReMatrixCopy(3, 3, arrayB);
    math::Matrix* c = host::NewReMatrixCopy(3, 3, arrayC);
    math::Matrix* o = host::NewReMatrixCopy(3, 3, 0);
    host::PrintReMatrix(o);
}

int main(int argc, char** argv) {
    floatt arrayA[] = {12, -51, 4,
        6, 167, -68,
        -4, 24, -41};
    math::Matrix* A = host::NewReMatrixCopy(3, 3, arrayA);
    //math::Matrix* q = host::NewReMatrixCopy(3, 3, arrayA);
    //math::Matrix* r = host::NewReMatrixCopy(3, 3, arrayA);
    //qrDecomposition(A, q, r);
    IraMethod iram(A, 3, 1);
    iram.start();
    host::PrintReMatrix(iram.V);
    host::PrintReMatrix(iram.H);
    //floatt* vector = new floatt[iram.V->rows];
    math::cpu::MathOperations mo;
    floatt value = 0;
    return 0;
}
