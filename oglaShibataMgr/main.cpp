/* 
 * File:   main.cpp
 * Author: mmatula
 *
 * Created on November 5, 2013, 7:19 PM
 */

#include <cstdlib>
#include "RealTransferMatrixCpu.h"
#include "RealTransferMatrixCuda.h"
#include "HostMatrixModules.h"
#include "DeviceMatrixModules.h"
#include "Matrix.h"
#include "KernelExecutor.h"
#include "main.h"

#include <cstdlib>
#include <math.h>
#include <algorithm>
#include <vector>
#include "HostMatrixModules.h"
#include "MathOperationsCpu.h"


#define THREADS_COUNT 4

#define PRINT_INFO() fprintf(stderr,"%s %s : %d", __FUNCTION__,__FILE__,__LINE__);

#define PRINT_INFO_1(b) fprintf(stderr,"%s %s : %d value == %f \n", __FUNCTION__,__FILE__,__LINE__,b);


using namespace std;
math::cpu::MathOperations matrixOperations;

void setPauliMatrixX(math::Matrix* matrix) {
    matrix->reValues[0] = 0;
    matrix->reValues[1] = 1;
    matrix->reValues[2] = 1;
    matrix->reValues[3] = 0;
}

void setPauliMatrixY(math::Matrix* matrix) {
    matrix->reValues[0] = 0;
    matrix->reValues[1] = -1;
    matrix->reValues[2] = 1;
    matrix->reValues[3] = 0;
}

void setPauliMatrixZ(math::Matrix* matrix) {
    matrix->reValues[0] = 1;
    matrix->reValues[1] = 0;
    matrix->reValues[2] = 0;
    matrix->reValues[3] = -1;
}

inline int pow(int a, int b) {
    int o = 1;
    for (int fa = 0; fa < b; fa++) {
        o = o*a;
    }
    return o;
}

math::Matrix* createHamiltonian(floatt J, math::Matrix* spin1, math::Matrix* spin2, math::Matrix* identity, math::Matrix* tempMatrix1) {
    HostMatrixAllocator mhm;
    math::Matrix* tempMatrix2 = mhm.newReMatrix(4, 4);
    matrixOperations.tensorProduct(tempMatrix1, spin1, identity);
    matrixOperations.tensorProduct(tempMatrix2, identity, spin2);
    matrixOperations.add(tempMatrix2, tempMatrix1, tempMatrix2);
    return tempMatrix2;
}

void qrtest() {
    floatt a[] = {12, -51, 4, 6, 167, -68, -4, 24, -41};
    math::Matrix* A = host::NewReMatrixCopy(3, 3, a);
    math::Matrix* q = host::NewMatrixCopy(A);
    math::Matrix* r = host::NewMatrixCopy(A);
    //qrDecomposition(A, q, r);
}

void qtest() {
    floatt aA[] = {0, 1, 1, 1, 0, 1, 1, 1, 0};
    math::cpu::MathOperations mo;
    math::Matrix* A = host::NewReMatrixCopy(3, 3, (floatt*) aA);
    math::Matrix* m = host::NewReMatrix(3, 3, 0);
    //host::SetDiagonals(m, 2);
    mo.substract(A, A, m);
    floatt d;
    mo.det(&d, A);
    fprintf(stderr, "f == %f \n", d);
    math::cpu::IraMethod iram(&mo);
    iram.setMatrix(A);
    iram.setHSize(4);
    iram.setRho(1. / 3.14);
    floatt output[2];
    iram.setReOutputValues(output, 2);
    iram.start();
}

floatt* reoutpus = NULL;
floatt* recount = 0;

void Callback_f(int event, void* object, void* userPtr) {
    if (event == math::cpu::IraMethodCallback::EVENT_MATRIX_MULTIPLICATION) {
        shibata::cpu::RealTransferMatrix* transferMatrixCpu =
                reinterpret_cast<shibata::cpu::RealTransferMatrix*> (userPtr);
        math::cpu::IraMethodCallback::Event* event =
                reinterpret_cast<math::cpu::IraMethodCallback::Event*> (object);

#if 0
        if (reoutpus == NULL) {
            reoutpus = new floatt[event->getCount()];
            recount = event->getCount();
        } else if (recount < event->getCount()) {
            delete[] reoutpus;
            reoutpus = new floatt[event->getCount()];
            recount = event->getCount();
        }
#endif

        transferMatrixCpu->setEntries(event->getMatrixEntries(),
                event->getCount());
        transferMatrixCpu->setReOutputEntries(event->getReOutputs());
        transferMatrixCpu->setImOutputEntries(event->getImOutputs());
        transferMatrixCpu->start();
    }
}

int main1(int argc, char** argv) {
    //qtest();
    //return 0;
    //cuda::Context context(1);
    //context.init();
    HostMatrixAllocator hmm;
    HostMatrixUtils mu;
    math::cpu::MathOperations mo;
    shibata::cpu::RealTransferMatrix transferMatrixCpu;
    shibata::cuda::RealTransferMatrix transferMatrixCuda;
    math::Matrix* identity = hmm.newReMatrix(2, 2);
    mu.setIdentityReMatrix(identity);
    math::Matrix* spin1 = hmm.newReMatrix(2, 2);
    math::Matrix* spin2 = hmm.newReMatrix(2, 2);
    math::Matrix* spin3 = hmm.newReMatrix(2, 2);
    math::Matrix* spin4 = hmm.newReMatrix(2, 2);
    math::Matrix* temp = hmm.newReMatrix(4, 4);

    setPauliMatrixZ(spin1);
    setPauliMatrixZ(spin3);
    setPauliMatrixZ(spin4);

    floatt b[16] = {0.286505, 0.000000, 0.000000, 0.000000,
        0.000000, 21.403793, -21.117289, 0.000000,
        0.000000, -21.117289, 21.403793, 0.000000,
        0.000000, 0.000000, 0.000000, 0.286505};

    floatt b1[16] = {0.0, 0.000000, 0.000000, 0.000000,
        0.000000, 0, -21.117289, 0.000000,
        0.000000, -21.117289, 0, 0.000000,
        0.000000, 0.000000, 0.000000, 0.0};

    math::Matrix* h1 = host::NewMatrixCopy(4, 4, (floatt*) b, (floatt*) b1);
    math::Matrix* h2 = host::NewMatrixCopy(4, 4, (floatt*) b, (floatt*) b1);
    math::Matrix* eh1 = host::NewMatrixCopy(4, 4, (floatt*) b, (floatt*) b1);
    math::Matrix* eh2 = host::NewMatrixCopy(4, 4, (floatt*) b, (floatt*) b1);

    //math::Matrix* h1 = host::NewReMatrixCopy(4, 4, (floatt*) b);
    //math::Matrix* h2 = host::NewReMatrixCopy(4, 4, (floatt*) b);
    //math::Matrix* eh1 = host::NewReMatrixCopy(4, 4, (floatt*) b);
    //math::Matrix* eh2 = host::NewReMatrixCopy(4, 4, (floatt*) b);


    transferMatrixCpu.PrepareHamiltonian(eh1, h1, shibata::ORIENTATION_REAL_DIRECTION);
    transferMatrixCpu.PrepareHamiltonian(eh2, h2, shibata::ORIENTATION_REAL_DIRECTION);

    int threadsCount = 4;
    int spinsCount = 3; //25;
    int q[] = {-1, 1};
    int qCount = 2;
    int serieLimit = 25;
    int M = 5;
    int l = pow(qCount, 2 * M);
    fprintf(stderr, "l = %llu\n", l);
    math::Matrix* transferMatrix = hmm.newMatrix(l, l);

    //csetReOutputEntries(reoutputs);
    //transferMatrixCpu.setEntries(entries, count);
    transferMatrixCpu.setOutputMatrix(transferMatrix);
    transferMatrixCpu.setThreadsCount(threadsCount);
    transferMatrixCpu.setSerieLimit(serieLimit);
    transferMatrixCpu.setSpinsCount(spinsCount);
    transferMatrixCpu.setQuantums(q);
    transferMatrixCpu.setQuantumsCount(qCount);
    transferMatrixCpu.setTrotterNumber(M);
    transferMatrixCpu.setExpHamiltonian1(eh1);
    transferMatrixCpu.setExpHamiltonian2(eh2);
    math::Status status = transferMatrixCpu.start();
    //transferMatrixCpu.setOutputMatrix(NULL);

    math::cpu::IraMethod/*Callback */iram(&mo/*, 16 * 3*/);
    iram.setHSize(4);
    iram.setRho(1. / 3.14);
    iram.setMatrix(transferMatrix);
    iram.setThreadsCount(THREADS_COUNT);
    //iram.registerCallback(Callback_f, &transferMatrixCpu);
    uintt eigenvaluesCount = 2;
    floatt revalues[eigenvaluesCount];
    floatt imvalues[eigenvaluesCount];
    memset(revalues, 0, eigenvaluesCount * sizeof (floatt));
    memset(imvalues, 0, eigenvaluesCount * sizeof (floatt));
    iram.setReOutputValues(revalues, eigenvaluesCount);
    iram.setImOutputValues(imvalues, eigenvaluesCount);
    iram.start();
    for (uintt fa = 0; fa < eigenvaluesCount; fa++) {
        fprintf(stderr, "eigenvalue = %f\n", revalues[fa]);
        fprintf(stderr, "eigenvalue = %f\n", imvalues[fa]);
    }
    return 0;
}

int main2(int argc, char** argv) {
    HostMatrixAllocator hmm;
    HostMatrixUtils mu;
    math::cpu::MathOperations mo;
    shibata::cpu::RealTransferMatrix transferMatrixCpu;
    shibata::cuda::RealTransferMatrix transferMatrixCuda;
    math::Matrix* identity = hmm.newReMatrix(2, 2);
    mu.setIdentityReMatrix(identity);
    math::Matrix* spin1 = hmm.newMatrix(2, 2);
    math::Matrix* spin2 = hmm.newMatrix(2, 2);
    math::Matrix* spin3 = hmm.newMatrix(2, 2);
    math::Matrix* spin4 = hmm.newMatrix(2, 2);
    math::Matrix* temp = hmm.newReMatrix(4, 4);

    setPauliMatrixZ(spin1);
    setPauliMatrixZ(spin3);
    setPauliMatrixZ(spin4);

    //floatt b[16] = {0.606531, 0.000000, 0.000000, 0.000000,
    //    0.000000, 2.544110, -1.937579, 0.000000,
    //    0.000000, -1.937579, 2.544110, 0.000000,
    //    0.000000, 0.000000, 0.000000, 0.606531};


    floatt b[16] = {0.731616, 0.000000, 0.000000, 0.000000,
        0.000000, 1.642603, -0.910987, 0.000000,
        0.000000, -0.910987, 1.642603, 0.000000,
        0.000000, 0.000000, 0.000000, 0.731616};

    floatt b1[16] = {0.434598, 0.000000, 0.000000, 0.000000,
        0.000000, 6.308546, -5.873948, 0.000000,
        0.000000, -5.873948, 6.308546, 0.000000,
        0.000000, 0.000000, 0.000000, 0.434598};


    floatt b2[16] = {0, 0.000000, 0.000000, 0.000000,
        0.000000, 0, -0.910987, 0.000000,
        0.000000, -0.910987, 0, 0.000000,
        0.000000, 0.000000, 0.000000, 0};

    floatt z[16] = {0, 0.000000, 0.000000, 0.000000,
        0.000000, 0, 0, 0.000000,
        0.000000, 0, 0, 0.000000,
        0.000000, 0.000000, 0.000000, 0};

    //math::Matrix* h1 = host::NewReMatrixCopy(4, 4, (floatt*) b1);
    //math::Matrix* h2 = host::NewReMatrixCopy(4, 4, (floatt*) b1);
    //math::Matrix* eh1 = host::NewReMatrixCopy(4, 4, (floatt*) b1);
    //math::Matrix* eh2 = host::NewReMatrixCopy(4, 4, (floatt*) b1);

    //math::Matrix* h1 = host::NewReMatrixCopy(4, 4, (floatt*) b);
    //math::Matrix* h2 = host::NewReMatrixCopy(4, 4, (floatt*) b);
    //math::Matrix* eh1 = host::NewReMatrixCopy(4, 4, (floatt*) b);
    //math::Matrix* eh2 = host::NewReMatrixCopy(4, 4, (floatt*) b);


    floatt ab[16] = {0.286505, 0.000000, 0.000000, 0.000000,
        0.000000, 21.403793, -21.117289, 0.000000,
        0.000000, -21.117289, 21.403793, 0.000000,
        0.000000, 0.000000, 0.000000, 0.286505};
    
    math::Matrix* h1 = host::NewMatrixCopy(4, 4, (floatt*) ab, (floatt*) z);
    math::Matrix* h2 = host::NewMatrixCopy(4, 4, (floatt*) ab, (floatt*) z);
    math::Matrix* eh1 = host::NewMatrixCopy(4, 4, (floatt*) ab, (floatt*) z);
    math::Matrix* eh2 = host::NewMatrixCopy(4, 4, (floatt*) ab, (floatt*) z);

    //math::Matrix* h1 = host::NewMatrixCopy(4, 4, (floatt*) b, (floatt*) z);
    //math::Matrix* h2 = host::NewMatrixCopy(4, 4, (floatt*) b, (floatt*) z);
    //math::Matrix* eh1 = host::NewMatrixCopy(4, 4, (floatt*) b, (floatt*) z);
    //math::Matrix* eh2 = host::NewMatrixCopy(4, 4, (floatt*) b, (floatt*) z);

    transferMatrixCpu.PrepareHamiltonian(eh1, h1, shibata::ORIENTATION_REAL_DIRECTION);
    transferMatrixCpu.PrepareHamiltonian(eh2, h2, shibata::ORIENTATION_REAL_DIRECTION);

    int threadsCount = 4;
    int spinsCount = 3; //25;
    int q[] = {-1, 1};
    int qCount = 2;
    int serieLimit = 25;
    int M = 2;
    int l = pow(qCount, 2 * M);
    fprintf(stderr, "l = %llu\n", l);
    math::Matrix* transferMatrix = hmm.newMatrix(1, l);

    uintt realCount = 2048;
    floatt* imoutputs = new floatt[realCount];
    floatt* reoutputs = new floatt[realCount];
    uintt* entries = new uintt[2];
    uintt count = 1;
    entries[0] = 0;
    entries[1] = 0;

    transferMatrixCpu.setOutputMatrix(transferMatrix);
    transferMatrixCpu.setThreadsCount(threadsCount);
    transferMatrixCpu.setSerieLimit(serieLimit);
    transferMatrixCpu.setSpinsCount(spinsCount);
    transferMatrixCpu.setQuantums(q);
    transferMatrixCpu.setQuantumsCount(qCount);
    transferMatrixCpu.setTrotterNumber(M);
    transferMatrixCpu.setExpHamiltonian1(eh1);
    transferMatrixCpu.setExpHamiltonian2(eh2);
    math::Status status = transferMatrixCpu.start();
    transferMatrixCpu.setOutputMatrix(NULL);

    math::cpu::IraMethodCallback iram(&mo, 3 * l);
    iram.setHSize(16);
    iram.setRho(1. / sqrt(2));
    iram.setMatrix(transferMatrix);
    iram.setThreadsCount(THREADS_COUNT);
    iram.registerCallback(Callback_f, &transferMatrixCpu);
    uintt eigenvaluesCount = 2;
    floatt nvalues[eigenvaluesCount];
    memset(nvalues, 0, eigenvaluesCount * sizeof (floatt));
    iram.setReOutputValues(nvalues, eigenvaluesCount);
    iram.start();
    for (uintt fa = 0; fa < eigenvaluesCount; fa++) {
        printf("eigenvalue = %f\n", nvalues[fa]);
    }
    return 0;
}

int main(int argc, char** argv) {
    //return main1(argc, argv);
    return main2(argc, argv);
}

//215494.289023