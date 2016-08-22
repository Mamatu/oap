/*
 * Copyright 2016 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */




#include <cstdlib>
#include "RealTransferMatrixCpu.h"
#include "RealTransferMatrixCuda.h"
#include "HostMatrixModules.h"
#include "DeviceMatrixModules.h"
#include "Matrix.h"
#include "KernelExecutor.h"
#include "ArnoldiMethodProcess.h"
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
math::MathOperationsCpu matrixOperations;

void setPauliMatrixX(math::Matrix* matrix) {
    matrix->reValues[0] = 0;
    matrix->reValues[1] = 1;
    matrix->reValues[2] = 1;
    matrix->reValues[3] = 0;
}

void setPauliMatrixY(math::Matrix* matrix) {
    matrix->imValues[0] = 0;
    matrix->imValues[1] = -1;
    matrix->imValues[2] = 1;
    matrix->imValues[3] = 0;
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

math::Matrix* createHamiltonian(floatt J, math::Matrix* spin1,
    math::Matrix* spin2, math::Matrix* identity, math::Matrix* tempMatrix1) {
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
    math::MathOperationsCpu mo;
    math::Matrix* A = host::NewReMatrixCopy(3, 3, (floatt*) aA);
    math::Matrix* m = host::NewReMatrix(3, 3, 0);
    //host::SetDiagonals(m, 2);
    mo.substract(A, A, m);
    floatt d;
    mo.det(&d, A);
    fprintf(stderr, "f == %f \n", d);
}

floatt* reoutpus = NULL;
floatt* recount = 0;

void Callback_f(int event, void* object, void* userPtr) {
    /*if (event == math::ArnoldiMethodCallbackGpu::EVENT_MATRIX_MULTIPLICATION) {
        shibataCpu::cpu::RealTransferMatrix* transferMatrixCpu =
                reinterpret_cast<shibataCpu::cpu::RealTransferMatrix*> (userPtr);
        math::ArnoldiMethodCallbackGpu::Event* event =
                reinterpret_cast<math::ArnoldiMethodCallbackGpu::Event*> (object);

        transferMatrixCpu->setEntries(event->getMatrixEntries(),
                event->getCount());
        transferMatrixCpu->setReOutputEntries(event->getReOutputs());
        transferMatrixCpu->setImOutputEntries(event->getImOutputs());
        transferMatrixCpu->start();
    }*/
}

int main1(int argc, char** argv) {
    //qtest();
    //return 0;
    //cuda::Context context(1);
    //context.init();
    HostMatrixAllocator hmm;
    HostMatrixUtils mu;
    math::MathOperationsCpu mo;
    shibataCpu::RealTransferMatrix transferMatrixCpu;
    shibataCuda::RealTransferMatrix transferMatrixCuda;
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

    floatt z[16] = {0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0};

    math::Matrix* h1 = host::NewMatrixCopy(4, 4, (floatt*) b, (floatt*) b1);
    math::Matrix* h2 = host::NewMatrixCopy(4, 4, (floatt*) b, (floatt*) b1);
    math::Matrix* eh1 = host::NewMatrixCopy(4, 4, (floatt*) b, (floatt*) b1);
    math::Matrix* eh2 = host::NewMatrixCopy(4, 4, (floatt*) b, (floatt*) b1);

    //math::Matrix* h1 = host::NewReMatrixCopy(4, 4, (floatt*) b);
    //math::Matrix* h2 = host::NewReMatrixCopy(4, 4, (floatt*) b);
    //math::Matrix* eh1 = host::NewReMatrixCopy(4, 4, (floatt*) b);
    //math::Matrix* eh2 = host::NewReMatrixCopy(4, 4, (floatt*) b);


    transferMatrixCpu.PrepareHamiltonian(eh1, h1, shibataCpu::ORIENTATION_REAL_DIRECTION);
    transferMatrixCpu.PrepareHamiltonian(eh2, h2, shibataCpu::ORIENTATION_REAL_DIRECTION);

    int threadsCount = 4;
    int spinsCount = 3; //25;
    int q[] = {-1, 1};
    int qCount = 2;
    int serieLimit = 25;
    int M = 2;
    int l = pow(qCount, 2 * M);
    fprintf(stderr, "l = %d\n", l);
    math::Matrix* transferMatrix = hmm.newMatrix(l, l);

    //csetReOutputEntries(reoutputs);
    //transferMatrixCpu.setEntries(entries, count);
    transferMatrixCpu.setOutputMatrix(transferMatrix);
    transferMatrixCpu.setThreadsCount(threadsCount);
    transferMatrixCpu.setSerieLimit(serieLimit);
    transferMatrixCpu.setSpinsCount(spinsCount);
    transferMatrixCpu.setQuantumsCount(qCount);
    transferMatrixCpu.setTrotterNumber(M);
    transferMatrixCpu.setExpHamiltonian1(eh1);
    transferMatrixCpu.setExpHamiltonian2(eh2);
    math::Status status = transferMatrixCpu.start();
    //transferMatrixCpu.setOutputMatrix(NULL);

    api::ArnoldiPackage/*Callback */iram(api::ArnoldiPackage::ARNOLDI_CPU/*, 16 * 3*/);
    iram.setHDimension(4);
    iram.setRho(1. / 3.14);
    iram.setMatrix(transferMatrix);
    //iram.setThreadsCount(THREADS_COUNT);
    //iram.registerCallback(Callback_f, &transferMatrixCpu);
    uintt eigenvaluesCount = 2;
    floatt revalues[eigenvaluesCount];
    floatt imvalues[eigenvaluesCount];
    memset(revalues, 0, eigenvaluesCount * sizeof (floatt));
    memset(imvalues, 0, eigenvaluesCount * sizeof (floatt));
    //iram.setEigenvaluesBuffer(revalues, eigenvaluesCount);
    //iram.setEigenvectorsBuffer(imvalues, eigenvaluesCount);
    iram.start();
    for (uintt fa = 0; fa < eigenvaluesCount; fa++) {
        fprintf(stderr, "eigenvalue = %f\n", revalues[fa]);
        fprintf(stderr, "eigenvalue = %f\n", imvalues[fa]);
    }
    return 0;
}

void prepareH(math::Matrix* o,
    math::Matrix* identity,
    math::Matrix* spin1,
    math::Matrix* spin2,
    math::Matrix* spin3,
    math::Matrix* temp,
    math::Matrix* temp1,
    uintt M,
    floatt T, math::MathOperationsCpu& mo) {
    floatt jx = -1;
    floatt jy = -1;
    floatt jz = -1;
    mo.tensorProduct(temp, spin1, identity);
    mo.tensorProduct(temp1, identity, spin1);
    mo.add(o, temp, temp1);

    host::SetZero(temp);
    host::SetZero(temp1);

    mo.tensorProduct(temp, spin3, identity);
    mo.tensorProduct(temp1, identity, spin3);
    mo.add(temp, temp1, temp);
    mo.add(temp1, o, temp);

    host::SetZero(temp);
    host::SetZero(temp1);

    mo.tensorProduct(temp, spin2, identity);
    mo.tensorProduct(o, identity, spin2);
    mo.add(temp, o, temp);

    mo.multiply(o, temp1, temp);

    floatt v = (-1. * 0.1);
    host::PrintMatrix("o = ", o);
    mo.multiply(temp1, temp1, &v);
    mo.exp(o, temp1);
}

int main2(int argc, char** argv) {
    HostMatrixAllocator hmm;
    HostMatrixUtils mu;
    math::MathOperationsCpu mo;
    shibataCpu::RealTransferMatrix transferMatrixCpu;
    char* impath = NULL;
    char* repath = NULL;
    char* countc = argv[1];
    int count = 20;
    if (argc > 1) {
        count = atoi(countc);
        if (argc > 2) {
            repath = argv[2];
            if (argc > 3) {
                impath = argv[3];
            }
        }
    }

    math::Matrix* h1 = host::NewMatrix(4, 4, 0);
    math::Matrix* h2 = host::NewMatrix(4, 4, 0);
    math::Matrix* eh1 = host::NewMatrix(4, 4, 0);
    math::Matrix* eh2 = host::NewMatrix(4, 4, 0);

    int threadsCount = 4;
    int spinsCount = 3; //25;
    int q[] = {-1, 1};
    int qCount = 2;
    int serieLimit = 25;
    int M = 2;
    floatt T = 2;
    int l = pow(qCount, 2 * M);
    math::Matrix* transferMatrix = hmm.newMatrix(1, l);

    math::Matrix* identity = host::NewMatrix(2, 2, 0);



    host::SetIdentity(identity);

    math::Matrix* spin1 = host::NewMatrix(2, 2);
    math::Matrix* spin2 = host::NewMatrix(2, 2);
    math::Matrix* spin3 = host::NewMatrix(2, 2);
    math::Matrix* spin4 = host::NewMatrix(2, 2);
    math::Matrix* temp = host::NewMatrix(4, 4);
    math::Matrix* temp1 = host::NewMatrix(4, 4);

    setPauliMatrixX(spin1);
    setPauliMatrixY(spin2);
    setPauliMatrixZ(spin3);


    for (uintt fa = 0; fa < count; ++fa) {
        if (repath) {
            host::LoadMatrix(h1, repath, impath, fa);
            host::LoadMatrix(h2, repath, impath, fa);
        } else {
            host::SetZero(h1);
            host::SetZero(h2);
            host::SetZero(temp);
            host::SetZero(temp1);
            prepareH(h1, identity, spin1, spin2, spin3,
                temp, temp1, M, 0.01 + (T * (floatt) fa) / (floatt) count, mo);
            prepareH(h2, identity, spin1, spin2, spin3,
                temp, temp1, M, 0.01 + (T * (floatt) fa) / (floatt) count, mo);
        }
        //math::Matrix* h1 = host::NewMatrixCopy(4, 4, (floatt*) b, (floatt*) z);
        //math::Matrix* h2 = host::NewMatrixCopy(4, 4, (floatt*) b, (floatt*) z);
        //math::Matrix* eh1 = host::NewMatrixCopy(4, 4, (floatt*) b, (floatt*) z);
        //math::Matrix* eh2 = host::NewMatrixCopy(4, 4, (floatt*) b, (floatt*) z);

        host::PrintMatrix("h1 =", h1);
        host::PrintMatrix("h2 =", h2);

        transferMatrixCpu.PrepareHamiltonian(eh1, h1, shibataCpu::ORIENTATION_REAL_DIRECTION);
        transferMatrixCpu.PrepareHamiltonian(eh2, h2, shibataCpu::ORIENTATION_REAL_DIRECTION);

        transferMatrixCpu.setOutputMatrix(transferMatrix);
        transferMatrixCpu.setThreadsCount(threadsCount);
        transferMatrixCpu.setSerieLimit(serieLimit);
        transferMatrixCpu.setSpinsCount(spinsCount);
        transferMatrixCpu.setQuantumsCount(qCount);
        transferMatrixCpu.setTrotterNumber(M);
        transferMatrixCpu.setExpHamiltonian1(eh1);
        transferMatrixCpu.setExpHamiltonian2(eh2);
        math::Status status = transferMatrixCpu.start();
        transferMatrixCpu.setOutputMatrix(NULL);

        /*math::cpu::IraMethodCallback iram(&mo, 3 * l);
        iram.setHSize(10);
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
        }*/
    }
    host::DeleteMatrix(identity);

    return 0;
}

int main(int argc, char** argv) {
    //return main1(argc, argv);
    return main2(argc, argv);
}

//215494.289023
