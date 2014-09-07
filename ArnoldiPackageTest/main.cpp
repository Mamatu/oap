/* 
 * File:   main.cpp
 * Author: mmatula
 *
 * Created on August 26, 2014, 10:29 PM
 */

#include <cstdlib>
#include "ArnoldiMethodProcess.h"
#include "HostMatrixModules.h"
#include "main.h"

using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {

    math::Matrix* m = host::NewReMatrixCopy(16, 16, tm16);

    host::PrintReMatrix("INPUT =", m);

    uintt count = 3;

    floatt revs[count];
    floatt imvs[count];

    api::ArnoldiPackage arnoldi(api::ArnoldiPackage::ARNOLDI_CPU);

    arnoldi.setMatrix(m);
    arnoldi.setHDimension(4);
    arnoldi.setEigenvaluesBuffer(revs, imvs, count);
    arnoldi.start();
    return 0;
}

