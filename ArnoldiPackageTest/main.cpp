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
#include "DeviceMatrixModules.h"
#include "KernelExecutor.h"

using namespace std;

void hostMain() {
    math::Matrix* m = host::NewReMatrixCopy(16, 16, tm16);

    host::PrintReMatrix("INPUT =", m);

    uintt count = 2;

    floatt revs[count];
    floatt imvs[count];

    api::ArnoldiPackage arnoldi(api::ArnoldiPackage::ARNOLDI_CPU);

    arnoldi.setMatrix(m);
    arnoldi.setHDimension(4);
    arnoldi.setEigenvaluesBuffer(revs, imvs, count);
    arnoldi.start();
}

void deviceMain() {
    debugFunc();
    cuda::Context context;
    context.init();
    math::Matrix* m = host::NewReMatrixCopy(16, 16, tm16);

    //[215589.901030, -4.613732, 2867.286904, -2441.909650
    //    0.000000, 210191.120185, 5990.606945, -4906.898751
    //    0.000000, 0.000000, 102796.017502, -172308.590888
    //    0.000000, -0.000000, 0.000000, 31225.460862]

    debugFunc();
    host::PrintReMatrix("INPUT =", m);
    debug("Host matrix = %p", m);
    //debug("Device matrix = %p", dm);

    uintt count = 3;
    floatt revs[count];
    floatt imvs[count];
    api::ArnoldiPackage arnoldi(api::ArnoldiPackage::ARNOLDI_GPU);
    arnoldi.setMatrix(m);
    arnoldi.setHDimension(4);
    arnoldi.setEigenvaluesBuffer(revs, imvs, count);
    debug("Start = %d", arnoldi.start());
    debugFunc();
}

int main(int argc, char** argv) {
    hostMain();
    deviceMain();
    return 0;
}

