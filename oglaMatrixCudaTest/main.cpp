/* 
 * File:   main.cpp
 * Author: mmatula
 *
 * Created on January 11, 2014, 1:22 PM
 */

#include <cstdlib>
#include "Matrix.h"

using namespace std;

void Test_Det() {

    debugFunc();

    floatt Aa[] = {
        75.2099, 0, 0, 2809.84, 0, 0, 2734.63, 0, 0, 2734.63, 0, 0, 2809.84, 0, 0, 75.2099,
        0, -5505.87, 0, 0, 36.605, 0, 0, 2734.63, 0, 0, 0, 0, 0, 2734.63, 0, 0,
        0, 0, -5505.87, 0, 0, 0, 0, 0, 36.605, 0, 0, 2734.63, 0, 0, 2734.63, 0,
        2734.63, 0, 0, -5505.87, 0, 0, 0, 0, 0, 0, 0, 0, 36.605, 0, 0, 2734.63,
        0, 36.605, 0, 0, -5505.87, 0, 0, 2734.63, 0, 0, 0, 0, 0, 2734.63, 0, 0,
        0, 0, 0, 0, 0, -10938.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        2809.84, 0, 0, 75.2099, 0, 0, -5505.87, 0, 0, 36.605, 0, 0, 75.2099, 0, 0, 2809.84,
        0, 2734.63, 0, 0, 2734.63, 0, 0, -5505.87, 0, 0, 0, 0, 0, 36.605, 0, 0,
        0, 0, 36.605, 0, 0, 0, 0, 0, -5505.87, 0, 0, 2734.63, 0, 0, 2734.63, 0,
        2809.84, 0, 0, 75.2099, 0, 0, 36.605, 0, 0, -5505.87, 0, 0, 75.2099, 0, 0, 2809.84,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10938.5, 0, 0, 0, 0, 0,
        0, 0, 2734.63, 0, 0, 0, 0, 0, 2734.63, 0, 0, -5505.87, 0, 0, 36.605, 0,
        2734.63, 0, 0, 36.605, 0, 0, 0, 0, 0, 0, 0, 0, -5505.87, 0, 0, 2734.63,
        0, 2734.63, 0, 0, 2734.63, 0, 0, 36.605, 0, 0, 0, 0, 0, -5505.87, 0, 0,
        0, 0, 2734.63, 0, 0, 0, 0, 0, 2734.63, 0, 0, 36.605, 0, 0, -5505.87, 0,
        75.2099, 0, 0, 2809.84, 0, 0, 2734.63, 0, 0, 2734.63, 0, 0, 2809.84, 0, 0, 75.2099
    };

    math::Matrix* m = host::NewReMatrixCopy(16, 16, Aa);

    floatt d = 1;
    //mo.det(&d, m);
    fprintf(stderr, "det == %f \n", d);

    debugFunc();
}

void Test_Multiply() {
    debugFunc();
    DeviceMatrixAllocator dmm;
    HostMatrixAllocator hma;
    DeviceMatrixUtils dmu;
    DeviceMatrixPrinter dmp;
    math::Matrix* hm1 = hma.newReMatrix(100, 100, 1);
    math::Matrix* hm2 = hma.newReMatrix(100, 100, 2);
    math::Matrix* m1 = cuda::NewDeviceMatrix(hm1);
    math::Matrix* m2 = cuda::NewDeviceMatrix(hm2);

    floatt* a = CudaUtils::AllocDeviceObj(2.);
    dmu.printInfo(m1);
    dmu.printInfo(m2);
    dmu.setIdentityMatrix(m2);

    math::Matrix* houtput = hma.newReMatrix(100, 100);
    math::Matrix* output = cuda::NewDeviceMatrix(houtput);
    dmu.printInfo(output);

    //mo.multiply(output, m1, a);
    //PrintMatrix("output =", output);
    dmp.printReMatrix(output);
    //mo.printMessage();
    dmm.deleteMatrix(m1);
    dmm.deleteMatrix(m2);
    dmm.deleteMatrix(output);

    debugFunc();
}

int main(int argc, char** argv) {
    debugFuncBegin();
    cuda::Context context(1);
    context.init();
    math::MathOperationsCuda mo;
    mo.setDeviceInfo(context);
    Test_Multiply(mo);
    Test_Det(mo);
    return 0;
}

