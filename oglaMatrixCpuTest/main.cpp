/* 
 * File:   main.cpp
 * Author: mmatula
 *
 * Created on December 23, 2013, 6:48 PM
 */

#include <cstdlib>
#include "Matrix.h"
#include "HostMatrixModules.h"
#include "MathOperationsCpu.h"

using namespace std;
HostMatrixAllocator hmm;
HostMatrixPrinter mp;

void Test_Addition(int tc = 1) {
    debugFunc();
    floatt array[] = {1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1};
    math::Matrix* matrix1 = host::NewReMatrixCopy(4, 4, array);
    math::Matrix* matrix2 = host::NewReMatrixCopy(4, 4, array);
    math::Matrix* output = hmm.newMatrix(4, 4);
    math::AdditionOperationCpu additionOperation;
    additionOperation.setThreadsCount(tc);
    additionOperation.setOutputMatrix(output);
    additionOperation.setMatrix1(matrix1);
    additionOperation.setMatrix2(matrix2);
    printf("status == %d \n", additionOperation.start());
    host::PrintReMatrix(stderr, output);
    debugFunc();
}

void Test_Substraction(int tc = 1) {
    debugFunc();
    floatt array[] = {1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1};
    math::Matrix* matrix1 = host::NewReMatrix(4, 4, 0);
    math::Matrix* matrix2 = host::NewReMatrixCopy(4, 4, array);
    math::Matrix* output = hmm.newMatrix(4, 4);
    math::SubstracionOperationCpu substractionOperation;
    substractionOperation.setThreadsCount(tc);
    substractionOperation.setOutputMatrix(output);
    substractionOperation.setMatrix1(matrix1);
    substractionOperation.setMatrix2(matrix2);
    printf("status == %d \n", substractionOperation.start());
    host::PrintReMatrix(stderr, output);
    debugFunc();
}

void Test_Addition1(int tc = 1) {
    debugFunc();
    floatt array[] = {1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1};
    math::Matrix* matrix1 = host::NewReMatrixCopy(4, 4, array);
    math::Matrix* matrix2 = host::NewReMatrixCopy(4, 4, array);
    math::Matrix* output = hmm.newMatrix(4, 4);
    math::MathOperationsCpu mo;
    mo.setThreadsCount(tc);
    mo.setSubColumns(0, 2);
    mo.add(output, matrix1, matrix2);
    //printf("status == %d \n", additionOperation.start());
    host::PrintReMatrix(stderr, output);
    debugFunc();
}

void Test_Multiplication(int tc = 1) {
    debugFunc();

    floatt array[] = {1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1};

    floatt array1[] = {1, 0, 0, 0, 0,
        0, 1, 2, 0, 0,
        0, 0, 1, 5.6, 0,
        0, 3, 0, 1, 0,
        0, 3, 0, 1, 0};

    math::Matrix* matrix1 = host::NewReMatrixCopy(5, 5, array);
    math::Matrix* matrix2 = host::NewReMatrixCopy(5, 5, array1);
    mp.printReMatrix(stderr, matrix1);
    mp.printReMatrix(stderr, matrix2);
    math::Matrix* output = hmm.newMatrix(5, 5);
    math::DotProductOperationCpu multiplicationOperation;
    multiplicationOperation.setThreadsCount(tc);
    multiplicationOperation.setOutputMatrix(output);
    uintt a[] = {1, 2};
    multiplicationOperation.setSubColumns(a);
    multiplicationOperation.setMatrix1(matrix1);
    multiplicationOperation.setMatrix2(matrix2);
    multiplicationOperation.setThreadsCount(tc);
    printf("status == %d \n", multiplicationOperation.start());
    mp.printReMatrix(stderr, output);
    debugFunc();
}

void Test_Diagonalization(int tc = 1) {
    debugFunc();

    floatt array[] = {2, -2, 1,
        -1, 3, -1,
        2, -4, 3};

    floatt array1[] = {2, -1, 1,
        1, 0, -1,
        0, 1, 2};

    floatt array2[] = {1, 2, 0,
        0, 3, 0,
        2, -4, 2};

    floatt array3[] = {-1, 0, -1,
        -1, 0, 0,
        2, 1, 2};

    math::Matrix* matrix1 = host::NewReMatrixCopy(3, 3, array2);
    math::Matrix* matrix2 = host::NewReMatrixCopy(3, 3, array3);
    mp.printReMatrix(matrix1);
    mp.printReMatrix(matrix2);
    math::Matrix* output = hmm.newMatrix(3, 3);
    math::DiagonalizationOperationCpu diagonalizationOperation;
    diagonalizationOperation.setThreadsCount(tc);
    diagonalizationOperation.setOutputMatrix(output);
    diagonalizationOperation.setMatrix1(matrix1);
    diagonalizationOperation.setMatrix2(matrix2);
    //diagonalizationOperation.setThreadsCount(1);
    printf("status == %d \n", diagonalizationOperation.start());
    mp.printReMatrix(output);
    debugFunc();

}

void Test_TensorProduct(int tc = 1) {
    debugFunc();
    floatt array[] = {1, 0, 0, 1};
    floatt array1[] = {2, 0, 0, 1};
    math::Matrix* matrix1 = host::NewReMatrixCopy(2, 2, array);
    math::Matrix* matrix2 = host::NewReMatrixCopy(2, 2, array1);
    mp.printReMatrix(matrix1);
    mp.printReMatrix(matrix2);
    math::Matrix* output = hmm.newMatrix(4, 4);
    math::TensorProductOperationCpu tpOperation;
    tpOperation.setThreadsCount(tc);
    tpOperation.setOutputMatrix(output);
    tpOperation.setMatrix1(matrix1);
    tpOperation.setMatrix2(matrix2);
    printf("status == %d \n", tpOperation.start());
    mp.printReMatrix(output);
    debugFunc();
}

void Test_Multiplication1(int tc = 1) {
    debugFunc();
    math::MathOperationsCpu mo;
    HostMatrixUtils funcs;
    math::Matrix* m1 = hmm.newReMatrix(10, 10, 1);
    math::Matrix* m2 = hmm.newReMatrix(10, 10, 2);
    //funcs.setIdentityHostMatrix(m2);
    //hmp.PrintHostReMatrix(m1);
    math::Matrix* output = hmm.newReMatrix(10, 10);
    mo.setThreadsCount(tc);
    mo.multiply(output, m1, m2);
    mp.printReMatrix(output);
    hmm.deleteMatrix(m1);
    hmm.deleteMatrix(m2);
    hmm.deleteMatrix(output);
    debugFunc();
}

void Test_MultiplicationConst(int tc = 1) {
    debugFunc();
    math::MathOperationsCpu mo;
    HostMatrixUtils funcs;
    math::Matrix* m1 = hmm.newReMatrix(10, 10, 1);
    floatt m2 = 2.f;
    math::Matrix* output = hmm.newReMatrix(10, 10);
    mo.setThreadsCount(tc);
    mo.setSubColumns(3, 4);
    mo.multiply(output, m1, &m2);
    mp.printReMatrix(output);
    hmm.deleteMatrix(m1);
    hmm.deleteMatrix(output);
    debugFunc();
}

void Test_SubMultiplication(int tc = 1) {
    debugFunc();
    math::MathOperationsCpu mo;
    math::Matrix* o = host::NewReMatrix(10, 10, 0);
    math::Matrix* m1 = host::NewReMatrix(10, 10, 1);
    math::Matrix* m2 = host::NewReMatrix(10, 10, 1);
    mo.setSubRows(0, 1);
    mo.setSubColumns(0, 1);
    mo.setThreadsCount(tc);
    mo.multiply(o, m1, m2);
    host::PrintReMatrix(o);
    host::PrintReMatrix(m1);
    host::PrintReMatrix(m2);
    host::DeleteMatrix(o);
    host::DeleteMatrix(m1);
    host::DeleteMatrix(m2);
    debugFunc();
}

void Test_SubMultiplication1(int tc = 1) {
    debugFunc();

    math::MathOperationsCpu mo;
    math::Matrix* o = host::NewReMatrix(1, 10, 0);
    math::Matrix* m1 = host::NewReMatrix(10, 10, 1);
    math::Matrix* m2 = host::NewReMatrix(1, 10, 1);
    //while (true) {
    mo.setSubColumns(0, 5);
    mo.setSubRows(0, 5);
    //mo.setSubColumns(0, 5);
    //mo.setSubColumns(0, 1);
    mo.setThreadsCount(tc);
    mo.multiply(o, m1, m2);
    host::PrintReMatrix(o);
    host::PrintReMatrix(m1);
    host::PrintReMatrix(m2);
    //}
    host::DeleteMatrix(o);
    host::DeleteMatrix(m1);
    host::DeleteMatrix(m2);
    debugFunc();
}

void Test_Transpose(int tc = 1) {
    debugFunc();
    math::MathOperationsCpu mo;
    math::Matrix* o = host::NewReMatrix(5, 10, 0);
    math::Matrix* m1 = host::NewReMatrix(10, 5, 1);
    mo.setSubRows(0, 4);
    //mo.setSubColumns(0, 1);
    mo.setThreadsCount(tc);
    mo.transpose(o, m1);
    host::PrintMatrix("o =", o);
    host::PrintMatrix("m1 = ", m1);
    host::DeleteMatrix(o);
    host::DeleteMatrix(m1);
    debugFunc();
}

void Test_Determinant(int tc = 1) {
    debugFunc();
    math::MathOperationsCpu mo;
    floatt array[] = {
        -2, 2, -3,
        -1, 1, 3,
        2, 0, -1
    };
    math::Matrix* o = host::NewReMatrixCopy(3, 3, array);
    floatt value;
    mo.setThreadsCount(tc);
    mo.det(&value, o);
    fprintf(stderr, "value == %f \n", value);
    host::DeleteMatrix(o);
    debugFunc();
}

void Test_Determinant2(int tc = 1) {
    debugFunc();
    math::MathOperationsCpu mo;
    floatt array[] = {
        -5618.170, 5543.450, -1.932, 150.394,
        5545.490, -10936.100, 5615.610, 152.460,
        0.000, 5615.550, -5772.430, 220.557,
        0.000, 0.000, 72.205, -11234.900
    };
    math::Matrix* o = host::NewReMatrixCopy(3, 3, array);
    floatt value;
    mo.setThreadsCount(tc);
    mo.det(&value, o);
    fprintf(stderr, "value == %f \n", value);
    host::DeleteMatrix(o);
    debugFunc();
}

void Test_Determinant3(int tc = 1) {
    debugFunc();
    math::MathOperationsCpu mo;
    floatt array[] = {
        -5618.170, 5543.450, -1.932, 150.394,
        5545.490, -10936.100, 5615.610, 152.460,
        0.000, 5615.550, -5772.430, 220.557,
        0.000, 0.000, 72.205, -11234.900
    };
    math::Matrix* o = host::NewReMatrixCopy(3, 3, array);
    floatt value;
    mo.setThreadsCount(tc);
    mo.det(&value, o);
    fprintf(stderr, "value == %f \n", value);
    host::DeleteMatrix(o);
    debugFunc();
}

void Test_Determinant1(int tc = 1) {
    debugFunc();
    math::MathOperationsCpu mo;
    floatt array[] = {
        3, -5, 3,
        2, 1, -1,
        1, 0, 4
    };
    math::Matrix* o = host::NewReMatrixCopy(3, 3, array);
    floatt value;
    mo.setThreadsCount(tc);
    mo.det(&value, o);
    fprintf(stderr, "value == %f \n", value);
    host::DeleteMatrix(o);
    debugFunc();
}

void Test_Cut(int tc = 1) {
    debugFunc();
    math::MathOperationsCpu mo;
    floatt array[] = {
        -2, 2, -3,
        -1, 1, 3,
        2, 0, -1
    };
    math::Matrix* o = host::NewReMatrixCopy(3, 3, array);
    math::Matrix* o1 = host::NewReMatrix(2, 2, 0);
    host::Copy(o1, o, 0, 2);
    host::PrintReMatrix(o1);
    host::DeleteMatrix(o);
    host::DeleteMatrix(o1);
    debugFunc();
}

void Test_SubMultiplication2(int tc = 1) {
    floatt w[] = {12, 0, 0};
    floatt A[] = {12, -51, 4,
        6, 167, -68,
        -4, 24, -41};
    floatt v[] = {1, 0, 0};
}

void Test_DotProduct(int tc = 1) {
    floatt Aa[] = {11013.7, 5543.45, -1.93192, 150.394,
        5545.49, 5695.82, 5615.61, 152.46,
        0, 5615.55, 10859.5, 220.557,
        0, 0, 72.2053, 5396.97};
    floatt tqa[] = {0.893171, 0.449718, 0, 0,
        -0.188611, 0.374595, 0.907802, 0,
        0.00229071, -0.00454951, 0.00235324, 0.999984,
        -0.40903, 0.810125, -0.419955, 0.00550712};


    debugFunc();
    math::MathOperationsCpu mo;
    math::Matrix* A = host::NewReMatrixCopy(4, 4, Aa);
    math::Matrix* tq = host::NewReMatrixCopy(4, 4, tqa);
    math::Matrix* r = host::NewReMatrix(4, 4, 0);
    mo.multiply(r, tq, A);


    host::PrintReMatrix(r);
    floatt ra[] = {12331.1, 7512.76, 2523.71, 202.891,
        -4.54747e-13, 6185.88, 11962.2, 228.967,
        7.32392e-11, 6.72529e-11, 72.2064, 5397.05,
        -12.4107, -11.3961, -9.96319, -0.906733};
    debugFunc();
}

void Test_Det1(int tc = 1) {

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

    math::MathOperationsCpu mo;
    floatt d = 1;
    mo.det(&d, m);
    fprintf(stderr, "det == %f \n", d);

    debugFunc();
}

void Test_Det2(int tc = 1) {

    debugFunc();

    floatt Aa[] = {
        -5505.87, 0.000000000000000, 0.000000000000000, 36.605000000000000, 0.000000000000000,
        0.000000000000000, -5505.870000000000000, 0.000000000000000, 0.000000000000000, 2734.630000000000000,
        0.000000000000000, 0.000000000000000, -5505.870000000000000, 0.000000000000000, 0.000000000000000,
        36.605000000000000, 0.000000000000000, 0.000000000000000, -5505.870000000000000, 0.000000000000000,
        0.000000000000000, 2809.840000000000000, 0.000000000000000, 0.000000000000000, 75.209900000000000
    };

    floatt a = -5505.87;

    fprintf(stderr, "dasd == %f \n", Aa[0]);
    fprintf(stderr, "dasd == %f \n", -5505.87);

    math::Matrix* m = host::NewReMatrixCopy(5, 5, Aa);

    host::PrintReMatrix("m ==", m);
    math::MathOperationsCpu mo;
    floatt d = 1;
    mo.det(&d, m);
    fprintf(stderr, "det == %f \n", d);

    debugFunc();
}

void Test_QRDet(int tc = 1) {

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

    math::Matrix* A = host::NewReMatrixCopy(16, 16, Aa);
    math::Matrix* Q = host::NewReMatrix(16, 16);
    math::Matrix* R = host::NewReMatrix(16, 16);

    math::MathOperationsCpu mo;
    mo.qrDecomposition(Q, R, A);

    host::PrintReMatrix("A = ", A);
    host::PrintReMatrix("Q = ", Q);
    host::PrintReMatrix("R = ", R);

    fprintf(stderr, "det == %f \n", host::GetTrace(R));

    debugFunc();
}

void Test_Det3(int tc = 1) {

    debugFunc();

    floatt Aa[] = {
        -5505.870000000000000, 0.000000000000000, 0.000000000000000, 36.605000000000000,
        0.000000000000000, -5505.870000000000000, 0.000000000000000, 0.000000000000000,
        0.000000000000000, 0.000000000000000, -5505.870000000000000, 0.000000000000000,
        36.605000000000000, 0.000000000000000, 0.000000000000000, -5505.870000000000000
    };

    math::Matrix* m = host::NewReMatrixCopy(4, 4, Aa);

    host::PrintReMatrix("m ==", m);
    math::MathOperationsCpu mo;
    floatt d = 1;
    mo.det(&d, m);
    fprintf(stderr, "det == %f \n", d);

    debugFunc();
}

void Test_QRDecomposition(int tc = 1) {

    debugFunc();

    floatt Aa[] = {
        6, 5, 0,
        5, 1, 4,
        0, 4, 3
    };

    math::Matrix* A = host::NewReMatrixCopy(3, 3, Aa);
    math::Matrix* Q = host::NewReMatrix(3, 3);
    math::Matrix* R = host::NewReMatrix(3, 3);

    math::MathOperationsCpu mo;
    mo.qrDecomposition(Q, R, A);

    host::PrintReMatrix("A = ", A);
    host::PrintReMatrix("Q = ", Q);
    host::PrintReMatrix("R = ", R);
    debugFunc();
}

void Test_QRDecomposition1(int tc = 1) {

    debugFunc();

    floatt Aa[] = {
        210165, 5397.29, -1.81985, 154.369,
        5399.38, 204582, 5755.02, 152.453,
        0, 5754.86, 209410, 214.621,
        0, 0, 70.4028, 204259
    };

    math::Matrix* A = host::NewReMatrixCopy(4, 4, Aa);
    math::Matrix* Q = host::NewReMatrix(4, 4);
    math::Matrix* R = host::NewReMatrix(4, 4);

    math::MathOperationsCpu mo;
    mo.qrDecomposition(Q, R, A);

    host::PrintReMatrix("A = ", A);
    host::PrintReMatrix("Q = ", Q);
    host::PrintReMatrix("R = ", R);
    debugFunc();
}

int main(int argc, char** argv) {
    /*Test_Addition1(1);
    Test_Substraction(2);
    Test_Addition1();
    Test_SubMultiplication1();
    Test_SubMultiplication1();
    Test_Multiplication1();
    Test_SubMultiplication1(1);
    Test_Determinant(1);
    Test_Determinant1(1);
    Test_Determinant2(1);
    Test_Determinant3(1);
    
    Test_Cut();
    Test_Diagonalization();
    Test_TensorProduct();
    Test_DotProduct();*/
    //Test_Transpose(1);
    //Test_Det1();
    //Test_Det1();
    //Test_Det2();
    //Test_QRDet();
    //Test_Multiplication(1);
    //Test_MultiplicationConst(1);
    //Test_Det3();
    //Test_DotProduct();
    //Test_QRDecomposition();
    //Test_QRDecomposition1();
    return 0;
}

