/* 
 * File:   main.cpp
 * Author: marcin
 *
 * Created on 02 February 2013, 19:37
 */

#include <cstdlib>
#include <iostream>
#include "RpcImpl.h"

using namespace std;
using namespace utils;

int main(int argc, char** argv) {

    Writer writer;

    int v2 = 2365;
    float v3 = 25.45f;
    double v4 = 5.314;
    unsigned long long int v5 = 56565885455;

    writer.write(v2);
    writer.putFloat(v3);
    writer.putDouble(v4);
    writer.putLong(v5);
    std::string v6("test_JDAJDLAKSDJALKSDKL");
    writer.putStr(v6.c_str());


    Writer writer1;
    writer1.putSerializable(&writer);


    char* buffer = NULL;
    int size = 0;
    writer1.getBuffer(&buffer, size);

    Reader reader1(buffer, size);
    //reader.setSetter(&writer);

    Reader reader;
    reader1.getSerializable(&reader);

    int c2 = reader.getInt();
    std::cout << c2 << std::endl;

    float c3 = reader.getFloat();
    std::cout << c3 << std::endl;
    double c4 = reader.getDouble();
    std::cout << c4 << std::endl;
    unsigned long long int c5 = reader.getLong();
    std::cout << c5 << std::endl;
    char* c6 = reader.getStr();

    if (
            v2 == c2 &&
            v3 == c3 &&
            v4 == c4 &&
            v5 == c5 &&
            strcmp(v6.c_str(), c6) == 0
            ) {

        fprintf(stderr, "SUCCES\n");
    } else {
        fprintf(stderr, "fail\n");
    }

    return 0;
}

