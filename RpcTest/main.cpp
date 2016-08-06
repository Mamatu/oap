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

/*
 * 
 */

utils::ArgumentType inargs[] = {utils::ARGUMENT_TYPE_FLOAT};
utils::ArgumentType outargs[] = {utils::ARGUMENT_TYPE_FLOAT};

class TestFunction : public utils::OapFunction {
public:

    TestFunction() : utils::OapFunction("TestFunction", inargs, 1, outargs, 1) {
    }

    ~TestFunction() {
    }
protected:

    void invoked(utils::Reader& reader, utils::Writter* writer) {
        float f = reader.getFloat();
        fprintf(stderr, "float = %f \n", f);
        f = f * 2.f;
        if (writer) {
            fprintf(stderr, "kk = %f \n", f);
            writer->write(f);
        }
    }
};

void* Execute1(void* ptr) {
    utils::RpcImpl rpc("127.0.0.1", 5000);
    rpc.registerCall(new TestFunction());
    uint rh = 0;
    rh = rpc.waitOnConnection();
    while (true) {
    }
    pthread_exit(0);
}

void* Execute2(void* ptr) {
    utils::RpcImpl rpc("127.0.0.1", 5001);
    uint handle = rpc.connect("127.0.0.1", 5000);
    //rpc.init(5001);
    utils::Writer writer;
    utils::Reader reader;
    writer.putFloat(5.f);

    rpc.call(handle, "TestFunction", writer, inargs, 1, reader);
    fprintf(stderr, "result = %f \n", reader.getFloat());
    sleep(3);
    pthread_exit(0);
}

int main(int argc, char** argv) {

    pthread_t threads[2];

    pthread_create(&threads[0], 0, Execute1, 0);
    
    void* o;
    pthread_join(threads[0], &o);
    
    return 0;
}

