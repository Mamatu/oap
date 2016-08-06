#include <cstdlib>
#include "OapServer.h"
#include <stdio.h>

using namespace std;

int main(int argc, char** argv) {
    int port = 4711;
    if (argc > 1) {
        char* arg1 = argv[1];
        port = atoi(arg1);
    }
    std::string version = "0.0.1";
    printf("Oap server, version: %s \n Port: %d\n\n Log: \n", version.c_str(), port);
    core::OapServer oapServer(port);
    oapServer.wait();
    return 0;
}

