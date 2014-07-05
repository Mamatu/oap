/* 
 * File:   main.cpp
 * Author: marcin
 *
 * Created on 02 February 2013, 19:37
 */

#include <cstdlib>
#include <iostream>
#include "Socket.h"

using namespace std;

void server_callback_f(utils::Socket* server, utils::Socket*, const char* buffer, int size, void* data_ptr) {
    utils::Reader reader(buffer, size);
    fprintf(stderr, "uint 1== %u \n", reader.getInt());
    fprintf(stderr, "uint 2== %u \n", reader.getInt());
}

void* Execute1(void* ptr) {
    fprintf(stderr, "Server\n");
    uint a = 0;
    utils::Socket* server = utils::Socket::CreateServer(5000, server_callback_f, &a);
    server->connect();
    sleep(3);
    server->close();
    delete server;
    pthread_exit(0);
}

void* Execute2(void* ptr) {
    fprintf(stderr, "CLIENT\n");
    utils::Socket* client = utils::Socket::CreateClient("127.0.0.1", 5000);
    client->connect();
    utils::Writer writer;
    writer.write(10);
    writer.write(3);
    client->send(&writer);
    client->close();
    sleep(3);
    delete client;
    pthread_exit(0);
}

int main(int argc, char** argv) {
    pthread_t threads[2];

    pthread_create(&threads[0], 0, Execute1, 0);
    sleep(1);
    pthread_create(&threads[1], 0, Execute2, 0);

    void* o;
    pthread_join(threads[0], &o);
    pthread_join(threads[1], &o);

    return 0;
}

