/* 
 * File:   Socket.h
 * Author: marcin
 *
 * Created on 2 sierpień 2012, 13:13
 */

#ifndef SOCKET_H
#define	SOCKET_H

#include "Buffer.h"
#include "Writer.h"
#include "ArrayTools.h"
#include "Reader.h"
#include <algorithm>
#include <vector>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <vector>

namespace utils {

class Socket {
public:
    Socket(bool createSocket = true);
    Socket(int socketID);
    virtual ~Socket();

    virtual bool connect() = 0;

    void close();

protected:
    void initID(bool createSocket = true);

    int getID() const;
    sockaddr_in* getSockaddr() const;
    sockaddr_in* m_socketAddress;
    int16_t m_port;
private:
    int m_socketID;
};

class Server : public Socket {
public:
    Server(int16_t port);
    virtual ~Server();

    bool connect();

protected:
    virtual void OnData(Socket* client, const char* buffer, int size) = 0;

    class ServerClient : public Socket {
    public:
        ServerClient(int socketID, Server* server);
        bool connect();
        void receive(void* buffer, size_t size);
    private:
        Server* m_server;
    };
    friend class ServerClient;

    void registerClient(ServerClient* serverClient);
    void unregisterClient(ServerClient* serverClient);
private:
    volatile bool isDestroyed;
    utils::sync::Cond cond;
    utils::sync::Mutex mutex;

    class ThreadData {
    public:
        ThreadData(Server* _server, int _index, ServerClient* _client);
        ServerClient* m_client;
        int m_index;
        Server* m_server;
    };

    class ClientData {
    public:
        utils::Thread* m_thread;
        ThreadData* m_threadData;
        ServerClient* m_client;
        bool operator==(const ClientData& clientData);
        bool operator==(const Server::ServerClient* serverClient);

        ClientData(utils::Thread* thread,
            ThreadData* threadData, ServerClient* client,
            bool shouldBeDeallocated = false);
        ~ClientData();
        void release();
    private:
        bool m_isReleased;
        bool m_shouldBeDeallocated;
    };

    typedef std::vector<ClientData> Clients;
    Clients m_clients;
    void invokeCallback(Socket* client, const char* buffer, int size);
    static void Execute(void* ptr);
};

class Client : public Socket {
public:

    Client(const char* address, int16_t port);

    void send(const char* buffer, size_t size);
    void send(utils::Serializable* serialize);

    bool connect();
private:

    void init(const char* address, int16_t port);
    void initAddress(const char* address, int16_t port);
    std::string m_address;

};

}

#endif	/* SOCKET_H */

