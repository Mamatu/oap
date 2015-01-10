/* 
 * File:   Socket.cpp
 * Author: marcin
 * 
 * Created on 2 sierpień 2012, 13:13
 */


#include "Socket.h"
#include "Writer.h"
#include "InternalTypes.h"

namespace utils {

Socket::Socket(bool createSocket) : m_port(0), m_socketAddress(new sockaddr_in()) {
    initID(createSocket);
}

Socket::Socket(int socketID) : m_socketID(socketID),
    m_socketAddress(new sockaddr_in()) {
    // not implemented
}

Socket::~Socket() {
    close();
    delete m_socketAddress;
}

void Socket::close() {
    if (this->m_socketID > 0) {
        ::close(this->m_socketID);
        this->m_socketID = 0;
    }
}

void Socket::initID(bool createSocket) {
    if (createSocket) {
        m_socketID = socket(AF_INET, SOCK_STREAM, 0);
    } else {
        m_socketID = 0;
    }
}

int Socket::getID() const {
    return m_socketID;
}

sockaddr_in* Socket::getSockaddr() const {
    return m_socketAddress;
}

Server::Server(int16_t port) : isDestroyed(false) {
    m_socketAddress->sin_family = AF_INET;
    m_socketAddress->sin_addr.s_addr = htonl(INADDR_ANY);
    m_socketAddress->sin_port = htons(port);
    if (bind(getID(), (struct sockaddr *) m_socketAddress,
        sizeof (*m_socketAddress)) < 0) {
    }
}

Server::~Server() {
    isDestroyed = true;
    for (size_t fa = 0; fa < m_clients.size(); ++fa) {
        m_clients[fa].release();
    }
    m_clients.clear();
}

bool Server::connect() {
    listen(getID(), 5);
    socklen_t s = sizeof (sockaddr_in);
    debug("Server %p waits on connection.", this);
    int id = accept(getID(), (sockaddr*) getSockaddr(), &s);
    debug("Client socket: %p - connection status == true (server) \n", this);
    if (id == -1) {
        return false;
    }
    ServerClient* serverClient = new ServerClient(id, this);
    return serverClient->connect();
}

Server::ServerClient::ServerClient(int socketID, Server* server) :
    Socket(socketID), m_server(server) {
    // not implemented
}

bool Server::ServerClient::connect() {
    m_server->registerClient(this);
    return true;
}

void Server::registerClient(Server::ServerClient* socket) {
    debugFuncBegin();
    mutex.lock();
    Server::ThreadData* threadData = new Server::ThreadData(this,
        m_clients.size(), socket);
    utils::Thread* thread = new utils::Thread();
    thread->setFunction(Server::Execute, threadData);
    thread->run();
    ClientData clientData(thread, threadData, socket, false);
    m_clients.push_back(clientData);
    mutex.unlock();
    debugFuncEnd();
}

void Server::ClientData::release() {
    if (!m_isReleased) {
        m_isReleased = true;
        m_client->close();
        m_thread->yield();
        delete m_thread;
        delete m_threadData;
        delete m_client;
    }
}

Server::ClientData::ClientData(utils::Thread* thread,
    ThreadData* threadData, ServerClient* client, bool shouldBeDeallocated) :
    m_thread(thread),
    m_threadData(threadData),
    m_client(client),
    m_isReleased(false),
    m_shouldBeDeallocated(shouldBeDeallocated) {
    // not implemented
}

Server::ClientData::~ClientData() {
    if (m_shouldBeDeallocated) {
        release();
    }
}

bool Server::ClientData::operator==(const Server::ClientData& clientData) {
    return this->m_client == clientData.m_client;
}

bool Server::ClientData::operator==(const Server::ServerClient* serverClient) {
    return this->m_client == serverClient;
}

void Server::unregisterClient(Server::ServerClient* client) {
    debugFuncBegin();
    mutex.lock();
    Clients::iterator it = std::find(m_clients.begin(), m_clients.end(), client);
    if (it != m_clients.end()) {
        it->release();
        m_clients.erase(it);
    }
    mutex.unlock();
    debugFuncEnd();
}

Server::ThreadData::ThreadData(Server* _server, int _index, ServerClient* _client) :
    m_server(_server), m_index(_index), m_client(_client) {
    // not implemented
}

void Server::ServerClient::receive(void* buffer, size_t size) {
    if (getID() != 0) {
        recv(getID(), buffer, size, MSG_WAITALL);
    }
}

void Server::Execute(void* ptr) {
    debugFuncBegin();
    Server::ThreadData* data = (Server::ThreadData*) ptr;
    Server* server = data->m_server;
    while (server->isDestroyed == false) {
        ServerClient* client = data->m_client;
        int size = -1;
        client->receive(&size, sizeof (int));
        if (size > -1) {
            char* buffer = new char[size];
            client->receive(buffer, size);
            server->invokeCallback(client, buffer, size);
            delete[] buffer;
        } else {
            server->unregisterClient(client);
            break;
        }
    }
    int temp = 0;
    pthread_exit(&temp);
    debugFuncEnd();
}

void Server::invokeCallback(Socket* client, const char* buffer, int size) {
    mutex.lock();
    OnData(client, buffer, size);
    mutex.unlock();
}

Client::Client(const char* address, int16_t port) : Socket() {
    init(address, port);
}

void Client::send(const char* buffer, size_t size) {
    ::send(getID(), &size, sizeof (int), MSG_WAITALL);
    ::send(getID(), buffer, size, MSG_WAITALL);
}

void Client::send(utils::Serializable* serialize) {
    debugFuncBegin();
    Writer writer;
    serialize->serialize(writer);
    unsigned int size = 0;
    char* bytes = NULL;
    writer.getBufferCopy(&bytes, size);
    this->send(bytes, size);
    if (NULL != bytes) {
        delete[] bytes;
    }
    debugFuncEnd();
}

void Client::init(const char* address, int16_t port) {
    debug("Create client: %s:%d \n", address ? address : "", port);
    initAddress(address, port);
    m_address = address;
    m_port = port;
}

void Client::initAddress(const char* address, int16_t port) {
    memset(m_socketAddress, 0, sizeof (*m_socketAddress));
    m_socketAddress->sin_family = AF_INET;
    if (address != NULL) {
        if (inet_pton(AF_INET, address, &(m_socketAddress->sin_addr)) <= 0) {
        }
    } else {
        m_socketAddress->sin_addr.s_addr = INADDR_ANY;
    }
    m_socketAddress->sin_port = htons(port);
}

bool Client::connect() {
    int status = ::connect(getID(), (struct sockaddr*) (m_socketAddress),
        sizeof (*(m_socketAddress)));
    debug("Client socket: %p - connection status == %b \n", this, status == 0);
    return status == 0;
}
}