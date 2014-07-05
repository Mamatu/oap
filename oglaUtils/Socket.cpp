/* 
 * File:   Socket.cpp
 * Author: marcin
 * 
 * Created on 2 sierpień 2012, 13:13
 */


#include "Socket.h"
#include "Writer.h"

namespace utils {

    class Client : public Socket {
    public:
        sockaddr_in* sockaddress;

        Client(int _socketID, sockaddr_in* _sockaddress) : Socket(_socketID), sockaddress(_sockaddress) {
            debugFuncBegin();
            debugFuncEnd();
        }

        ~Client() {
            debugFuncBegin();
            free(sockaddress);
            debugFuncEnd();
        }
    };

    class Server : public Socket {
        typedef std::vector<Socket*> Clients;
        Clients clients;

        typedef std::vector<pthread_t> Threads;
        Threads threads;

        bool wasCreatedThread;
        bool willBeDestroyed;

        server_callback_f server_callback;
        bool destroyed;

        synchronization::Mutex mutex;
        synchronization::Cond cond;
        void* data_ptr;

        class Local {
        public:

            Local(Server* _server, int _index, Socket* _client) : server(_server), index(_index), client(_client) {
            }

            Socket* client;
            int index;
            Server* server;
        };

    public:
        struct sockaddr_in serverAddress;

        void invokeCallback(Socket* server, Socket* client, const char* buffer, int size, void* data_ptr) {
            debugFuncBegin();
            if (server_callback != NULL) {
                server_callback(server, client, buffer, size, data_ptr);
            }
            debugFuncEnd();
        }

        void createThread(Socket* client) {
            debugFuncBegin();
            pthread_t thread;
            Local* local = new Local(this, 0, client);
            pthread_create(&thread, 0, Server::Execute, local);
            threads.push_back(thread);
            debugFuncEnd();
        }

        static void* Execute(void* ptr) {
            debugFuncBegin();
            Server::Local* local = (Server::Local*) ptr;
            Server* server = local->server;
            while (server->willBeDestroyed == false) {
                Socket* client = local->client;
                int size = -1;
                Socket::Receive(client, &size, sizeof (int));
                if (size > -1) {
                    char* buffer = new char[size];
                    Socket::Receive(client, buffer, size);
                    server->mutex.lock();
                    server->invokeCallback(server, client, buffer, size, server->data_ptr);
                    server->mutex.unlock();
                    delete[] buffer;
                } else {
                    client->close();
                    server->mutex.lock();
                    server->unregisterClient(client);
                    server->mutex.unlock();
                    delete local;
                    delete client;
                    break;
                }
            }
            int a = 0;
            pthread_exit(&a);
            debugFuncEnd();
        }

        void registerClient(Socket* socket) {
            debugFuncBegin();
            mutex.lock();
            clients.push_back(socket);
            if (clients.size() == 1 && wasCreatedThread == false) {
                this->createThread(socket);
                wasCreatedThread = true;
            } else if (clients.size() == 1) {
                cond.broadcast();
            }
            mutex.unlock();
            debugFuncEnd();
        }

        void unregisterClient(Socket* socket) {
            debugFuncBegin();
            Clients::iterator it = std::find(clients.begin(), clients.end(), socket);
            if (it != clients.end()) {
                clients.erase(it);
            }
            debugFuncEnd();
        }

        Server(int _socket, int port, server_callback_f _callback, void* _data_ptr) :
        Socket(_socket), wasCreatedThread(false), willBeDestroyed(false),
        server_callback(_callback), destroyed(false), data_ptr(_data_ptr) {
            debugFuncBegin();
            serverAddress.sin_family = AF_INET;
            serverAddress.sin_addr.s_addr = htonl(INADDR_ANY);
            serverAddress.sin_port = htons(port);
            if (bind(socketID, (struct sockaddr *) &serverAddress, sizeof (serverAddress)) < 0) {
            }
            debugFuncEnd();
        }

        ~Server() {
            debugFuncBegin();
            willBeDestroyed = true;
            for (int fa = 0; fa < this->threads.size(); fa++) {
                void* o;
                pthread_join(this->threads[fa], &o);
            }
            debugFuncEnd();
        }
    };

    Socket::Socket(int _socketID) : socketID(_socketID) {
        debugFuncBegin();
        debugFuncEnd();
    }

    Socket::~Socket() {
        debugFuncBegin();
        if (socketID != 0) {
            this->close();
        }
        debugFuncEnd();
    }

    void Socket::SetAddress(sockaddr_in* socketAddress, const char* address, int16_t port) {
        debugFuncBegin();
        memset(socketAddress, '0', sizeof (*socketAddress));
        socketAddress->sin_family = AF_INET;
        if (address != NULL) {
            if (inet_pton(AF_INET, address, &(socketAddress->sin_addr)) <= 0) {
            }
        } else {
            socketAddress->sin_addr.s_addr = INADDR_ANY;
        }
        socketAddress->sin_port = htons(port);
        debugFuncEnd();
    }

    Socket* Socket::CreateClient(const char* address, int16_t port) {
        debugFuncBegin();
        debug("Create client: %s:%d \n", address ? address : "", port);
        sockaddr_in* sockaddress = (sockaddr_in*) malloc(sizeof (sockaddr_in));
        int socketID = socket(AF_INET, SOCK_STREAM, 0);
        Socket::SetAddress(sockaddress, address, port);

        Client* client = new Client(socketID, sockaddress);

        client->address = address;
        client->port = port;
        debugFuncEnd();
        return client;
    }

    Socket* Socket::CreateServer(int16_t port, server_callback_f server_callback, void* data_ptr) {
        debugFuncBegin();
        debug("Create server: %d \n", port);
        struct sockaddr_in serv_addr;
        int socketID = socket(AF_INET, SOCK_STREAM, 0);
        Server* server = new Server(socketID, port, server_callback, data_ptr);
        server->port = port;
        debugFuncEnd();
        return server;
    }

    void Socket::send(const char* buffer, int size) {
        debugFuncBegin();
        debug("Buffer sent: size == %d \n", size);
        ::send(this->socketID, &size, sizeof (int), MSG_WAITALL);
        ::send(this->socketID, buffer, size, MSG_WAITALL);
    }

    void Socket::send(utils::Serializable* serialize) {
        debugFuncBegin();
        Writer writer;
        serialize->serialize(writer);
        unsigned int size = 0;
        char* bytes = NULL;
        writer.getBuffer(&bytes, size);
        this->send(bytes, size);
        if (bytes) {
            delete[] bytes;
        }
        debugFuncEnd();
    }

    void Socket::Receive(Socket* socket, void* buffer, int size) {
        debugFuncBegin();
        if (socket) {
            recv(socket->socketID, buffer, size, MSG_WAITALL);
        }
        debugFuncEnd();
    }

    void Socket::close() {
        debugFuncBegin();
        if (this->socketID > 0) {
            ::close(this->socketID);
            this->socketID = 0;
        }
        debugFuncEnd();
    }

    bool Socket::connect() {
        debugFuncBegin();
        Client* client = dynamic_cast<Client*> (this);
        if (client != NULL) {
            int status = ::connect(client->socketID, (struct sockaddr*) (client->sockaddress), sizeof (*(client->sockaddress)));
            debug("Client socket: %p - connection status == %d \n", client, status);
            return status == 0 ? true : false;
        } else {
            Server* server = dynamic_cast<Server*> (this);
            listen(server->socketID, 5);
            socklen_t s = sizeof (sockaddr_in);
            server->registerClient(new Socket(accept(server->socketID,
                    (sockaddr*) &(server->serverAddress), &s)));
            debug("Client socket: %p - connection status == true (server) \n", server);
            return true;
        }
        debugFuncEnd();
    }
}