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

    class Socket;

    typedef void (*server_callback_f)(Socket* server, Socket* client, const char* buffer, int size, void* data_ptr);

    class Socket {
    protected:
	std::string address;
	int16_t port;
	int socketID;
	Socket(int _socketID);

	static void Receive(Socket* socket, void* buffer, int size);
	static void SetAddress(sockaddr_in* socketAddress, const char* address, int16_t port);

    public:
	virtual ~Socket();

	static Socket* CreateClient(const char* address, int16_t port);
	static Socket* CreateServer(int16_t port, server_callback_f server_callback, void* data_ptr);

	void send(const char* buffer, int size);
	void send(utils::Serializable* serialize);

	bool connect();
	void close();
    };
}

#endif	/* SOCKET_H */

