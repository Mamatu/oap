// Copyright 2008, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: wan@google.com (Zhanyong Wan)

// Google Mock - a framework for writing C++ mock classes.
//
// This file tests code in gmock.cc.

#include <string>
#include <stdio.h>
#include <pthread.h>
#include "gtest/gtest.h"
#include "Socket.h"
#include "gmock/gmock-generated-function-mockers.h"

class P1Data {
 public:
  P1Data(utils::sync::CondBool& cbool, utils::sync::CondBool& cbool1)
      : m_cbool(cbool), m_cbool1(cbool1) {}
  utils::sync::CondBool& m_cbool;
  utils::sync::CondBool& m_cbool1;
  std::pair<bool, bool> flags1;
};

class P2Data {
 public:
  P2Data(utils::sync::CondBool& cbool, utils::sync::CondBool& cbool1)
      : m_cbool(cbool), m_cbool1(cbool1) {
    flag2 = false;
  }
  utils::sync::CondBool& m_cbool;
  utils::sync::CondBool& m_cbool1;
  bool flag2;
};

class ServerImpl : public utils::Server {
 public:
  static std::vector<int> values;

  ServerImpl(int16_t port, P1Data* p1Data) : Server(port), m_p1Data(p1Data) {}

  bool Compare(const std::vector<int>& values) { return m_values == values; }

  P1Data* m_p1Data;

 protected:
  void OnData(Socket* client, const char* buffer, int size) {
    utils::Reader reader(buffer, size);
    m_values.push_back(reader.readInt());
    m_values.push_back(reader.readInt());
    this->close();
    m_p1Data->m_cbool.signal();
  }

 private:
  std::vector<int> m_values;
};

std::vector<int> ServerImpl::values;

class OglaSocketTests : public testing::Test {
 public:
  utils::Client* client;
  utils::Server* server;

  virtual void SetUp() { ServerImpl::values.clear(); }

  virtual void TearDown() {}
};

void* Execute1(void* ptr) {
  debug("Server");
  P1Data* p1Data = static_cast<P1Data*>(ptr);
  ServerImpl* server = new ServerImpl(5000, p1Data);
  bool connected = server->connect();
  debug("connected server = %d", connected);
  p1Data->m_cbool.wait();
  std::pair<bool, bool>* flag1 = &p1Data->flags1;
  (*flag1).first = connected;
  (*flag1).second = server->Compare(ServerImpl::values);
  delete server;
  pthread_exit(0);
}

void* Execute2(void* ptr) {
  P2Data* p2Data = static_cast<P2Data*>(ptr);
  debug("Client");
  utils::Client* client = new utils::Client("127.0.0.1", 5000);
  bool connected = false;
  do {
    connected = client->connect();
    debug("connected client = %d", connected);
  } while (connected == false);
  if (connected) {
    utils::Writer writer;
    for (int fa = 0; fa < ServerImpl::values.size(); ++fa) {
      writer.write(ServerImpl::values[fa]);
    }
    client->send(&writer);
    client->close();
  }
  delete client;
  bool* flag2 = &p2Data->flag2;
  (*flag2) = connected;
  pthread_exit(0);
}

TEST_F(OglaSocketTests, SocketCommunication1) {
  pthread_t threads[2];

  ServerImpl::values.push_back(10);
  ServerImpl::values.push_back(3);

  utils::sync::CondBool cbool;
  utils::sync::CondBool cbool1;

  P1Data p1Data(cbool, cbool1);
  P2Data p2Data(cbool, cbool1);

  pthread_create(&threads[0], 0, Execute1, &p1Data);
  pthread_create(&threads[1], 0, Execute2, &p2Data);
  void* o;
  pthread_join(threads[0], &o);
  pthread_join(threads[1], &o);
  EXPECT_TRUE(p1Data.flags1.first);
  EXPECT_TRUE(p1Data.flags1.second);
  EXPECT_TRUE(p2Data.flag2);
}
