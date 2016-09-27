/*
 * Copyright 2016 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */


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

class OapSocketTests : public testing::Test {
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

TEST_F(OapSocketTests, DISABLED_SocketCommunication1) {
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

  ServerImpl::values.clear();

  EXPECT_TRUE(p1Data.flags1.first);
  EXPECT_TRUE(p1Data.flags1.second);
  EXPECT_TRUE(p2Data.flag2);
}
