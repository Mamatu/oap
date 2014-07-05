/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.core.rpc;

import java.io.IOException;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 *
 * @author mmatula
 */
public class Server {

    private ServerSocket ssocket = null;

    public interface ServerListener {

        public void getData(Socket socket, byte[] bytes);
    }

    public Server(int port) throws IOException {
        ssocket = new ServerSocket(port);
        this.port = port;
    }

    private boolean closed = false;
    private int port = -1;
    private List<ServerListener> listeners = new ArrayList<ServerListener>();

    public void addListener(ServerListener sl) {
        synchronized (this) {
            this.listeners.add(sl);
        }
    }

    public void removeListener(ServerListener sl) {
        synchronized (this) {
            this.listeners.remove(sl);
        }
    }

    public void close() throws IOException {
        closed = true;
        synchronized (clients) {
            for (Socket socket : clients) {
                socket.close();
            }
        }
    }

    private List<Socket> clients = new ArrayList<Socket>();

    private Socket createThread() {
        Runnable1 runnable = new Runnable1(this);
        Thread thread = new Thread(runnable);
        try {
            thread.start();
            runnable.lock.lock();
            if (runnable.client == null) {
                try {
                    runnable.condition.await();
                } catch (InterruptedException ie) {
                }
            }
            synchronized (clients) {
                this.clients.add(runnable.client);
            }
        } finally {
            runnable.lock.unlock();
        }
        return runnable.client;
    }

    public Socket connect() throws IOException {
        Socket socket = this.createThread();
        return socket;
    }

    public int getPort() {
        return this.port;
    }

    public String getAddress() throws UnknownHostException {
        InetAddress inetAddress = InetAddress.getLocalHost();
        return String.valueOf(inetAddress.getHostAddress());
    }

    private static class Runnable1 implements Runnable {

        private ServerSocket serverSocket;
        private Lock lock = new ReentrantLock();
        private Condition condition = lock.newCondition();
        private Socket client = null;
        private Server server = null;

        public Runnable1(Server server) {
            this.server = server;
            this.serverSocket = server.ssocket;
        }

        @Override
        public void run() {
            Socket socket = null;
            try {
                socket = this.serverSocket.accept();
            } catch (Exception e) {

            }
            this.lock.lock();
            try {
                client = socket;
                this.condition.signal();
            } finally {
                this.lock.unlock();
            }
            if (socket != null) {
                while (this.server.closed == false) {
                    byte[] sizeBytes = new byte[4];
                    byte[] data = null;
                    try {
                        this.client.getInputStream().read(sizeBytes);
                        int size = ByteBuffer.wrap(sizeBytes).order(ByteOrder.LITTLE_ENDIAN).getInt();
                        data = new byte[size];
                        this.client.getInputStream().read(data);
                    } catch (IOException io) {
                    }
                    synchronized (this.server) {
                        if (data != null) {
                            for (ServerListener sl : this.server.listeners) {
                                sl.getData(socket, data);
                            }
                        }
                    }
                }
            }
        }
    }
}
