    /*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.core.rpc;

import ogla.core.util.ArgumentType;
import ogla.core.util.Writer;
import java.io.BufferedReader;
import java.io.IOException;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author mmatula
 */
public class Rpc extends Callbacks {

    public Rpc() {
        this.addListener(eventListenerImpl);
    }

    public class Output {

        public int outargc = 0;
        public ByteBuffer data = null;
        public ArgumentType[] args = null;
    }

    private Map<String, List<RpcFunction>> functions = new HashMap<String, List<RpcFunction>>();
    private Map<Integer, RpcFunction> hashesFunctions = new HashMap<Integer, RpcFunction>();
    private Map<Integer, Socket> sockets = new HashMap<Integer, Socket>();
    private Map<Integer, Integer> clients = new HashMap<Integer, Integer>();
    private List<Integer> handlers = new ArrayList<Integer>();

    private final static int EXECUTE = 0;
    private final static int RETURN = 1;
    private final static int ESTABLISH_CONNECTION = 2;
    private final static int SET_CLIENT_ID = 3;
    private final static int GET_REGISTERED_FUNCTIONS = 4;
    private final static int SET_REGISTERED_FUNCTIONS = 5;
    private final static int MAGIC_NUMBER = 0xF98765;

    private static void send(Socket socket, Writer writer) throws IOException {
        byte[] sizeBytes = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(writer.size()).array();
        socket.getOutputStream().write(sizeBytes);
        socket.getOutputStream().write(writer.getBytes());
    }

    class FunctionInfoImpl extends FunctionInfo {

        private String[] names = null;

        private int connectionID;

        public FunctionInfoImpl(Object connection) {
            super(connection);
            this.names = null;
            this.connectionID = 0;
        }

        public FunctionInfoImpl(String name, ArgumentType[] inargs, ArgumentType[] outargs, Object connection) {
            super(name, inargs, outargs, connection);
        }

        public FunctionInfoImpl(RpcObject rpcObject, String name, ArgumentType[] inargs, ArgumentType[] outargs, Object connection) {
            super(rpcObject, name, inargs, outargs, connection);
        }

        public int getConnectionID() {
            return connectionID;
        }

        private void init(ByteBuffer Reader) {
            ArgumentType[] inArgs = null;
            ArgumentType[] outArgs = null;
            String[] names = null;
            String name = "";
            int bytes = Reader.getInt();
            int namesCount = Reader.getInt();
            if (namesCount > 1) {
                names = new String[namesCount - 1];
                for (int fa = 0; fa < namesCount - 1; fa++) {
                    names[fa] = Rpc.getString(Reader);
                }
            }
            name = Rpc.getString(Reader);
            final int inArgc = Reader.getInt();
            inArgs = new ArgumentType[inArgc];
            for (int fa = 0; fa < inArgc; fa++) {
                inArgs[fa] = (ArgumentType.fromInteger(Reader.getInt()));
            }
            final int outArgc = Reader.getInt();
            outArgs = new ArgumentType[outArgc];
            for (int fa = 0; fa < outArgc; fa++) {
                outArgs[fa] = (ArgumentType.fromInteger(Reader.getInt()));
            }
            this.setName(name);
            this.setObjectsNames(names);
            this.setInputArgs(inArgs);
            this.setOutputArgs(outArgs);
            this.connectionID = Reader.getInt();
        }
    }

    class ConnectionImpl implements Callbacks.Connection {

        private String address = "";
        private int port = -1;
        private int clientID = 0;

        public ConnectionImpl(String address, int port, int clientID) {
            this.address = address;
            this.port = port;
            this.clientID = clientID;
        }

        @Override
        public String getAddress() {
            return this.address;
        }

        @Override
        public int getPort() {
            return this.port;
        }

        @Override
        public int getClientID() {
            return this.clientID;
        }
    }

    class ResultImpl implements Callbacks.Result {

        int actionID;
        Object object;

        public ResultImpl(int actionID, Object object) {
            this.actionID = actionID;
            this.object = object;
        }

        @Override
        public Object getResult() {
            return object;
        }

        @Override
        public int getActionID() {
            return actionID;
        }
    }

    private ByteBuffer invokeFunction(List<String> names, int hashCode, List<ArgumentType> args, ByteBuffer byteBuffer) {
        synchronized (this) {
            ByteBuffer output = null;
            RpcFunction function = null;
            if (names != null) {
                String key = Rpc.createLinkedName(names);
                List<RpcFunction> functions = this.functions.get(key);
                if (functions != null) {
                    for (RpcFunction function1 : functions) {
                        ArgumentType[] args1 = new ArgumentType[function1.getInputArgumentsCount()];
                        args1 = function1.getInputArguments(args1);
                        if (args.equals(Arrays.asList(args1))) {
                            function = function1;
                            break;
                        }
                    }
                }
            } else {
                function = this.hashesFunctions.get(hashCode);
            }
            if (function != null) {
                output = function.invoke(byteBuffer);
            }
            return output;
        }
    }

    private static String createLinkedName(RpcFunction function) {
        String[] strs = new String[function.getObjectsNamesCount() + 1];
        strs = function.getObjectsNames(strs);
        strs[strs.length - 1] = function.getName();
        return Rpc.createLinkedName(Arrays.asList(strs));
    }

    private static String createLinkedName(List<String> names) {
        StringBuilder builder = new StringBuilder();
        for (int fa = 0; fa < names.size() - 1; fa++) {
            builder.append(names.get(fa));
            builder.append("::");
        }
        builder.append(names.get(names.size() - 1));
        return builder.toString();
    }

    public void registerFunction(RpcFunction function) {
        synchronized (this.functions) {
            String key = Rpc.createLinkedName(function);
            if (this.functions.get(key) == null) {
                this.functions.put(key, new ArrayList<RpcFunction>());
            }
            this.functions.get(key).add(function);
            this.hashesFunctions.put(function.hashCode(), function);
            this.invokeCallbacks(Events.EVENT_REGISTER_FUNCTION, function);
        }
    }

    public void init(int port) throws IOException {
        this.clientsBytes = new HashMap<Socket, DynamicBuffer>();
        this.server = new Server(port);
        this.server.addListener(serverListenerImpl);
    }

    public Object connect(String address, int port) throws IOException {
        Socket socket = new Socket(address, port);
        Writer writer = new Writer();
        int actionID = this.generateActionID();
        writer.write(ESTABLISH_CONNECTION);
        writer.write(actionID);
        writer.write(socket.hashCode());
        writer.write(this.server.getAddress());
        writer.write(this.server.getPort());
        Writer writer1 = new Writer();
        writer1.write(Rpc.MAGIC_NUMBER);
        writer1.write(writer);
        this.clients.put(port, socket.hashCode());
        Rpc.send(socket, writer1);
        server.connect();
        int remoteClientSocketID = (Integer) this.waitOnOutput(actionID);
        this.clients.put(socket.hashCode(), remoteClientSocketID);
        this.sockets.put(socket.hashCode(), socket);
        ConnectionImpl rcei = new ConnectionImpl(address, port, remoteClientSocketID);
        this.invokeCallbacks(Rpc.Events.EVENT_CONNECTED, rcei);
        return (Object) (socket);
    }
    /*
     public void getRegsiterredFunctions(Object connection) throws IOException {
     if ((connection instanceof Socket) == false) {
     return;
     }
     int actionID = this.generateActionID();
     Socket socket = (Socket) connection;
     Putter writer = new Putter();
     writer.write(GET_REGISTERED_FUNCTIONS);
     writer.write(actionID);
     writer.write(this.clients.get(connection.hashCode()));
     writer.write(0);
     socket.getOutputStream().write(writer.getBytesRef());
     }*/

    public void getRegisteredFunctionsLock(Object connection) throws IOException {
        if ((connection instanceof Socket) == false) {
            return;
        }
        int actionID = this.generateActionID();
        Socket socket = (Socket) connection;
        Writer writer = new Writer();
        Writer writer1 = new Writer();
        writer.write(GET_REGISTERED_FUNCTIONS);
        writer.write(actionID);
        writer.write(this.clients.get(connection.hashCode()));
        writer.write(0);
        writer1.write(Rpc.MAGIC_NUMBER);
        writer1.write(writer);
        Rpc.send(socket, writer1);
        Object result = this.waitOnOutput(actionID);
        if (result instanceof List) {
            List<FunctionInfoImpl> list = (List<FunctionInfoImpl>) result;
            for (FunctionInfoImpl functionInfoImpl : list) {
                this.invokeCallbacks(Events.EVENT_REGISTER_FUNCTION, functionInfoImpl);
            }
        }
    }

    private Object waitOnOutput(int actionID) {
        Entry entry = null;
        this.resultsLock.lock();
        try {
            entry = results.get(actionID);
        } finally {
            this.resultsLock.unlock();
        }
        if (entry != null) {
            entry.lock.lock();
            if (entry.object == null) {
                try {
                    entry.condtion.await();
                } catch (InterruptedException ie) {

                }
            }
            entry.lock.unlock();
        }
        this.releaseActionID(actionID);
        return entry.object;
    }

    private void notifyOutput(int actionID, Object output) {
        Entry entry = null;
        this.resultsLock.lock();
        try {
            entry = results.get(actionID);
        } finally {
            this.resultsLock.unlock();
        }
        if (entry != null) {
            entry.lock.lock();
            entry.object = output;
            entry.condtion.signalAll();
            entry.lock.unlock();
        }
    }

    private Server server = null;
    private BufferedReader bufferedReader = null;
    private Thread thread = null;
    private Thread thread1 = null;

    private Lock clientsBytesLock = new ReentrantLock();
    private Map<Socket, DynamicBuffer> clientsBytes = null;

    private class RunnableImpl implements Runnable {

        public void run() {
            try {
                while (!closed) {
                    Socket client = Rpc.this.server.connect();
                    synchronized (Rpc.this) {
                        Rpc.this.clientsBytes.put(client, null);
                    }
                }
            } catch (Exception e) {

            }
        }
    }

    private static String getString(ByteBuffer byteBuffer) {
        int charsCount = byteBuffer.getInt();
        StringBuilder builder = new StringBuilder();
        byte[] dst = new byte[charsCount];
        byteBuffer.get(dst);
        for (int fa = 0; fa < charsCount; fa++) {
            builder.append((char) dst[fa]);
        }
        return builder.toString();
    }

    private ServerListenerImpl serverListenerImpl = new ServerListenerImpl();

    private class ServerListenerImpl implements Server.ServerListener {

        public void getData(Socket client, byte[] bytes) {
            clientsBytesLock.lock();
            if (Rpc.this.clientsBytes.containsKey(client) == false) {
                Rpc.this.clientsBytes.put(client, new DynamicBuffer());
            }
            Rpc.this.clientsBytes.get(client).add((bytes));
            byte[] bytes1 = Rpc.this.clientsBytes.get(client).toArray();
            clientsBytesLock.unlock();
            ByteBuffer byteBuffer = ByteBuffer.wrap(bytes1).order(ByteOrder.LITTLE_ENDIAN);
            boolean canbeContinued = true;
            while (canbeContinued) {
                clientsBytesLock.lock();
                bytes1 = Rpc.this.clientsBytes.get(client).toArray();
                clientsBytesLock.unlock();
                if (bytes1 != null && bytes1.length > 0) {
                    byteBuffer = ByteBuffer.wrap(bytes1).order(ByteOrder.LITTLE_ENDIAN);
                    canbeContinued = Rpc.this.executeAction(byteBuffer, client);
                } else {
                    canbeContinued = false;
                }
            }
        }
    }

    public Writer invokeFunction(int actionID, int clientID, ByteBuffer input) {
        int functionID = input.getInt();
        List<String> names = null;
        int hashCode = -1;
        if (functionID == 0) {
            int namesCount = input.getInt();
            names = new ArrayList<String>();
            for (int fa = 0; fa < namesCount; fa++) {
                names.add(Rpc.this.getString(input));
            }
        } else {
            hashCode = input.getInt();
        }
        int inargc = input.getInt();
        List<ArgumentType> inargs = new ArrayList<ArgumentType>();
        for (int fa = 0; fa < inargc; fa++) {
            inargs.add(ArgumentType.fromInteger(input.getInt()));
        }
        int size = input.getInt();
        ByteBuffer output = Rpc.this.invokeFunction(names, hashCode, inargs, input);

        Writer writer = new Writer();
        writer.write(Rpc.RETURN);
        writer.write(actionID);
        Integer cId = clients.get(clientID);
        if (cId != null) {
            writer.write(cId);
        } else {
            writer.write(0);
        }
        writer.write(output);

        Writer writer1 = new Writer();
        writer1.write(Rpc.MAGIC_NUMBER);
        writer1.write(writer);
        Socket socket = sockets.get(clientID);
        try {
            if (socket != null) {
                Rpc.send(socket, writer1);
            }
        } catch (IOException ex) {
            Logger.getLogger(Rpc.class.getName()).log(Level.SEVERE, null, ex);
        }
        if (socket == null) {
            return writer1;
        }
        return null;
    }

    private void getReturn(int actionID, int clientID, ByteBuffer reader) {
        Rpc.this.invokeCallbacks(Events.EVENT_GET_RESULT, new ResultImpl(actionID, reader));
    }

    void establishConnection(int actionID, int remoteClientID, ByteBuffer reader) {
        Writer writer = new Writer();
        String address = Rpc.getString(reader);
        int port = reader.getInt();
        Socket socket = Rpc.this.establishConnection(address, port);
        writer.write(SET_CLIENT_ID);
        writer.write(actionID);
        writer.write(socket.hashCode());
        Writer writer1 = new Writer();
        writer1.write(Rpc.MAGIC_NUMBER);
        writer1.write(writer);
        if (socket != null) {
            try {
                socket.getOutputStream().write(writer1.getBytes());
            } catch (IOException ex) {
                Logger.getLogger(Rpc.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        ConnectionImpl rce = new ConnectionImpl(address, port, socket.hashCode());
        Rpc.this.invokeCallbacks(Rpc.Events.EVENT_CONNECTED, rce);
    }

    void setMyClientID(int actionID, int connectionID, ByteBuffer input) {
        Rpc.this.notifyOutput(actionID, connectionID);
    }

    void registerFunctionEvent(int actionID, int connectionID, ByteBuffer reader) {
        int functionsCount = reader.getInt();
        List<FunctionInfoImpl> functions = new ArrayList<FunctionInfoImpl>();
        for (int fa = 0; fa < functionsCount; fa++) {
            FunctionInfoImpl functionInfo = new FunctionInfoImpl(sockets.get(connectionID));
            functionInfo.init(reader);
            functions.add(functionInfo);
        }
        this.notifyOutput(actionID, functions);
    }

    public boolean executeAction(ByteBuffer byteBuffer, Socket client) {
        int magicNumber = byteBuffer.getInt();
        int size = byteBuffer.getInt();
        if (byteBuffer.capacity() - byteBuffer.position() >= size) {
            Rpc.this.executeAction(byteBuffer, null, magicNumber, size);
            clientsBytesLock.lock();
            Rpc.this.clientsBytes.get(client).cutTo(size + 8);
            clientsBytesLock.unlock();
            return true;
        } else {
            return false;
        }
    }

    public void executeAction(ByteBuffer byteBuffer, Writer[] writers, int magicNumber, int size) {
        try {
            int actionType = byteBuffer.getInt();
            int actionID = byteBuffer.getInt();
            int clientID = byteBuffer.getInt();
            //byteBuffer = ByteBuffer.wrap(byteBuffer.array(), byteBuffer.position(), byteBuffer.capacity() - byteBuffer.position());
            switch (actionType) {
                case Rpc.EXECUTE:
                    Writer writer = this.invokeFunction(actionID, clientID, byteBuffer);
                    if (writer != null && writers != null && writers.length > 0) {
                        writers[0] = writer;
                    }
                    break;
                case Rpc.RETURN:
                    this.getReturn(actionID, clientID, byteBuffer);
                    break;
                case Rpc.ESTABLISH_CONNECTION:
                    this.establishConnection(actionID, clientID, byteBuffer);
                    break;
                case SET_CLIENT_ID:
                    this.setMyClientID(actionID, clientID, byteBuffer);
                    break;
                case SET_REGISTERED_FUNCTIONS:
                    this.registerFunctionEvent(actionID, clientID, byteBuffer);
                    break;
            }
        } catch (Exception ex) {
            Logger.getLogger(Rpc.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private Socket establishConnection(String address, int port) {
        try {
            Socket client = new Socket(address, port);
            Rpc.this.sockets.put(client.hashCode(), client);
            return client;
        } catch (IOException e) {
            return null;
        }
    }

    private class Entry {

        public int size = 0;
        public Lock lock = new ReentrantLock();
        public Condition condtion = lock.newCondition();
        public Object object = null;
    }

    private Lock resultsLock = new ReentrantLock();
    private HashMap<Integer, Entry> results = new HashMap<Integer, Entry>();

    EventListenerImpl eventListenerImpl = new EventListenerImpl();

    class EventListenerImpl implements Callbacks.RpcListener {

        @Override
        public void wasRegisteredFunction(Object connection, String[] names, ArgumentType[] inargs, ArgumentType[] outargs) {
        }

        @Override
        public void wasUnregisteredFunction(Object connection, String[] names, ArgumentType[] inargs, ArgumentType[] outargs) {

        }

        @Override
        public void wasConnected(String address, int port, int clientID) {
        }

        @Override
        public void wasReturnedResult(Object result, int actionID) {
            Entry entry = null;
            Rpc.this.notifyOutput(actionID, result);
        }
    }

    public ByteBuffer call(String[] names, ByteBuffer input, ArgumentType[] inargs) throws IOException {
        return this.call(null, names, input, inargs);
    }

    public ByteBuffer call(Object connectionID, String[] names, ByteBuffer input, ArgumentType[] inargs) throws IOException {
        ByteBuffer output = null;
        int id = this.callAsync(connectionID, names, input, inargs);
        output = (ByteBuffer) this.waitOnOutput(id);
        return output;
    }

    public int callAsync(Object connectionID, String[] names, ByteBuffer input, ArgumentType[] inargs) throws IOException {
        Socket socket = null;
        int actionID = this.generateActionID();
        Writer writer = new Writer();
        writer.write(EXECUTE);
        writer.write(actionID);
        if (connectionID != null && this.clients.get(connectionID.hashCode()) != null) {
            writer.write(this.clients.get(connectionID.hashCode()));
        } else {
            writer.write(0);
        }
        int functionID = 0;
        if (functionID == 0) {
            writer.write(0);
            writer.write(names.length);
            for (int fa = 0; fa < names.length; fa++) {
                writer.write(names[fa]);
            }
        } else {
            writer.write(1);
            writer.write(functionID);
        }
        if (inargs != null && inargs.length > 0) {
            writer.write(inargs.length);
            for (int fa = 0; fa < inargs.length; fa++) {
                int argCode = ArgumentType.toInteger(inargs[fa]);
                writer.write(argCode);
            }
        } else {
            writer.write(0);
        }
        writer.write(input);
        Writer writer1 = new Writer();
        writer1.write(Rpc.MAGIC_NUMBER);
        writer1.write(writer);
        if (connectionID != null && connectionID instanceof Socket) {
            socket = (Socket) connectionID;
        }
        if (socket != null) {
            Rpc.send(socket, writer1);
        } else {
            final Writer writer2 = writer1;
            Thread async = new Thread(new Runnable() {
                @Override
                public void run() {
                    Writer[] writers = new Writer[1];
                    ByteBuffer byteBuffer = ByteBuffer.wrap(writer2.getBytes());
                    int magicNumber = byteBuffer.order(ByteOrder.LITTLE_ENDIAN).getInt();
                    int size = byteBuffer.order(ByteOrder.LITTLE_ENDIAN).getInt();
                    Rpc.this.executeAction(byteBuffer.order(ByteOrder.LITTLE_ENDIAN), writers, magicNumber, size);
                    byteBuffer = ByteBuffer.wrap(writers[0].getBytes());
                    magicNumber = byteBuffer.order(ByteOrder.LITTLE_ENDIAN).getInt();
                    size = byteBuffer.order(ByteOrder.LITTLE_ENDIAN).getInt();
                    Rpc.this.executeAction(byteBuffer.order(ByteOrder.LITTLE_ENDIAN), null, magicNumber, size);
                }
            });
            async.start();
        }
        return actionID;
    }

    boolean closed = false;

    private int generateActionID() {
        int actionId = 0;
        synchronized (handlers) {
            Entry entry = new Entry();

            if (handlers.size() != 0 && handlers.size() < handlers.get(handlers.size() - 1)) {
                for (int fa = 0; fa < handlers.size(); fa++) {
                    if (handlers.contains(fa) == false) {
                        actionId = fa;
                        handlers.set(fa, fa);
                    }
                }
            } else {

                actionId = handlers.size();
                handlers.add(actionId);
                resultsLock.lock();
                this.results.put(actionId, entry);
                resultsLock.unlock();
            }
        }
        return actionId;
    }

    private void releaseActionID(int id) {
        synchronized (handlers) {
            handlers.remove(id);
        }
    }

    public void close() throws IOException {
        closed = true;
        this.bufferedReader.close();
        this.server.close();
    }
}
