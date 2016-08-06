/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.groovy;

import groovy.lang.Binding;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Level;
import java.util.logging.Logger;
import ogla.core.rpc.Callbacks;
import ogla.core.rpc.Rpc;
import ogla.core.rpc.RpcFunction;
import ogla.core.rpc.RpcObject;
import ogla.core.ui.OglaMethod;
import ogla.core.ui.OglaObject;
import ogla.core.util.ArgumentType;
import ogla.core.util.ArgumentsUtils;
import ogla.core.util.Writer;
import org.codehaus.groovy.runtime.HandleMetaClass;
import org.codehaus.groovy.runtime.InvokerHelper;

/**
 *
 * @author mmatula
 */
public class ObjectsManager {

    public static class RpcMethod extends OglaMethod {

        protected Rpc rpc = null;

        public RpcMethod(String[] names, ArgumentType[] inargs, ArgumentType[] outargs, Rpc rpc) {
            super(names, inargs, outargs);
            this.rpc = rpc;
        }

        public RpcMethod(String[] names, ArgumentType[] inargs, Rpc rpc) {
            super(names, inargs);
            this.rpc = rpc;
        }

        public Object invoke(Object args, OglaObject groovyObjectImpl, Object userData) {
            Object connection = userData;
            ByteBuffer input = null;
            if (args != null && ((Object[]) args).length > 0) {
                Writer writer = new Writer();
                ArgumentsUtils.convertObject(writer, args, this.getInputArgsRef());
                input = ByteBuffer.allocate(writer.getBytes().length).put(writer.getBytes());
            }
            try {
                String[] names = this.getNamesRef();
                if (connection != null && ObjectsManager.connectionsNames.containsKey(connection)) {
                    String name = ObjectsManager.connectionsNames.get(connection);
                    if (this.getNamesRef()[0].equals(name)) {
                        List<String> list = new ArrayList<String>();
                        list.addAll(Arrays.asList(names));
                        list.remove(0);
                        names = new String[list.size()];
                        names = list.toArray(names);
                    }
                }
                ByteBuffer output = this.rpc.call(connection, names, input, this.getInputArgsRef());
                return output;
            } catch (IOException ex) {
                Logger.getLogger(RpcMethod.class.getName()).log(Level.SEVERE, null, ex);
                return Integer.valueOf(-1);
            }
        }
    }

    private static Lock lock = new ReentrantLock();
    private static Map<List<String>, OglaObject> namesObjects = Collections.synchronizedMap(new HashMap<List<String>, OglaObject>());
    static List<OglaObject> objects = Collections.synchronizedList(new ArrayList<OglaObject>());
    private static Map<Object, String> connectionsNames = Collections.synchronizedMap(new HashMap<Object, String>());

    private static class RpcListenerImpl implements Callbacks.RpcListener {

        private Binding binding;
        private Rpc rpc = null;

        public RpcListenerImpl(Binding binding, Rpc rpc) {
            this.binding = binding;
            this.rpc = rpc;
        }

        private static void insert(List<String> list, String name) {
            int size = list.size();
            String n = name;
            for (int fa = 0; fa < size; fa++) {
                n = list.set(fa, n);
            }
            list.add(n);
        }

        private String[] getNames(String name, String[] names) {
            List<String> list = new ArrayList<String>();
            list.addAll(Arrays.asList(names));
            insert(list, name);
            String[] newNames = new String[names.length + 1];
            return list.toArray(newNames);
        }

        public void wasRegisteredFunction(Object connection, String[] names, ArgumentType[] inargs, ArgumentType[] outargs) {
            ObjectsManager.lock.lock();
            try {
                String name = ObjectsManager.connectionsNames.get(connection);
                if (name != null && name.length() > 0) {
                    names = this.getNames(name, names);
                }
                List<String> objNames = new ArrayList<String>();
                objNames.addAll(Arrays.asList(names));
                objNames.remove(objNames.size() - 1);
                OglaObject ogla = ObjectsManager.namesObjects.get(objNames);
                if (ogla != null) {
                    if (ogla.isMethod(names[names.length - 1], inargs) == false) {
                        ObjectsManager.RpcMethod method = new ObjectsManager.RpcMethod(names, inargs, outargs, rpc);
                        ogla.addMethod(method, connection);
                    }
                } else {
                    String[] names1 = new String[objNames.size()];
                    ObjectsManager.RpcMethod method = new ObjectsManager.RpcMethod(names, inargs, outargs, rpc);
                    ogla = ObjectsManager.registerObjects(objNames.toArray(names1), this.binding);
                    ogla.addMethod(method, connection);
                }
            } finally {
                ObjectsManager.lock.unlock();
            }
        }

        public void wasUnregisteredFunction(Object connection, String[] names, ArgumentType[] inargs, ArgumentType[] outargs) {
        }

        public void wasConnected(String address, int port, int clientID) {
        }

        public void wasReturnedResult(Object result, int resultID) {
        }
    }

    private static Rpc rpc = null;
    private static Binding binding = null;

    public static void init(Binding binding) {
        ObjectsManager.rpc = new Rpc();
        ObjectsManager.rpc.addListener(new RpcListenerImpl(binding, rpc));
        ObjectsManager.binding = binding;
    }

    static Rpc getRpc() {
        return ObjectsManager.rpc;
    }

    public static List<OglaObject> getOglaObjects() {
        List<OglaObject> out = new ArrayList<OglaObject>();
        out.addAll(objects);
        return out;
    }

    public static void connect(String address, int port, String variableName) {
        ObjectsManager.lock.lock();
        try {
            Object connection = ObjectsManager.rpc.connect(address, port);
            ObjectsManager.connectionsNames.put(connection, variableName);
            String[] names = {variableName};
            ObjectsManager.registerObjects(names, binding);
            ObjectsManager.rpc.getRegisteredFunctionsLock(connection);
        } catch (IOException ex) {
            Logger.getLogger(ObjectsManager.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            ObjectsManager.lock.unlock();
        }
    }

    private static OglaObject registerObjects(String[] names, Binding binding) {
        lock.lock();
        OglaObject root = null;
        List<OglaObject> roots = new ArrayList<OglaObject>();
        try {
            List<String> objNames = new ArrayList<String>();
            for (String name : names) {
                objNames.add(name);
                OglaObject oglaObject = ObjectsManager.namesObjects.get(objNames);
                if (oglaObject == null) {
                    List<OglaObject> roots1 = new ArrayList<OglaObject>();
                    roots1.addAll(roots);
                    oglaObject = new OglaObject(name, root, roots1);
                    ObjectsManager.namesObjects.put(objNames, oglaObject);
                    if (root != null) {
                        String[] names1 = new String[objNames.size()];
                        names1 = objNames.toArray(names1);
                        OglaObjectWrapper oglaObjectWrapper = new OglaObjectWrapper(oglaObject);
                        OglaObjectWrapper oglaObjectRootWrapper = new OglaObjectWrapper(root);
                        HandleMetaClass hmc = new HandleMetaClass(InvokerHelper.getMetaClass(oglaObjectRootWrapper), oglaObjectRootWrapper);
                        hmc.setProperty(names1[names1.length - 1], oglaObjectWrapper);
                    } else {
                        binding.setVariable(objNames.get(0), oglaObject);
                    }
                }
                root = oglaObject;
                roots.add(root);
            }
        } finally {
            lock.unlock();
        }
        return root;
    }
    
    public static void registerObject(OglaObject oglaObject) {
        OglaObjectWrapper oglaObjectWrapper = new OglaObjectWrapper(oglaObject);
        OglaObjectWrapper oglaObjectRootWrapper = new OglaObjectWrapper(OglaObject.getFirstRoot(oglaObject));
        HandleMetaClass hmc = new HandleMetaClass(InvokerHelper.getMetaClass(oglaObjectRootWrapper), oglaObjectRootWrapper);
        hmc.setProperty(oglaObject.getName(), oglaObjectWrapper);
    }

    public static OglaObject get(String[] names) {
        List<String> names1 = Arrays.asList(names);
        OglaObject oglaObject = namesObjects.get(names1);
        return oglaObject;
    }

    public static OglaObject create(String[] names, Binding binding) {
        if (ObjectsManager.get(names) == null) {
            return ObjectsManager.registerObjects(names, binding);
        }
        return null;
    }

    public static OglaObject create(List<String> names, Binding binding) {
        String[] names1 = new String[names.size()];
        names1 = names.toArray(names1);
        return ObjectsManager.create(names1, binding);
    }

    public static Object getConection(OglaObject oglaObject) {
        if (oglaObject.getRoot() == null) {
            ObjectsManager.lock.lock();
            try {
                String name = oglaObject.getName();
                for (Map.Entry< Object, String> entry : ObjectsManager.connectionsNames.entrySet()) {
                    if (entry.getValue().equals(name)) {
                        return entry.getKey();
                    }
                }
            } finally {
                ObjectsManager.lock.unlock();
            }
            return null;
        } else {
            return getConection(oglaObject.getRoot());
        }
    }

    static class RpcObjectImpl implements RpcObject {

        public RpcObjectImpl(Binding binding) {
            String[] names = {"rpc"};
            OglaObject oglaObject = ObjectsManager.create(names, binding);
            ObjectsManager.getRpc().registerFunction(this.new ShowObjectsList());
            ObjectsManager.getRpc().registerFunction(this.new ShowObjectsDetailedList());
            oglaObject.addMethod(this.new Connect());
        }

        public String getName() {
            return "rpc";
        }

        public RpcObject getRoot() {
            return null;
        }

        private static String getString(ByteBuffer input) {
            int len = input.getInt();
            byte[] chars = new byte[len];
            input.get(chars);
            StringBuilder builder = new StringBuilder();
            for (int fa = 0; fa < len; fa++) {
                builder.append((char) chars[fa]);
            }
            return builder.toString();
        }

        class Connect extends OglaMethod {

            private Rpc rpc = null;
            private String name = "connect";
            List<Object> connections = new ArrayList<Object>();

            public Connect() {
                super();
                this.setName(name);
                this.setObjects(RpcObjectImpl.this);
                this.setInputArgs(input);
                this.setOutputArgs(output);
            }

            private ArgumentType[] input = new ArgumentType[]{ArgumentType.ARGUMENT_TYPE_STRING, ArgumentType.ARGUMENT_TYPE_INT, ArgumentType.ARGUMENT_TYPE_STRING};
            private ArgumentType[] output = new ArgumentType[]{ArgumentType.ARGUMENT_TYPE_BOOL};

            public Object invoke(Object args, OglaObject oglaObject, Object userData) {
                Object[] inputArray = (Object[]) input;
                String address = String.valueOf(inputArray[0]);
                int port = (Integer) inputArray[1];
                ObjectsManager.connect(address, port, name);
                return null;
            }

        }

        class ShowObjectsList extends RpcFunction {

            private String name = "showObjects";

            public ShowObjectsList() {
                super();
                this.setName(name);
                this.setObjects(RpcObjectImpl.this);
                this.setInputArgs(input);
                this.setOutputArgs(output);
            }

            @Override
            public ByteBuffer invoke(ByteBuffer input) {
                StringBuilder builder = new StringBuilder();
                List<OglaObject> list = ObjectsManager.getOglaObjects();
                for (OglaObject oglaObject : list) {
                    builder.append(oglaObject.getFullName());
                    builder.append("\n");
                }
                String info = builder.toString();
                System.out.print(info);
                return null;
            }

            private ArgumentType[] input = null;
            private ArgumentType[] output = null;
        }

        class ShowObjectsDetailedList extends RpcFunction {

            private String name = "show";

            public ShowObjectsDetailedList() {
                super();
                this.setName(name);
                this.setObjects(RpcObjectImpl.this);
                this.setInputArgs(input);
                this.setOutputArgs(output);
            }

            @Override
            public ByteBuffer invoke(ByteBuffer input) {
                StringBuilder builder = new StringBuilder();
                List<OglaObject> list = ObjectsManager.getOglaObjects();
                for (OglaObject oglaObject : list) {
                    builder.append(oglaObject.getTextWithChildren());
                    builder.append("\n");
                }
                String info = builder.toString();
                System.out.print(info);
                return null;
            }
            private ArgumentType[] input = null;
            private ArgumentType[] output = null;
        }
    }
}
