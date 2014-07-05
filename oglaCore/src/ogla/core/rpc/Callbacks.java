/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.core.rpc;

import ogla.core.util.ArgumentType;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 *
 * @author mmatula
 */
public class Callbacks {

    public interface Connection {

        public String getAddress();

        public int getPort();

        public int getClientID();
    }

    public interface Result {

        public Object getResult();

        public int getActionID();
    }

    public enum Events {

        EVENT_REGISTER_FUNCTION,
        EVENT_CONNECTED,
        EVENT_GET_RESULT
    }

    public interface RpcListener {

        public void wasRegisteredFunction(Object connection, String[] names, ArgumentType[] inargs, ArgumentType[] outargs);

        public void wasUnregisteredFunction(Object connection, String[] names, ArgumentType[] inargs, ArgumentType[] outargs);

        public void wasConnected(String address, int port, int clientID);

        public void wasReturnedResult(Object result, int resultID);
    }

    private List<RpcListener> lisnteners = new ArrayList<RpcListener>();

    public void addListener(RpcListener listener) {
        if (listener != null) {
            this.lisnteners.add(listener);
        }
    }

    public void removeListener(RpcListener listener) {
        if (listener != null) {
            this.lisnteners.remove(listener);
        }
    }

    protected void invokeCallbacks(Callbacks.Events event, Object object) {
        if (event == Callbacks.Events.EVENT_CONNECTED) {
            Connection connection = (Connection) object;
            for (RpcListener l : lisnteners) {
                l.wasConnected(connection.getAddress(), connection.getPort(), connection.getClientID());
            }
        } else if (event == Callbacks.Events.EVENT_REGISTER_FUNCTION) {
            FunctionInfo functionInfo = (FunctionInfo) object;
            for (RpcListener l : lisnteners) {
                ArgumentType[] input = null;
                ArgumentType[] output = null;
                if (functionInfo.getInputArgumentsCount() > 0) {
                    input = new ArgumentType[functionInfo.getInputArgumentsCount()];

                }
                if (functionInfo.getOutputArgumentsCount() > 0) {
                    output = new ArgumentType[functionInfo.getOutputArgumentsCount()];
                }
                String[] names = new String[functionInfo.getObjectsNamesCount() + 1];
                names = functionInfo.getObjectsNames(names);
                names[names.length - 1] = functionInfo.getName();
                l.wasRegisteredFunction(functionInfo.getConnection(), names, functionInfo.getInputArguments(input), functionInfo.getOutputArguments(output));
            }

        } else if (event == Callbacks.Events.EVENT_GET_RESULT) {
            Result result = (Result) object;
            for (RpcListener l : lisnteners) {
                l.wasReturnedResult(result.getResult(), result.getActionID());
            }
        }
    }
}
