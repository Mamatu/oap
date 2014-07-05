/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package orgla.rpc.test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.logging.Level;
import java.util.logging.Logger;
import ogla.core.rpc.Callbacks;
import ogla.core.rpc.Rpc;
import ogla.core.util.ArgumentType;

/**
 *
 * @author mmatula
 */
public class main {

    static class Listener implements Callbacks.RpcListener {

        @Override
        public void wasConnected(String address, int port, int clientID) {
        }

        @Override
        public void wasReturnedResult(Object result, int resultID) {
        }

        @Override
        public void wasRegisteredFunction(Object connection, String[] names, ArgumentType[] inargs, ArgumentType[] outargs) {
            System.out.println(names[names.length - 1]);
            System.out.println(inargs.length);
            System.out.println(outargs.length);
        }

        @Override
        public void wasUnregisteredFunction(Object connection, String[] names, ArgumentType[] inargs, ArgumentType[] outargs) {
        }
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] aaa) {
        try {
            Rpc rpc = new Rpc();
            rpc.init(5001);
            rpc.addListener(new Listener());
            Object connection = rpc.connect("127.0.0.1", 5000);
            String[] names = {new String("TestFunction")};
            ArgumentType[] args = {ArgumentType.ARGUMENT_TYPE_FLOAT};
            ByteBuffer bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(10.41f);
            ByteBuffer output = rpc.call(connection, names, bb, args);
            System.out.println(output.getFloat());
        } catch (Exception ie) {
            Logger.getLogger(main.class.getName()).log(Level.SEVERE, null, ie);
        }
    }

}
